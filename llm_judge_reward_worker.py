from typing import Optional, Union, Dict, List, Any
import json
import re
import torch
import requests
import time
import traceback
import numpy as np
from functools import partial
import tensordict
from tensordict import TensorDict
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider, default_reward_model_provider
from roll.platforms import current_platform
from roll.utils.logging import get_logger
from roll.utils.context_managers import state_offload_manger
from roll.utils.prompt import *
from roll.datasets.chat_template import get_chat_template


class LLMJudgeRewardWorker(Worker):
    """
    Reward Worker that uses LLM-as-judge to compute rewards.
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        # LLM judge相关配置
        self.judge_prompt = self.worker_config.judge_prompt if hasattr(self.worker_config, "judge_prompt") else None
        # self.judge_prompt = prompt_maps[self.judge_prompt]
        self.judge_model_type = (
            self.worker_config.judge_model_type if hasattr(self.worker_config, "judge_model_type") else "api"
        )
        self.judge_model_name = (
            self.worker_config.judge_model_name if hasattr(self.worker_config, "judge_model_name") else None
        )
        self.judge_api_url = self.worker_config.judge_api_url if hasattr(self.worker_config, "judge_api_url") else None
        self.judge_api_key = self.worker_config.judge_api_key if hasattr(self.worker_config, "judge_api_key") else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.actor_tokenizer = default_tokenizer_provider(pipeline_config.actor_train.model_args)
        if self.judge_model_type == "api":
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            print(f"{self.worker_name} initialized with API model")

        elif self.judge_model_type == "inference":
            self.strategy = create_strategy(worker=self)
            self.strategy.initialize(model_provider=default_reward_model_provider)
            self.tokenizer = self.strategy.tokenizer
            print(f"{self.worker_name} initialized with inference model")
            self.strategy.offload_states()
            current_platform.init()
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

    def _call_api_model(self, messages: Dict, retry_times=3) -> str:
        from openai import OpenAI

        ouput = ""
        if not self.judge_api_url or not self.judge_api_key:
            raise ValueError("API URL and API key must be provided for API model type")
        while retry_times > 0:
            retry_times -= 1
            try:
                client = OpenAI(
                    api_key=self.judge_api_key,
                    base_url=self.judge_api_url,
                )
                completion = client.chat.completions.create(model=self.judge_model_name, messages=messages)
                output = completion.choices[0].message.content
                total_tokens = completion.usage.total_tokens
                prompt_token = completion.usage.prompt_tokens
                completion_token = completion.usage.completion_tokens
                token_info = {
                    "total_tokens": total_tokens,
                    "prompt_token": prompt_token,
                    "completion_token": completion_token,
                }
                print(token_info)
                if output != None and output != "":
                    break
            except Exception as e:
                print(e)
                continue
        self.logger.info(f"judge model api output: {str(output)}")
        return output

    # def _run_local_inference(self, messages: Dict) -> str:
    #     if not self.strategy:
    #         raise ValueError("Strategy not initialized for local inference")

    #     template_name = self.worker_config.data_args.template
    #     chat_template_func = get_chat_template(template_name, self.tokenizer)
    #     text = chat_template_func(messages)
    #     # self.logger.info(f"**text input**: {str(text)}")
    #     tokenized = self.tokenizer(text, return_tensors="pt")
    #     input_ids = tokenized["input_ids"].to(current_platform.device_type)
    #     attention_mask = tokenized["attention_mask"].to(current_platform.device_type)

    #     generation_config = self.worker_config.generating_args.to_dict()
    #     generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
    #     generation_config["pad_token_id"] = self.tokenizer.pad_token_id

    #     data = DataProto(
    #         batch=TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=input_ids.shape[0])
    #     )
    #     data = data.to(current_platform.device_type)
    #     data.meta_info = {"micro_batch_size": self.worker_config.infer_batch_size}

    #     with torch.no_grad():
    #         output = self.strategy.generate(batch=data, generation_config=generation_config)
    #         if isinstance(output, torch.Tensor):
    #             generate_ids = output[:, len(input_ids[0]) :]
    #         else:
    #             generate_ids = output.batch["input_ids"][:, len(input_ids[0]) :]

    #     output = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    #     self.logger.info(f"judge model inference output: {str(output)}")
    #     return output.strip()
    def _run_local_inference(self, messages: Dict) -> str:
        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")
        template_name = self.worker_config.data_args.template
        chat_template_func = get_chat_template(template_name, self.tokenizer)
        text = chat_template_func(messages)
        
        tokenized = self.tokenizer(text, return_tensors="pt")
        # 确保到正确的 device (cuda)
        input_ids = tokenized["input_ids"].to(current_platform.device_type)
        attention_mask = tokenized["attention_mask"].to(current_platform.device_type)

        # 1. 获取基础配置并注入控制参数
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        
        # 【核心修改点】：开启分数输出和返回字典格式
        generation_config["output_scores"] = True
        generation_config["return_dict_in_generate"] = True
        # 即使使用了 vLLM 或其他，通常设置这个能让底层知道需要 logits

        data = DataProto(
            batch=TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=input_ids.shape[0])
        )
        data = data.to(current_platform.device_type)
        data.meta_info = {"micro_batch_size": self.worker_config.infer_batch_size}

        # 2. 准备 1, 2, 3, 4 的 Token ID 用于概率提取
        response_tokens = ["1", "2", "3", "4"]
        # 使用 add_special_tokens=False 保证拿到的就是数字本身的 ID
        response_ids = [self.tokenizer.encode(t, add_special_tokens=False)[-1] for t in response_tokens]

        with torch.no_grad():
            # 执行生成。此时返回的可能是 DataProto 或 HF 的 ModelOutput 结构
            output_obj = self.strategy.generate(batch=data, generation_config=generation_config)
            
            # ROLL 的 strategy.generate 行为取决于底层实现：
            # 如果是 hf_infer，且设置了 return_dict_in_generate=True
            # 结果通常在 output_obj 中
            if hasattr(output_obj, "sequences"):
                # 这种是标准的 HF ModelOutput 格式
                all_ids = output_obj.sequences
                # scores 是一个 tuple，对应生成的每一个 token 的 logits
                scores = output_obj.scores 
            elif isinstance(output_obj, DataProto):
                # 这种是 ROLL 包装后的格式，检查 meta_info 或 batch 里有没有 scores
                all_ids = output_obj.batch["input_ids"]
                scores = output_obj.meta_info.get("scores", None)
            else:
                # 回退方案
                all_ids = output_obj
                scores = None

            # 提取生成部分的 ID (去掉 prompt 部分)
            prompt_len = input_ids.shape[1]
            generate_ids = all_ids[:, prompt_len:]

        # 3. 解码生成的文本
        decoded_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True).strip()
        # 4. 提取第一个 token 在 1,2,3,4 上的归一化概率
        if scores is not None and len(scores) > 0:
            # 取生成的第一个 token 对应的 logits: (batch_size, vocab_size)
            # 针对 batch[0]
            first_token_logits = scores[0][0] 
            
            # 提取 1, 2, 3, 4 的 logits
            target_logits = first_token_logits[response_ids]
            # 计算归一化概率
            probs = torch.softmax(target_logits.float(), dim=-1).cpu().tolist()
            # 实际生成的第一个字符
            actual_first_token = self.tokenizer.decode(generate_ids[0, 0:1], skip_special_tokens=True).strip()
            # 构建 logits_token_str (仿照 HuggingFaceGenerateEngine 格式)
            if actual_first_token in response_tokens:
                idx = response_tokens.index(actual_first_token)
                logits_token_str = f"{actual_first_token}\t{probs[idx]}"
            else:
                logits_token_str = f"{actual_first_token}\t-1"

            # 构建 1-4 完整的概率分布字符串
            logits_str = "\n".join([f"{t}\t{p}" for t, p in zip(response_tokens, probs)])
            
            # 拼接最终结果
            final_res = f"{decoded_text}\n{logits_token_str}\n{logits_str}"
        else:
            # 兜底：如果没拿到概率，返回原文本
            final_res = decoded_text

        self.logger.info(f"judge model inference output:\n{final_res}")
        return final_res

    def _extract_score(self, response: str) -> float:
        try:
            match = re.search("Score: ([0-9.]+)", response)
            if match:
                score = float(match.group(1))
                normalized_score = score / 10
                return normalized_score
            else:
                self.logger.warning(f"Could not extract score from response: {response}")
                return 0.5
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return 0.5

    def _extract_score_v2(self, response: str) -> float:
        response = response.lower()
        try:
            if "yes" in response:
                return 1
            elif "no" in response:
                return 0
            else:
                self.logger.warning(f"Could not extract score from response: {response}")
                return 0
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return 0
    # 1. 提取档位逻辑保持不变
    def _extract_correlation_score(self, response: str):
        response = response.lower()
        if not response:
            return None
        match = re.search(r'^(\d+)', response.strip())
        if match:
            return int(match.group(1))
        return None

    def _format_judge_prompt(self, prompt: str, response: str, reference: str = None) -> str:
        if "user\n" in prompt:
            prompt = prompt.split("user\n")[-1].strip()
        if not self.judge_prompt:
            title, label_answer, wo_feed_answer = reference.split('<splite_token>')
            formatted_prompt = f"""
            {title}\n评论信息：{response}
            """
        else:
            formatted_prompt = self.judge_prompt.format(question=prompt, response=response, reference=reference)
        messages = [{"role": "user", "content": formatted_prompt}]
        return messages
        
    # 2. 修改后的 Reward 计算逻辑
    def _get_reward(self, current_probs, label_answer, baseline_probs, response_text):
        """
        current_probs: dict, e.g., {"1": 0.01, "2": 0.04, "3": 0.1, "4": 0.85}
        label_answer: int (GT)
        baseline_probs: dict, 同上
        response_text: 生成的摘要
        """
        try:
            gt = int(label_answer)
            
            # 计算期望误差的辅助函数
            def calc_expected_error(p_dict, target):
                # 确保 p_dict 包含 1,2,3,4
                error = 0.0
                for grade_str, prob in p_dict.items():
                    grade_int = int(grade_str)
                    error += prob * abs(grade_int - target)
                return error

            # 计算当前模型的期望误差
            e_error_current = calc_expected_error(current_probs, gt)
            # 计算 Baseline 的期望误差
            e_error_baseline = calc_expected_error(baseline_probs, gt)

            # 基础奖励：Baseline误差 - 当前误差
            base_reward = float(e_error_baseline - e_error_current)

        except (ValueError, TypeError, KeyError):
            return -2.0

        # --- 长度惩罚 (保持不变) ---
        actual_length = len(response_text)
        length_threshold = 128
        penalty_weight = 0.1 
        
        lp_penalty = 0.0
        if actual_length > length_threshold:
            lp_penalty = (actual_length - length_threshold) * penalty_weight

        return base_reward - lp_penalty


    # 3. 修改后的评判主逻辑
    def _get_llm_judgment(self, prompt_id: str, prompt: str, response: str, reference: str = None) -> float:
        messages = self._format_judge_prompt(prompt, response, reference)

        if self.judge_model_type == "api":
            # API 模式不支持期望奖励（除非 API 能返回 logprobs）
            llm_response = self._call_api_model(messages)
            current_probs = None
        elif self.judge_model_type == "inference":
            llm_response_raw = self._run_local_inference(messages)
            
            # --- 解析多行输出 ---
            lines = llm_response_raw.split('\n')
            llm_response = lines[0] # 第一行文本
            
            current_probs = {}
            # 从最后 4 行解析概率 (1\tprob, 2\tprob ...)
            if len(lines) >= 6: 
                for prob_line in lines[-4:]:
                    parts = prob_line.split('\t')
                    if len(parts) == 2:
                        current_probs[parts[0]] = float(parts[1])
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

        # --- 解析 Reference 中的信息 ---
        try:
            parts = reference.split('<splite_token>')
            title = parts[0]
            label_answer = int(parts[1]) # GT
            
            # 解析 Baseline 概率字符串 "0.01,0.02,0.07,0.90"
            b_probs_list = [float(x) for x in parts[2].split(',')]
            baseline_probs = {
                "1": b_probs_list[0],
                "2": b_probs_list[1],
                "3": b_probs_list[2],
                "4": b_probs_list[3]
            }
        except Exception as e:
            self.logger.error(f"Reference parse error: {e}")
            return -1.0, {"error": "ref format error"}

        # --- 计算 Reward ---
        if not current_probs:
            # 如果没拿到概率，回退到硬档位计算或给予固定惩罚
            predict_score = self._extract_correlation_score(llm_response)
            if predict_score is None: return -1.5, {}
            # 伪造一个 1.0 的概率分布用于计算
            current_probs = {str(i): 1.0 if i == predict_score else 0.0 for i in range(1, 5)}

        reward_score = self._get_reward(current_probs, label_answer, baseline_probs, response)

        info = {
            "prompt_id": prompt_id,
            "reference": reference,
            "review": response,
            "score": reward_score,
            "current_probs": current_probs,
            "baseline_probs": baseline_probs,
            "gt": label_answer,
            "raw_pred": max(baseline_probs, key=baseline_probs.get),
            "llm_text": llm_response,
            "response_length": len(response)
        }
        return reward_score, info



    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}

        if self.judge_model_type == "inference" and self.strategy:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/compute_rewards",
                is_offload_states=is_offload_states,
            ):
                return self._compute_rewards_impl(data, metrics)
        else:
            return self._compute_rewards_impl(data, metrics)

    def _compute_rewards_impl(self, data: DataProto, metrics: Dict):
        prompts_text_list = self.actor_tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.actor_tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

        scores = []
        for prompt_id, prompt_txt, response, reference in zip(
            data.non_tensor_batch["id"], prompts_text_list, response_text_list, data.non_tensor_batch["ground_truth"]
        ):
            score, info = self._get_llm_judgment(prompt_id, prompt_txt, response, reference)
            scores.append(score)
            self.logger.info(f"{json.dumps(info, ensure_ascii=False)}")

        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = scores_tensor

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
            }
        )

        output.meta_info = {"metrics": metrics}
        print(f"Computed rewards for {len(scores)} samples")
        return output
