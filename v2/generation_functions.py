from typing import Callable, Optional, Union
import torch
import types
from transformers.utils import auto_docstring, logging
from transformers import AutoTokenizer

# Constants for Fast_dLLM model
FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645

MASK_COLOR = 0.5
TOKEN_COLOR = -0.5

# 定义相关的dLLM类，以注入相关的生成方法
# 为 Qwen 类 CausalLM 注入“掩码扩展 + 分块解码”的自定义生成逻辑
@auto_docstring
class Fast_dLLM_QwenForCausalLM:

    # 进行batch推理
    @torch.no_grad()
    def batch_sample(
        self,
        input_ids: torch.Tensor,
        tokenizer: AutoTokenizer,
        # 用于控制block size
        block_size: int,
        # 生成的长度
        max_new_tokens: int,
        small_block_size: int,
        # 输入序列中的最小长度
        min_len: int,
        # 代表了输入的请求的长度集合
        seq_len: torch.Tensor,
        # 掩码ID
        mask_id: int=151665,
        # 采样过程中的置信度
        threshold: float=0.95,
        # 停止token
        stop_token: int=151645,
        # 是否使用block cache
        use_block_cache: bool=False,
        # 采样过程中的top-p
        top_p: float=0.95,
        # 采样过程中的温度
        temperature: float=0.0,
    ):
        # 生成的block数量 + 输入对其的block数量
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        # 批量大小 即为 请求的数量大小
        batch_size = input_ids.shape[0]

        # == 1. 首先处理全局cache：past_key_values 保存到目前为止的上下文
        # 如果输入的序列的最小长度都比一个block大，将prompt部分转换为cache
        if min_len > block_size:
            output = self.forward(
                input_ids=input_ids[:, :(min_len // block_size * block_size)],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size
            )
            logits, past_key_values = output.logits, output.past_key_values
            # 当prompt完全填满block时，立即预测下一个token
            # 减少需要去掩码的位置数量
            if min_len % block_size == 0:
                # 进行采样的id
                predict_sample_idx = (seq_len == min_len)
                # 获取下一个token的logits
                predict_logits = logits[predict_sample_idx, -1:, :]
                # 获取下一个token的id
                next_token = predict_logits.argmax(dim=-1)
                # 将下一个token添加到输入序列中
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        # 用于记录哪些样本已经完成生成，创建batch个
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        # 用于记录哪些样本需要进行采样，创建batch个
        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        # 进行逐个block生成，在每个block内部进行diffusion的采样
        for block_idx in range(start_block_idx, num_blocks):
            # 全部结束就结束
            if finished_flag.all():
                break

            # 检查是否所有样本都在同一个 block 位置
            if (seq_block_idx == block_idx).all():
                # 计算需要填充的mask数量
                x_init = mask_id * torch.ones((input_ids.shape[0], block_size-input_ids.shape[1]%block_size), device=self.device, dtype=torch.long)
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1)*block_size]

            # 处理已经完成的样本
            # 将已完成样本的当前 block 填充为 pad token
            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            # 复制一份用于 MDM 采样
            x_t = x_init.clone()
            step = 0
            block_past_key_values = None

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # 没有掩码了，说明当前 block 已经完成
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx]+stop_token_idx+1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    # 使用cache进行相关的推理
                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size
                    )
                    # 获取下一个token的logits和更新后的cache
                    logits, past_key_values = output.logits, output.past_key_values
                    # 获取下一个token的id
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1

                    break

                for small_block_idx in range(num_small_blocks):
                    # 计算当前小块的起始和结束位置
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    # 计算在当前 block 中的相对位置
                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        if use_block_cache:
                            # 没有可用的cache
                            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                                # 推理并创建cache
                                output = self.forward(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    # 这里不更新cache
                                    update_past_key_values=False,
                                    use_block_cache=True
                                )
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.forward(
                                    input_ids=x_t[:,start:end],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    # 这里不更新cache
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    # 提供 block_past_key_values
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx
                                ).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        # 在不使用cache的情况下，直接进行推理
                        else:
                            # 得到logits
                            logits = self.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False
                            ).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                        # 对logits进行采样
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        # 根据阈值选择需要去掩码的位置
                        unmask_idx = (x1_p > threshold)
                        # 选择概率最大的token
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        # 检查是否需要停止生成
                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1) # shape: [B]
                        finished_flag = finished_flag | finished_row_flags

                        step += 1

            # 更新序列和清理已完成的样本
            if input_ids.shape[1] ==  x_t.shape[1]:
                input_ids = x_t
            else:
                # 更新输入序列：拼接完成的token
                input_ids[:, :(block_idx + 1)*block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    # 如果输入序列的长度小于当前block的长度，直接更新
                    if input_ids.shape[1] <= (block_idx + 1)*block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1)*block_size] = x_t[seq_block_idx == block_idx, (block_idx + 1)*block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

            # 清理已完成的样本
            if finished_flag.any():
                # 将已完成的样本添加到finished_samples中
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)
                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

                finished_flag = finished_flag[~finished_flag]



        # add not finished samples since max_new_tokens is reached
        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)

        assert len(finished_samples) == batch_size
        return finished_samples


    # 定义MDM采样函数，用于单条的可视化生成过程
    # 在演示中主要使用此函数，用于进行可视化展示
    @torch.no_grad()
    def mdm_sample_with_visualization(
        self,
        input_ids,
        tokenizer,
        block_size=32,
        max_new_tokens=1024,
        mask_id=FAST_DLLM_MASK_ID,
        threshold=0.95,
        small_block_size=32,
        stop_token=FAST_DLLM_STOP_TOKEN,
        temperature=0.0,
        top_p=0.95,
    ):
        """
        MDM sampling function with visualization
        with intermediate state output for Gradio visualization
        """
        nfe = 0
        self.model.bd_size = block_size
        num_blocks = max_new_tokens // block_size

        # Initialize state - show all positions as mask
        initial_state = []

        if input_ids.shape[1] > block_size:
            output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True)
            logits, past_key_values = output.logits, output.past_key_values
            nfe += 1
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size
        original_input_length = input_ids.shape[1]

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]

            # Use the length of the first block to initialize state
            first_block_length = block_size - (input_ids.shape[1] % block_size)

            if len(initial_state) == 0:
                for i in range(first_block_length):
                    initial_state.append(("[MASK]", MASK_COLOR))
                yield initial_state
            else:
                for i in range(first_block_length):
                    current_state.append(("[MASK]", MASK_COLOR))
                yield current_state


            # Initialize x_init as mask_id
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)

            x_t = x_init.clone()
            block_past_key_values = None
            step = 0

            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate next token
                if mask_idx.sum() == 0:
                    nfe += 1
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    token_text = tokenizer.decode([next_token[0].item()], skip_special_tokens=True)
                    # Handle special characters
                    token_text = token_text
                    current_state.append((token_text, TOKEN_COLOR))
                    yield current_state
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                break

                        logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]

                        step += 1
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)

                        # Select tokens with probability greater than threshold in p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, small_block_start_idx:small_block_end_idx], x1_p, -torch.inf)
                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        # Generate visualization state
                        current_state = []
                        generated_tokens = x_t[0, original_input_length:]

                        # Display generated tokens
                        for i, token_id in enumerate(generated_tokens):
                            if token_id == mask_id:
                                current_state.append(("[MASK]", MASK_COLOR))
                            else:
                                token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
                                # Handle special characters
                                token_text = token_text
                                current_state.append((token_text, TOKEN_COLOR))

                        yield current_state

            input_ids = x_t

        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]

        # Final state - display complete text
        final_state = []
        generated_tokens = input_ids[0, original_input_length:]
        for token_id in generated_tokens:
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
            token_text = token_text
            final_state.append((token_text, TOKEN_COLOR))

        # Final state doesn't need mask padding, only show actually generated tokens

        yield final_state

        # Return final text
        final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield final_text


def setup_model_with_custom_generation(model):
    """
    Set up custom generation functions for the model
    """
    # Add mdm_sample method with visualization
    model.mdm_sample_with_visualization = types.MethodType(Fast_dLLM_QwenForCausalLM.mdm_sample_with_visualization, model)
    return model
