import os
from typing import Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM,BertModel,CLIPModel,LlamaForCausalLM,Wav2Vec2ForCTC
from diffusers import StableVideoDiffusionPipeline

def load(model_path):
    """
        将模型加载后返回model和model_name
    """
    # FP32 models
    if 'bert-base-uncased' in model_path:
        model = BertModel.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float32  # 必须指定模型参数的数据类型
        )
        model_name = 'bert-base-uncased'
    elif 'OLMo-1B-hf' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float32  # 必须指定模型参数的数据类型
        )
        model_name = 'OLMo-1B-hf'
    elif 'wav2vec2-large-xlsr-53-english' in model_path:
        model = Wav2Vec2ForCTC.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float32  # 必须指定模型参数的数据类型
        )
        model_name = 'wav2vec2-large-xlsr-53-english'
    # FP16 models
    elif 'stable-video-diffusion-img2vid-fp16' in model_path:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16"
        )
        return pipe, 'stable-video-diffusion-img2vid-fp16'
    elif 'Llama-2-7b-ms' in model_path:
        model = LlamaForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float16  # 必须指定模型参数的数据类型
        )
        model_name = 'Llama-2-7b-ms'
    elif 'CapybaraHermes-2.5-Mistral-7B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float16  # 必须指定模型参数的数据类型
        )
        model_name = 'CapybaraHermes-2.5-Mistral-7B'
    # BF16 models
    elif 'deepseek-llm-7b-base' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model_name = 'deepseek-llm-7b-base'
    # elif 'DeepSeek-R1-Distill-Llama-8B' in model_path:
    #     model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
    #     torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
    #     )
    #     model_name = 'DeepSeek-R1-Distill-Llama-8B'
    # elif 'DeepSeek-R1-Distill-Qwen-7B' in model_path:
    #     model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
    #     torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
    #     )
    #     model_name = 'DeepSeek-R1-Distill-Qwen-7B'
    elif 'falcon-7b' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
        )
        model_name = 'falcon-7b'
    elif 'Qwen3-32B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
        ).cpu()  # 这里不能单卡放下，所以得一个个tensor处理
        model_name = 'Qwen3-32B'
    elif 'Qwen3-8B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
        )
        model_name = 'Qwen3-8B'
    elif 'Meta-Llama-3.1-8B-Instruct' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
        )
        model_name = 'Meta-Llama-3.1-8B-Instruct'
    else:
        raise ValueError(f"Unsupported model type or path: {model_path}")

    return model,model_name




def read_bin_to_tensor(
    file_path: str,
    num_bytes: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    读取二进制文件的一部分，并将其转换为指定形状和类型的PyTorch Tensor。

    Args:
        file_path (str): 二进制文件的路径。
        num_bytes (int): 要从文件开头读取的字节数。
        shape (Tuple[int, ...]): 输出Tensor的目标形状。
        dtype (torch.dtype, optional): 输出Tensor的目标数据类型。
            默认为 torch.bfloat16。

    Returns:
        torch.Tensor: 转换后的Tensor。
        
    Raises:
        FileNotFoundError: 如果文件路径不存在。
        ValueError: 如果参数不合法（例如，字节数和形状不匹配）。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：文件 '{file_path}' 未找到。")

    # 获取目标数据类型的字节大小（例如 bfloat16 是 2 字节）
    element_size = torch.finfo(dtype).bits // 8
    
    # 检查要读取的字节数是否是元素大小的整数倍
    if num_bytes % element_size != 0:
        raise ValueError(
            f"错误：要读取的字节数 ({num_bytes}) "
            f"必须是目标类型 {dtype} 每个元素字节数 ({element_size}) 的整数倍。"
        )

    # 计算形状所需的总元素数量
    expected_elements = np.prod(shape)
    # 计算从字节数可以得到的元素数量
    actual_elements = num_bytes // element_size
    
    if expected_elements != actual_elements:
        raise ValueError(
            f"错误：形状 {shape} 需要 {expected_elements} 个元素, "
            f"但根据读取的字节数 ({num_bytes}) 和数据类型 ({dtype})，"
            f"只能得到 {actual_elements} 个元素。"
        )

    # --- 核心逻辑 ---
    print(f"正在读取文件: {file_path}")
    with open(file_path, 'rb') as f:
        # 1. 读取指定数量的字节
        byte_data = f.read(num_bytes)
        
        # 检查是否成功读取了足够的字节
        if len(byte_data) < num_bytes:
            raise IOError(
                f"错误：尝试读取 {num_bytes} 字节，但文件只包含 {len(byte_data)} 字节。"
            )

    # 2. 将字节数据通过Numpy转换为无符号16位整数数组 (因为bf16是2字节)
    #    这是将字节流正确分组为16位块的关键步骤。
    if dtype == torch.bfloat16 or dtype == torch.float16:
        # 对于 bfloat16 和 float16，使用 np.uint16
        numpy_array = np.frombuffer(byte_data, dtype=np.uint16)
    else:
        numpy_array = np.frombuffer(byte_data, dtype=np.uint32)
    
    # 3. 将Numpy数组转换为PyTorch Tensor
    #    此时Tensor的数据类型是 torch.int16 (或者 uint16，取决于系统)
    tensor_int16 = torch.from_numpy(numpy_array)
    
    # 4. 将16位整数的底层二进制位重新解释为 bfloat16 类型
    #    这是一个高效的、零拷贝的操作
    tensor_bf16 = tensor_int16.view(dtype)
    
    # 5. 调整Tensor的形状
    output_tensor = tensor_bf16.reshape(shape)
    
    return output_tensor


def get_result_path(processed_file_path,result_dir):
    """
        主要是把结果文件的路径转换为结果目录下的路径(根据公共前缀找到input的后缀部分)，例如：
        input:/data/yjw/2.7B_weight_tensors/xxx.txt /data/yjw/results/gradient。
        return:/data/yjw/results/gradient/2.7B_weight_tensors/xxx.txt
    Returns the full path of the result file in the result folder,
    mirroring the structure relative to the common prefix.
    
    :param file_path: Path to the file to process.
    :param result_root: Root directory for results.
    :return: Full path in the result directory.
    """
    common = os.path.commonpath([processed_file_path, result_dir])
    relative = os.path.relpath(processed_file_path, common)
    return os.path.join(result_dir, relative)


def are_bits_equal(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """
    判断两个 Tensor 的底层 bit 表示是否完全相等。
    
    这个方法可以正确处理 NaN, inf, -0.0 等特殊浮点数值。

    Args:
        t1 (torch.Tensor): 第一个输入 Tensor。
        t2 (torch.Tensor): 第二个输入 Tensor。

    Returns:
        bool: 如果两个 Tensor 的形状、数据类型和所有元素的 bit 表示都相同，则返回 True，否则返回 False。
    """
    # 1. 检查形状和数据类型是否一致
    if t1.shape != t2.shape or t1.dtype != t2.dtype:
        return False
        
    # 2. 如果是浮点数类型，则重新解释为整数类型进行比较
    if t1.is_floating_point():
        # 定义浮点数到等位宽整数的映射
        dtype_map = {
            torch.float64: torch.int64,
            torch.float32: torch.int32,
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16, # bfloat16 和 float16 都是16位
        }
        
        # 检查dtype是否在映射中
        if t1.dtype not in dtype_map:
            raise TypeError(f"不支持的浮点数类型: {t1.dtype}")
        
        int_dtype = dtype_map[t1.dtype]
        
        # 使用 view(dtype) 将 tensor 的 bit 重新解释为整数
        t1 = t1.view(int_dtype)
        t2 = t2.view(int_dtype)
        
        # 整数 tensor 可以直接用 torch.equal 比较
        equal = torch.equal(t1, t2)

    # 3. 如果不是浮点数（如 int, bool），直接用 torch.equal 比较即可
    else:
        equal = torch.equal(t1, t2)
    # for debug
    # if not equal:
    #     # diff_indices = (t1 != t2).reshape(4,-1).nonzero(as_tuple=True)
    #     diff_indices = (t1 != t2).nonzero(as_tuple=True)
    #     num_diffs = len(diff_indices[0])
    #     print(f"发现 {num_diffs} 处不同:")
    #     for i in range(min(num_diffs, 10)):  # 最多打印前10处不同
    #         idx = tuple(idx[i].item() for idx in diff_indices)
    #         print(f"位置 {idx}: t1={t1[idx].item()}, t2={t2[idx].item()}")

    return equal


def split_model(model_path,dtype='BF16'):
    """
        将模型加载后进行逐个tensor的数据进行直接二进制保存
    """
    from transformers import AutoModelForCausalLM,BertModel,CLIPModel,LlamaForCausalLM,Wav2Vec2ForCTC
    # FP32 models
    if 'bert-base-uncased' in model_path:
        model = BertModel.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float32  # 必须指定模型参数的数据类型
        )
        model_name = 'bert-base-uncased'
    elif 'OLMo-1B-hf' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float32  # 必须指定模型参数的数据类型
        )
        model_name = 'OLMo-1B-hf'
    elif 'wav2vec2-large-xlsr-53-english' in model_path:
        model = Wav2Vec2ForCTC.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float32  # 必须指定模型参数的数据类型
        )
        model_name = 'wav2vec2-large-xlsr-53-english'
    # FP16 models
    elif 'stable-video-diffusion-img2vid-fp16' in model_path:
        model = StableVideoDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16"
        )
        model_name = 'stable-video-diffusion-img2vid-fp16'
    elif 'CapybaraHermes-2.5-Mistral-7B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.float16  # 必须指定模型参数的数据类型
        )
        model_name = 'CapybaraHermes-2.5-Mistral-7B'
    # BF16 models
    elif 'deepseek-llm-7b-base' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model_name = 'deepseek-llm-7b-base'
    # elif 'DeepSeek-R1-Distill-Llama-8B' in model_path:
    #     model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
    #     torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
    #     )
    #     model_name = 'DeepSeek-R1-Distill-Llama-8B'
    # elif 'DeepSeek-R1-Distill-Qwen-7B' in model_path:
    #     model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
    #     torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
    #     )
    #     model_name = 'DeepSeek-R1-Distill-Qwen-7B'
    elif 'falcon-7b' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
        )
        model_name = 'falcon-7b'
    elif 'Qwen3-32B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
        ).cpu()  # 这里不能单卡放下，所以得一个个tensor处理
        model_name = 'Qwen3-32B'
    elif 'Qwen3-8B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
        )
        model_name = 'Qwen3-8B'
    elif 'Meta-Llama-3.1-8B-Instruct' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 如果模型包含自定义代码，需要启用此选项
        torch_dtype=torch.bfloat16  # 必须指定模型参数的数据类型
        )
        model_name = 'Meta-Llama-3.1-8B-Instruct'
    else:
        raise ValueError(f"Unsupported model type or path: {model_path}")

    # 在model_path下创建一个split目录
    save_dir = os.path.join(model_path, 'split')
    if os.path.exists(save_dir):
        return
    os.makedirs(save_dir, exist_ok=True)
    print(f"正在将模型拆分并保存到目录: {save_dir}")
    model.eval()
    dtype = dtype.lower()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if  param.dim() == 1:
                continue
            if dtype == 'fp32':
                param_np = param.view(torch.uint32).cpu().numpy()
            else:
                param_np = param.view(torch.uint16).cpu().numpy()
            param_path = os.path.join(save_dir, f"{name}.bin")
            param_np.tofile(param_path)
            print(f"已保存: {param_path}, 形状: {param_np.shape}, 数据类型: {dtype}")

    

