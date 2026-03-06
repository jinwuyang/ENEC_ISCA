import os
from typing import Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM,BertModel,CLIPModel,LlamaForCausalLM,Wav2Vec2ForCTC
from diffusers import StableVideoDiffusionPipeline

def load(model_path):
    """
    Load the model and return the model along with its model_name.
    """
    # FP32 models
    if 'bert-base-uncased' in model_path:
        model = BertModel.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float32  # Must specify the data type for model parameters
        )
        model_name = 'bert-base-uncased'
    elif 'OLMo-1B-hf' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float32  # Must specify the data type for model parameters
        )
        model_name = 'OLMo-1B-hf'
    elif 'wav2vec2-large-xlsr-53-english' in model_path:
        model = Wav2Vec2ForCTC.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float32  # Must specify the data type for model parameters
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
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float16  # Must specify the data type for model parameters
        )
        model_name = 'Llama-2-7b-ms'
    elif 'CapybaraHermes-2.5-Mistral-7B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float16  # Must specify the data type for model parameters
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
    #     trust_remote_code=True,  # Need to enable this if the model contains custom code
    #     torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
    #     )
    #     model_name = 'DeepSeek-R1-Distill-Llama-8B'
    # elif 'DeepSeek-R1-Distill-Qwen-7B' in model_path:
    #     model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,  # Need to enable this if the model contains custom code
    #     torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
    #     )
    #     model_name = 'DeepSeek-R1-Distill-Qwen-7B'
    elif 'falcon-7b' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
        )
        model_name = 'falcon-7b'
    elif 'Qwen3-32B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
        ).cpu()  # Cannot fit on a single GPU, so tensors must be processed one by one
        model_name = 'Qwen3-32B'
    elif 'Qwen3-8B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
        )
        model_name = 'Qwen3-8B'
    elif 'Meta-Llama-3.1-8B-Instruct' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
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
    Read a portion of a binary file and convert it into a PyTorch Tensor with the specified shape and type.

    Args:
        file_path (str): The path to the binary file.
        num_bytes (int): The number of bytes to read from the beginning of the file.
        shape (Tuple[int, ...]): The target shape of the output Tensor.
        dtype (torch.dtype, optional): The target data type of the output Tensor. Defaults to torch.bfloat16.

    Returns:
        torch.Tensor: The converted Tensor.
        
    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If parameters are invalid (e.g., bytes and shape mismatch).
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")

    # Get the byte size of the target data type (e.g., bfloat16 is 2 bytes)
    element_size = torch.finfo(dtype).bits // 8
    
    # Check if the number of bytes to read is an integer multiple of the element size
    if num_bytes % element_size != 0:
        raise ValueError(
            f"Error: The number of bytes to read ({num_bytes}) "
            f"must be an integer multiple of the bytes per element ({element_size}) for target type {dtype}."
        )

    # Calculate total elements required by the shape
    expected_elements = np.prod(shape)
    # Calculate actual elements obtained from the byte count
    actual_elements = num_bytes // element_size
    
    if expected_elements != actual_elements:
        raise ValueError(
            f"Error: Shape {shape} requires {expected_elements} elements, "
            f"but based on bytes read ({num_bytes}) and dtype ({dtype}), "
            f"only {actual_elements} elements can be obtained."
        )

    # --- Core Logic ---
    print(f"Reading file: {file_path}")
    with open(file_path, 'rb') as f:
        # 1. Read the specified number of bytes
        byte_data = f.read(num_bytes)
        
        # Check if enough bytes were successfully read
        if len(byte_data) < num_bytes:
            raise IOError(
                f"Error: Attempted to read {num_bytes} bytes, but the file only contains {len(byte_data)} bytes."
            )

    # 2. Convert byte data through Numpy to an unsigned 16-bit integer array (since bf16 is 2 bytes)
    #    This is a key step to properly group byte streams into 16-bit chunks.
    if dtype == torch.bfloat16 or dtype == torch.float16:
        # Use np.uint16 for bfloat16 and float16
        numpy_array = np.frombuffer(byte_data, dtype=np.uint16)
    else:
        numpy_array = np.frombuffer(byte_data, dtype=np.uint32)
    
    # 3. Convert Numpy array to PyTorch Tensor
    #    The Tensor data type here is torch.int16 (or uint16, depending on the system)
    tensor_int16 = torch.fromnumpy(numpy_array)
    
    # 4. Reinterpret the underlying bits of the 16-bit integer into the target dtype (e.g., bfloat16)
    #    This is an efficient, zero-copy operation
    tensor_bf16 = tensor_int16.view(dtype)
    
    # 5. Reshape the Tensor
    output_tensor = tensor_bf16.reshape(shape)
    
    return output_tensor


def get_result_path(processed_file_path,result_dir):
    """
    Converts the result file path to a path within the result directory by finding the common prefix.
    For example:
    input: /data/yjw/2.7B_weight_tensors/xxx.txt and /data/yjw/results/gradient
    return: /data/yjw/results/gradient/2.7B_weight_tensors/xxx.txt
    
    Returns the full path of the result file in the result folder,
    mirroring the structure relative to the common prefix.
    
    :param processed_file_path: Path to the file to process.
    :param result_dir: Root directory for results.
    :return: Full path in the result directory.
    """
    common = os.path.commonpath([processed_file_path, result_dir])
    relative = os.path.relpath(processed_file_path, common)
    return os.path.join(result_dir, relative)


def are_bits_equal(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """
    Check if the underlying bit representations of two Tensors are exactly equal.
    
    This method properly handles special floating-point values such as NaN, inf, and -0.0.

    Args:
        t1 (torch.Tensor): The first input Tensor.
        t2 (torch.Tensor): The second input Tensor.

    Returns:
        bool: True if both Tensors have the same shape, data type, and identical bit representations for all elements, False otherwise.
    """
    # 1. Check if shapes and data types match
    if t1.shape != t2.shape or t1.dtype != t2.dtype:
        return False
        
    # 2. If it is a floating-point type, reinterpret as an integer type for comparison
    if t1.is_floating_point():
        # Map floating-point types to integers of equal bit-width
        dtype_map = {
            torch.float64: torch.int64,
            torch.float32: torch.int32,
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16, # Both bfloat16 and float16 are 16-bit
        }
        
        # Check if dtype is in the map
        if t1.dtype not in dtype_map:
            raise TypeError(f"Unsupported floating-point type: {t1.dtype}")
        
        int_dtype = dtype_map[t1.dtype]
        
        # Reinterpret the tensor bits as integers using view(dtype)
        t1 = t1.view(int_dtype)
        t2 = t2.view(int_dtype)
        
        # Integer tensors can be directly compared using torch.equal
        equal = torch.equal(t1, t2)

    # 3. If it is not a floating-point number (e.g., int, bool), compare directly with torch.equal
    else:
        equal = torch.equal(t1, t2)
    # for debug
    # if not equal:
    #     # diff_indices = (t1 != t2).reshape(4,-1).nonzero(as_tuple=True)
    #     diff_indices = (t1 != t2).nonzero(as_tuple=True)
    #     num_diffs = len(diff_indices[0])
    #     print(f"Found {num_diffs} differences:")
    #     for i in range(min(num_diffs, 10)):  # Print at most the first 10 differences
    #         idx = tuple(idx[i].item() for idx in diff_indices)
    #         print(f"Position {idx}: t1={t1[idx].item()}, t2={t2[idx].item()}")

    return equal


def split_model(model_path,dtype='BF16'):
    """
    Load the model and split its parameter data into individual tensors, 
    saving each as a raw binary file.
    """
    from transformers import AutoModelForCausalLM,BertModel,CLIPModel,LlamaForCausalLM,Wav2Vec2ForCTC
    # FP32 models
    if 'bert-base-uncased' in model_path:
        model = BertModel.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float32  # Must specify the data type for model parameters
        )
        model_name = 'bert-base-uncased'
    elif 'OLMo-1B-hf' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float32  # Must specify the data type for model parameters
        )
        model_name = 'OLMo-1B-hf'
    elif 'wav2vec2-large-xlsr-53-english' in model_path:
        model = Wav2Vec2ForCTC.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float32  # Must specify the data type for model parameters
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
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.float16  # Must specify the data type for model parameters
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
    #     trust_remote_code=True,  # Need to enable this if the model contains custom code
    #     torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
    #     )
    #     model_name = 'DeepSeek-R1-Distill-Llama-8B'
    # elif 'DeepSeek-R1-Distill-Qwen-7B' in model_path:
    #     model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,  # Need to enable this if the model contains custom code
    #     torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
    #     )
    #     model_name = 'DeepSeek-R1-Distill-Qwen-7B'
    elif 'falcon-7b' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
        )
        model_name = 'falcon-7b'
    elif 'Qwen3-32B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
        ).cpu()  # Cannot fit on a single GPU, so tensors must be processed one by one
        model_name = 'Qwen3-32B'
    elif 'Qwen3-8B' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
        )
        model_name = 'Qwen3-8B'
    elif 'Meta-Llama-3.1-8B-Instruct' in model_path:
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # Need to enable this if the model contains custom code
        torch_dtype=torch.bfloat16  # Must specify the data type for model parameters
        )
        model_name = 'Meta-Llama-3.1-8B-Instruct'
    else:
        raise ValueError(f"Unsupported model type or path: {model_path}")

    # Create a 'split' directory under model_path
    save_dir = os.path.join(model_path, 'split')
    if os.path.exists(save_dir):
        return
    os.makedirs(save_dir, exist_ok=True)
    print(f"Splitting model and saving to directory: {save_dir}")
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
            print(f"Saved: {param_path}, Shape: {param_np.shape}, Dtype: {dtype}")
