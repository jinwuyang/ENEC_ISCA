import os
import argparse
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

def find_weights_files(model_path):
    """
    Find all weight files (.safetensors or .bin) in the model directory.
    Prefer .safetensors files.
    """
    safetensors_files = []
    bin_files = []

    for filename in os.listdir(model_path):
        if filename.endswith(".safetensors"):
            safetensors_files.append(os.path.join(model_path, filename))
        elif filename.endswith(".bin"):
            bin_files.append(os.path.join(model_path, filename))

    # Prefer returning the safetensors file list
    if safetensors_files:
        return sorted(safetensors_files)
    elif bin_files:
        return sorted(bin_files)
    else:
        raise FileNotFoundError(f"No .safetensors or .bin files found in {model_path}")

def load_weights(weights_files):
    """Load weights from multiple .safetensors or .bin files."""
    all_weights = {}
    print(f"Found {len(weights_files)} weights files, starting to load...")
    
    for weights_file in weights_files:
        print(f"  > Loading: {os.path.basename(weights_file)}")
        if weights_file.endswith(".safetensors"):
            # Load using safetensors library
            with safe_open(weights_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)
        elif weights_file.endswith(".bin"):
            # Load using PyTorch
            weights = torch.load(weights_file, map_location="cpu")
            all_weights.update(weights)

    if not all_weights:
        raise ValueError("Failed to load weights from any file.")

    print(f"Successfully loaded a total of {len(all_weights)} tensors from all files.")
    return all_weights

def save_tensors(weights, output_dir, output_format):
    """Save each tensor to a separate file."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving tensors to {output_dir}...")
    total_tensors = len(weights)
    for i, (key, tensor) in enumerate(weights.items()):
        # Convert key name to a safe file name
        safe_key = key.replace('/', '_').replace('.', '_')
        output_path = os.path.join(output_dir, f"{safe_key}.{output_format}")

        if output_format == "pt":
            torch.save(tensor, output_path)
        elif output_format == "bin":
            if tensor.dtype == torch.bfloat16:
                # numpy does not support bfloat16, so convert its view to uint16 to save raw bytes
                tensor.view(torch.uint16).cpu().numpy().tofile(output_path)
            else:
                tensor.cpu().numpy().tofile(output_path)
        
        print(f"[{i+1}/{total_tensors}] Saved {key} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert model weight files to individual tensor files.")
    parser.add_argument("--model_name", 
                        type=str, 
                        help="Name of the model directory located under the 'models' directory.",
                        default="stable-video-diffusion")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=['pt', 'bin'],
        default='bin',
        help="Output tensor file format ('pt' or 'bin'). 'pt' saves PyTorch tensor objects, 'bin' saves raw binary data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/root/workspaces/datasets/weights_data',
        help="Root directory to save tensor files."
    )
    args = parser.parse_args()

    # The script is located in the models/ directory, so the model directory is its sibling directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model_name)

    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found at {model_path}")
        return

    try:
        weights_files = find_weights_files(model_path)
        weights = load_weights(weights_files)
        
        # Determine the data type from the first tensor to be used for naming
        first_tensor_dtype = next(iter(weights.values())).dtype
        dtype_str_map = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }
        dtype_str = dtype_str_map.get(first_tensor_dtype, str(first_tensor_dtype).replace("torch.", ""))

        # Include model name and format with data type in the output path
        model_name_with_dtype = f"{args.model_name}_{dtype_str}"
        output_dir = Path(args.output_dir) / model_name_with_dtype
        save_tensors(weights, output_dir, args.output_format)
        print("\nConversion complete.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
