# ENEC Code

## Directory Structure
- enec: Operator implementation code, including ablation experiments and the final ENEC version.
- param_search: Hyperparameter search code.

## Usage
### Hyperparameter Search:
```python
# Please modify the model_dir in param_search/param_search.py to your own model storage directory, and results_dir to the result saving directory.
# Then run:
python param_search.py
```

### ENEC Operator Usage:
- Compile
``` shell
# Enter the corresponding folder to run
mkdir build
cd build
cmake ..
make
```
- Test
```shell
# Compress
./compress sourcefile tempfile inputBytesNum
# Decompress
./decompress tempfile resfile sourcefile 
```
- Compression Ratio: The compression ratio will be printed during compression. You can also calculate the compression ratio by dividing the file size before compression by the file size after compression.
- Performance: Please use msprof for compression/decompression testing, and then check the results in the op_statistic csv file within the mindstudio_profiler_output folder in the prof result folder.



## Core API Interfaces
### Hyperparameter Search
#### find_hyperparams

- Overview: A function for finding the optimal hyperparameters based on an input tensor. It returns the optimal hyperparameters `b`, `n`, `m`, `L`, and the average bit length based on a specific algorithm.

- Signature:

  ```python
  def find_hyperparams(tensor: torch.Tensor) -> dict
  ```

- Parameters:

  | Parameter Name | Type         | Description                                                  |
  | -------------- | ------------ | ------------------------------------------------------------ |
  | tensor         | torch.Tensor | The input tensor. Supported data types are `torch.bfloat16`, `torch.float16`, or `torch.float32`. |

- Return Value:

  | Return Type | Description                            |
  | ----------- | -------------------------------------- |
  | dict        | A dictionary containing the following keys: |
  |             | - `b`: Hyperparameter b, branchless integer transformation parameter. |
  |             | - `n`: Hyperparameter n, quantization bit-width parameter. |
  |             | - `m`: Hyperparameter m, quantization bit-width parameter. |
  |             | - `L`: Hyperparameter L, grouping parameter. |
  |             | - `average_bit_length`: The average bit length. |

### ENEC

#### Compression:

- Overview

  This is a compression algorithm implementation for the BF16 data type, primarily used for efficient data compression on AI chips. Through complex bit operations and hierarchical compression techniques, the algorithm decomposes BF16 floating-point data into multiple components (mantissa, exponent, sign bit, etc.) and compresses them for storage, in order to reduce memory footprint and improve data transfer efficiency.

- Signature

Main entry function:

```c++
extern "C" void enec_compress(
    Header *cphd, 
    void *stream, 
    uint8_t* srcDevice, 
    uint8_t* compressedDevice, 
    uint8_t* compressedFinal, 
    uint8_t* histogramDevice, 
    uint8_t* blockCompSize
)
```

Device kernel function:

```c++
__global__ __aicore__ void compBF16(
    uint32_t datablockNum,
    uint32_t datablockSize, 
    uint32_t elementNum,
    uint32_t tileLength,
    __gm__ uint8_t* srcDevice,
    __gm__ uint8_t* msGlobal, 
    __gm__ uint8_t* e0Global,
    __gm__ uint8_t* mblGlobal,
    __gm__ uint8_t* e1Global,
    __gm__ uint8_t* histogramDevice,
    __gm__ uint8_t* blockCompSize
)
```

Compression kernel class:

```c++
template <typename T>
class CompressKernelBF16
{
public:
    __aicore__ inline CompressKernelBF16();
    __aicore__ inline void Init(TPipe *pipe, ...);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn(uint32_t i);
    __aicore__ inline void Compute(...);
    __aicore__ inline void CopyOut(uint32_t i);
}
```

- Input Parameters

cphd: Compression header information structure, containing metadata such as data type, data block quantity, size, etc.
stream: Computation stream pointer, used for asynchronous execution management.
srcDevice: Source data device memory pointer, the original BF16 data to be compressed.
histogramDevice: Histogram data device pointer, providing compression statistics.

- Output Parameters

compressedDevice: Compressed data device pointer (primary compression result).
compressedFinal: Final compressed data device pointer (complete compressed package).
blockCompSize: Block compression size information device pointer.

- Internal Parameters

datablockNum: Total number of data blocks.
datablockSize: Size of a single data block (in bytes).
elementNum: Number of elements in each data block.
tileLength: Tile length, used for data chunking processing.

- Return Value

All functions have no return value (`void` type). Execution results are returned via output parameters. Compression status and error information need to be handled through other mechanisms (such as return values or exceptions), which are not reflected in the current code.

#### Decompression

- Overview

  This is a decompression algorithm implementation for the BF16 data type, primarily used for efficient data decompression on AI chips. This algorithm is the inverse process of the compression algorithm. Through complex bit operations, cumulative sum calculations, and data recombination techniques, it restores the compressed data back to the original BF16 format data.

- Signature

Main entry function:

```c++
extern "C" void enec_decompress(
    Header* cphd, 
    void* stream, 
    uint8_t* compressed, 
    uint8_t* decompressed
)
```

Device kernel function:

```c++
__global__ __aicore__ void decompBF16(
    uint32_t BUFFER_NUM,
    uint32_t elementNum,
    uint32_t tileLength,
    uint32_t tileNum,
    uint32_t threadblockNum,
    uint32_t datablockNum,
    uint32_t datablockSize,
    uint32_t totalCompressedBytes,
    __gm__ uint8_t* msGlobal,
    __gm__ uint8_t* eGlobal0,
    __gm__ uint8_t* mblGlobal,
    __gm__ uint8_t* compSizePrefix,
    __gm__ uint8_t* eGlobal1,
    __gm__ uint8_t* decompressedGlobal
)
```

Decompression kernel class:

```c++
template <typename T>
class DecompressKernelBF16
{
public:
    __aicore__ inline DecompressKernelBF16();
    __aicore__ inline void Init(TPipe *pipe, ...);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn_mbl(uint32_t i);
    __aicore__ inline void Compute(...);
}
```

- Input Parameters

cphd: Pointer to compression header information structure, containing metadata required for decompression.
stream: Computation stream pointer, used for asynchronous execution management.
compressed: Compressed data device memory pointer, containing all compression components.

- Internal Processing Parameters

BUFFER_NUM: Number of buffers, used for queue management.
elementNum: Number of elements in each data block.
tileLength: Tile length, used for data chunking processing.
tileNum: Number of tiles, calculated as elementNum / tileLength.
threadblockNum: Number of thread blocks, controlling the degree of parallelism.
datablockNum: Total number of data blocks.
datablockSize: Size of a single data block (in bytes).
totalCompressedBytes: Total compressed bytes.

- Compressed Data Components

msGlobal: Mantissa part data input.
eGlobal0: Exponent part data 0 input.
mblGlobal: Mask bit length data input.
compSizePrefix: Compression size prefix information.
eGlobal1: Exponent part data 1 input.

- Output Parameters

decompressed: Decompressed original data device memory pointer.
decompressedGlobal: Global output for decompressed data.

- Return Value

All functions have no return value (`void` type). Execution results are returned via output parameters. Decompression status and error information need to be handled through other mechanisms, which are not reflected in the current code.
