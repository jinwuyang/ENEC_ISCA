# ENEC代码

## 目录结构
- enec:算子实现代码，包括消融实验和最终ENEC版本。
- param_search：超参搜索代码

## 使用方法
### 超参数搜索：
```python
# 请修改param_search/param_search.py代码中的model_dir为自己的模型存储目录，results_dir为结果保存目录
# 然后运行：
python param_search.py
```

### ENEC算子使用：
- 编译
``` shell
# 进入对应文件夹运行
mkdir build
cd build
cmake ..
make
```
- 测试
```shell
# 压缩
./compress sourcefile tempfile inputBytesNum
# 解压缩
./decompress tempfile resfile sourcefile 
```
- 压缩率：压缩时会打印出压缩率。也可通过将压缩前文件大小除以压缩后文件大小作为压缩率。
- 性能：请基于msprof进行测试压缩/解压缩，然后在prof结果文件夹中的mindstudio_profiler_output文件夹中查看op_statistic的csv文件中的结果



## 核心API接口
### 超参数搜索
#### find_hyperparams

- 概述：一个用于根据输入张量寻找最优超参数的函数，基于特定的算法返回最优的超参数 `b`, `n`, `m`, `L` 以及平均比特长度。

- 签名：

  ```python
  def find_hyperparams(tensor: torch.Tensor) -> dict
  ```

- 参数：

  | 参数名 | 类型         | 描述                                                         |
  | ------ | ------------ | ------------------------------------------------------------ |
  | tensor | torch.Tensor | 输入张量，支持的数据类型为 `torch.bfloat16`、`torch.float16` 或 `torch.float32`。 |

- 返回值：

  | 返回类型 | 描述                                   |
  | -------- | -------------------------------------- |
  | dict     | 包含以下键值的字典：                   |
  |          | - `b`: 超参数 b，无分支整数变换参数。  |
  |          | - `n`: 超参数 n，量化位宽参数。        |
  |          | - `m`: 超参数 m，量化位宽参数。        |
  |          | - `L`: 超参数 L，分组参数。            |
  |          | - `average_bit_length`: 平均比特长度。 |

### ENEC

#### 压缩：

- 概述

​	这是一个针对BF16数据类型的压缩算法实现，主要用于AI芯片上的高效数据压缩。该算法通过复杂的位操作和分层压缩技术，将BF16浮点数据分解为多个组成部分（尾数、指数、符号位等）并进行压缩存储，以减小内存占用和提高数据传输效率。

- 签名

主入口函数：

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

设备核函数

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

压缩内核类

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

- 输入参数

cphd: 压缩头信息结构体，包含数据类型、数据块数量、大小等元数据
stream: 计算流指针，用于异步执行管理
srcDevice: 源数据设备内存指针，待压缩的原始BF16数据
histogramDevice: 直方图数据设备指针，提供压缩统计信息

- 输出参数

compressedDevice: 压缩后数据设备指针（主要压缩结果）
compressedFinal: 最终压缩数据设备指针（完整压缩包）
blockCompSize: 块压缩大小信息设备指针

- 内部参数

datablockNum: 数据块总数
datablockSize: 单个数据块大小（字节）
elementNum: 每个数据块中的元素数量
tileLength: 瓦片长度，用于数据分块处理

- 返回值

所有函数均无返回值（void类型），执行结果通过输出参数返回。压缩状态和错误信息需要通过其他机制（如返回值或异常）处理，在当前代码中未体现。

#### 解压缩

- 概述

​	这是一个针对BF16数据类型的解压缩算法实现，主要用于AI芯片上的高效数据解压缩。该算法是压缩算法的逆过程，通过复杂的位操作、累积和计算、数据重组等技术，将压缩后的数据恢复为原始的BF16格式数据。

- 签名

主入口函数

```c++
extern "C" void enec_decompress(
    Header* cphd, 
    void* stream, 
    uint8_t* compressed, 
    uint8_t* decompressed
)
```

设备核函数

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

解压缩内核类

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

- 输入参数

cphd: 压缩头信息结构体指针，包含解压缩所需的元数据
stream: 计算流指针，用于异步执行管理
compressed: 压缩数据设备内存指针，包含所有压缩组件

- 内部处理参数

BUFFER_NUM: 缓冲区数量，用于队列管理
elementNum: 每个数据块中的元素数量
tileLength: 瓦片长度，用于数据分块处理
tileNum: 瓦片数量，计算为elementNum / tileLength
threadblockNum: 线程块数量，控制并行度
datablockNum: 数据块总数
datablockSize: 单个数据块大小（字节）
totalCompressedBytes: 总压缩字节数

- 压缩数据组件

msGlobal: 尾数部分数据输入
eGlobal0: 指数部分数据0输入
mblGlobal: 掩码位长度数据输入
compSizePrefix: 压缩大小前缀信息
eGlobal1: 指数部分数据1输入

- 输出参数

decompressed: 解压缩后的原始数据设备内存指针
decompressedGlobal: 解压缩数据全局输出

- 返回值

所有函数均无返回值（void类型），执行结果通过输出参数返回。解压缩状态和错误信息需要通过其他机制处理，在当前代码中未体现。
