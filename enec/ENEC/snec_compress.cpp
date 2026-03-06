#include "snec_utils.h"
#include "snec_device.h"

template <typename T>
class CompressKernelBF16
{
public:
    __aicore__ inline CompressKernelBF16() {}

    __aicore__ inline void Init(TPipe *pipe,
                                uint32_t datablockNum,   
                                uint32_t datablockSize,  
                                uint32_t elementNum,   
                                uint32_t tileLength,
                                __gm__ uint8_t *srcDevice,  
                                __gm__ uint8_t *msGlobal, 
                                __gm__ uint8_t *e0Global,    
                                __gm__ uint8_t *mblGlobal,   
                                __gm__ uint8_t *e1Global,   
                                __gm__ uint8_t *histogramDevice,
                                __gm__ uint8_t *blockCompSize) 
    {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();   
        this->blockNum = GetBlockNum(); 
        this->computeNum = elementNum;
        this->tileLength = tileLength;  
        this->tileNum = elementNum / tileLength; 
        this->datablockNum = datablockNum; 
        this->datablockSize = datablockSize;
        
        int datablockNumPerBLOCK = (datablockNum + blockNum - 1) / blockNum;
        this->bufferSize = (datablockSize * datablockNumPerBLOCK);

        srcShape_cmp[0] = 1;
        srcShape_cmp[1] = tileNum / 8 / sizeof(half);
        dstShape_cmp[0] = tileLength;
        dstShape_cmp[1] = tileNum / 8 / sizeof(half);

        input.SetGlobalBuffer((__gm__ T *)(srcDevice));
        table_input.SetGlobalBuffer((__gm__ T *)(histogramDevice));
        ms_output.SetGlobalBuffer((__gm__ T *)(msGlobal));
        e_output0.SetGlobalBuffer((__gm__ T *)(e0Global));
        mbl_output.SetGlobalBuffer((__gm__ T *)(mblGlobal));
        e_output1.SetGlobalBuffer((__gm__ T *)(e1Global + bufferSize * blockId));
        blockCompSizeOutput.SetGlobalBuffer((__gm__ T *)(blockCompSize + 32 * blockId));

        pipe->InitBuffer(inQueue, BUFFER_NUM, computeNum * sizeof(T));    
        pipe->InitBuffer(e_outQueue0, BUFFER_NUM, computeNum * sizeof(T));  
        pipe->InitBuffer(ms_outQueue, BUFFER_NUM, computeNum);           
        pipe->InitBuffer(mbl_outQueue, BUFFER_NUM, tileLength * tileNum / 8);
    }

    __aicore__ inline void Process()
    {
        pipe->InitBuffer(e_out1, computeNum * sizeof(T)); 
        pipe->InitBuffer(merge, computeNum * sizeof(T));   
        pipe->InitBuffer(mask7, 32);                      

        LocalTensor<T> e_outLocal1 = e_out1.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        LocalTensor<T> mask7Local = mask7.Get<T>();

        AIV_WITH_BARRIER(Duplicate, mergeLocal, (T)0, computeNum);
        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)7, 32 / sizeof(T));

        uint64_t outerNum = 0;
        uint32_t totalouterNum = 0;
        uint32_t totalcompressedSize = 0;
        uint32_t cumulated_amount = 0;
        uint32_t new_cumulated_amount = 0;
        uint32_t low_write_num = 0;
        uint32_t high_unwrite_num = 0;
        uint32_t write_offset = 0;
        
        for (uint32_t i = blockId; i < datablockNum; i += blockNum)
        {
            CopyIn(i);
            Compute(i, cumulated_amount, low_write_num, high_unwrite_num, write_offset, outerNum,
                    e_outLocal1, mergeLocal, mask7Local);
            totalouterNum = totalouterNum + outerNum;
            CopyOut(i);
        }
        
        totalouterNum = (totalouterNum + 256 - 1) / 256 * 256;
        totalcompressedSize = totalcompressedSize + totalouterNum * 3 / 8;

        cumulated_amount = (cumulated_amount + 256 - 1) / 256 * 256;
        
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)13, cumulated_amount);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)13, cumulated_amount);
        
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 2], mergeLocal[cumulated_amount / 2], (uint16_t)3, cumulated_amount / 2);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 2], cumulated_amount / 2);
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4], (uint16_t)6, cumulated_amount / 4);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 4], cumulated_amount / 4);

        AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount / 4], mergeLocal, (uint16_t)8, cumulated_amount / 4);
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, cumulated_amount / 4);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, cumulated_amount / 4);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 4  + cumulated_amount / 8], mergeLocal[cumulated_amount / 4 + cumulated_amount / 8], (uint16_t)4, cumulated_amount / 8);
        AIV_WITH_BARRIER(Or, mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4  + cumulated_amount / 8], cumulated_amount / 8);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[(cumulated_amount * 3 / 16)], mergeLocal[(cumulated_amount * 3 / 16)], (uint16_t)8, cumulated_amount * 3 / 16);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[(cumulated_amount * 3 / 16)], cumulated_amount * 3 / 16);

        AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, cumulated_amount * 3 / 16);

        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)0, 32 / sizeof(T));
        mask7Local.template ReinterpretCast<int32_t>()(0) = totalcompressedSize;
        AIV_WITH_BARRIER(DataCopy, blockCompSizeOutput, mask7Local, 32 / sizeof(T));
    }

private:
    __aicore__ inline void CopyIn(uint32_t i)
    {
        uint32_t offset = i * (computeNum * sizeof(uint16_t) / sizeof(T));
        LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        AIV_WITHOUT_BARRIER(DataCopy, inLocal, input[offset], computeNum);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute( uint32_t i,
                                    uint32_t &cumulated_amount,
                                    uint32_t &low_write_num,
                                    uint32_t &high_unwrite_num,
                                    uint32_t &write_offset,
                                    uint64_t &outerNum,
                                    LocalTensor<T> &e_outLocal1,
                                    LocalTensor<T> &mergeLocal,
                                    LocalTensor<T> &mask7Local)
    {
        LocalTensor<T> e_inLocal = inQueue.DeQue<T>();
        LocalTensor<T> e_outLocal0 = e_outQueue0.AllocTensor<T>();
        LocalTensor<T> ms_outLocal = ms_outQueue.AllocTensor<T>();
        LocalTensor<T> compareMask = mbl_outQueue.AllocTensor<T>();

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0, e_inLocal, (uint16_t)9, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, e_inLocal, (uint16_t)7, computeNum);
        AIV_WITH_BARRIER(Or, e_inLocal, e_inLocal, e_outLocal0, computeNum);

        AIV_WITH_BARRIER(ShiftRight, e_outLocal0, e_inLocal, (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[computeNum / 2], e_outLocal0[computeNum / 2], (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(Or, ms_outLocal, e_outLocal0, e_outLocal0[computeNum / 2], computeNum / 2);

        AIV_WITH_BARRIER(ShiftLeft, e_inLocal, e_inLocal, (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, e_inLocal, (uint16_t)8, computeNum);

        AIV_WITH_BARRIER(Adds, e_inLocal.template ReinterpretCast<int16_t>(), e_inLocal.template ReinterpretCast<int16_t>(), (int16_t)(-123), computeNum);
        AIV_WITH_BARRIER(Muls, e_inLocal.template ReinterpretCast<int16_t>(), e_inLocal.template ReinterpretCast<int16_t>(), (int16_t)(-1), computeNum);
        
        AIV_WITH_BARRIER(ShiftLeft, e_inLocal, e_inLocal, (uint16_t)10, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_inLocal, e_inLocal, (uint16_t)10, computeNum);

        AIV_WITH_BARRIER(Or, e_outLocal0, e_inLocal, e_inLocal[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 4], computeNum / 4);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 8], computeNum / 8);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 16], computeNum / 16);

        AIV_WITH_BARRIER(CompareScalar, compareMask.template ReinterpretCast<uint8_t>(), e_outLocal0.template ReinterpretCast<half>(),
                      (mask7Local.template ReinterpretCast<half>())(0), CMPMODE::GT, tileNum);

        AIV_WITH_BARRIER(DataCopy, compareMask[64], compareMask, 64);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 1], compareMask, 64 << 1);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 2], compareMask, 64 << 2);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 3], compareMask, 64 << 3);

        AIV_WITH_BARRIER(GatherMask, e_outLocal1.template ReinterpretCast<half>(), e_inLocal.template ReinterpretCast<half>(),
                   compareMask.template ReinterpretCast<uint16_t>(), true, computeNum, {1, 1, 1, 0}, outerNum);

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0, e_inLocal, (uint16_t)13, computeNum);
        AIV_WITH_BARRIER(ShiftRight, e_outLocal0, e_outLocal0, (uint16_t)13, computeNum);

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[computeNum / 2], e_outLocal0[computeNum / 2], (uint16_t)3, computeNum / 2);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[computeNum / 4], e_outLocal0[computeNum / 4], (uint16_t)6, computeNum / 4);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[computeNum / 4], computeNum / 4);

        AIV_WITH_BARRIER(ShiftRight, e_outLocal0[computeNum / 4], e_outLocal0, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0, e_outLocal0, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftRight, e_outLocal0, e_outLocal0, (uint16_t)8, computeNum / 4);

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[computeNum / 4  + computeNum / 8], e_outLocal0[computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
        AIV_WITH_BARRIER(Or, e_outLocal0[computeNum / 4], e_outLocal0[computeNum / 4], e_outLocal0[computeNum / 4  + computeNum / 8], computeNum / 8);

        AIV_WITH_BARRIER(ShiftLeft, e_outLocal0[(computeNum * 3 / 16)], e_outLocal0[(computeNum * 3 / 16)], (uint16_t)8, computeNum * 3 / 16);
        AIV_WITH_BARRIER(Or, e_outLocal0, e_outLocal0, e_outLocal0[(computeNum * 3 / 16)], computeNum * 3 / 16);

        if(cumulated_amount + outerNum >= computeNum){
            low_write_num = computeNum - cumulated_amount;
            high_unwrite_num = outerNum - low_write_num;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], e_outLocal1, (uint16_t)3, low_write_num);
            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 2], mergeLocal[computeNum / 2], (uint16_t)3, computeNum / 2);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 2], computeNum / 2);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 4], mergeLocal[computeNum / 4], (uint16_t)6, computeNum / 4);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 4], computeNum / 4);

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum / 4], mergeLocal, (uint16_t)8, computeNum / 4);
            AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, computeNum / 4);
            AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, computeNum / 4);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 4  + computeNum / 8], mergeLocal[computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
            AIV_WITH_BARRIER(Or, mergeLocal[computeNum / 4], mergeLocal[computeNum / 4], mergeLocal[computeNum / 4  + computeNum / 8], computeNum / 8);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[(computeNum * 3 / 16)], mergeLocal[(computeNum * 3 / 16)], (uint16_t)8, computeNum * 3 / 16);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[(computeNum * 3 / 16)], computeNum * 3 / 16);

            AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, (computeNum * 3 / 16));
            write_offset = write_offset + computeNum * 3 / 16;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal, e_outLocal1[low_write_num], (uint16_t)3, high_unwrite_num);
            cumulated_amount = high_unwrite_num;
        }
        else {
            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], e_outLocal1, (uint16_t)3, outerNum);
            cumulated_amount = cumulated_amount + outerNum;
        }

        e_outQueue0.EnQue(e_outLocal0);
        ms_outQueue.EnQue(ms_outLocal);
        mbl_outQueue.EnQue(compareMask);
        inQueue.FreeTensor(e_inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t i)
    {
        LocalTensor<T> e_outLocal0 = e_outQueue0.DeQue<T>();
        LocalTensor<T> ms_outLocal = ms_outQueue.DeQue<T>();
        LocalTensor<T> compareMask = mbl_outQueue.DeQue<T>();

        AIV_WITH_BARRIER(DataCopy, e_output0[i * (computeNum * 3 / 16)], e_outLocal0, computeNum * 3 / 16);
        AIV_WITH_BARRIER(DataCopy, ms_output[i * (computeNum / sizeof(T))], ms_outLocal, (computeNum / sizeof(T)));
        AIV_WITH_BARRIER(DataCopy, mbl_output[i * (tileNum / 8 / sizeof(T))], compareMask, tileNum / 8 / sizeof(T));

        e_outQueue0.FreeTensor(e_outLocal0);
        ms_outQueue.FreeTensor(ms_outLocal);
        mbl_outQueue.FreeTensor(compareMask);
    }

private:
    TPipe *pipe;

    TQue<QuePosition::VECIN, 1> inQueue;    
    TQue<QuePosition::VECOUT, 1> e_outQueue0;  
    TQue<QuePosition::VECOUT, 1> ms_outQueue;  
    TQue<QuePosition::VECOUT, 1> mbl_outQueue;

    TBuf<TPosition::VECCALC> temp0;   
    TBuf<TPosition::VECCALC> e_out1;     
    TBuf<TPosition::VECCALC> table;             
    TBuf<TPosition::VECCALC> merge;      
    TBuf<TPosition::VECCALC> cmbl;          
    TBuf<TPosition::VECCALC> mask7;           

    GlobalTensor<T> input;                   
    GlobalTensor<T> table_input;        
    GlobalTensor<T> mbl_output;         
    GlobalTensor<T> e_output0;                
    GlobalTensor<T> e_output1;              
    GlobalTensor<T> ms_output;            
    GlobalTensor<T> blockCompSizeOutput;         

    uint32_t blockId;                       
    uint32_t blockNum;                         
    uint32_t computeNum;              
    uint32_t tileLength;                   
    uint32_t tileNum;                 
    uint32_t threadblockNum;              
    uint32_t datablockNum;        
    uint32_t datablockSize;                   
    uint32_t bufferSize;                       

    uint32_t srcShape_cmp[2];             
    uint32_t dstShape_cmp[2];           
};

template <typename T>
class CompressKernelFP16
{
public:
    __aicore__ inline CompressKernelFP16() {}

    __aicore__ inline void Init(TPipe *pipe,
                                uint32_t datablockNum, 
                                uint32_t datablockSize,  
                                uint32_t elementNum,    
                                uint32_t tileLength,    
                                __gm__ uint8_t *srcDevice,       
                                __gm__ uint8_t *ms0Global,        
                                __gm__ uint8_t *ms1Global,        
                                __gm__ uint8_t *e0Global,        
                                __gm__ uint8_t *mblGlobal,        
                                __gm__ uint8_t *e1Global,        
                                __gm__ uint8_t *histogramDevice,   
                                __gm__ uint8_t *blockCompSize)   
    {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();  
        this->blockNum = GetBlockNum();  
        
        this->computeNum = elementNum;
        this->tileLength = tileLength;
        this->tileNum = elementNum / tileLength;
        this->datablockNum = datablockNum; 
        this->datablockSize = datablockSize; 
        
        int datablockNumPerBLOCK = (datablockNum + blockNum - 1) / blockNum;
        this->bufferSize = (datablockSize * datablockNumPerBLOCK);

        srcShape_cmp[0] = 1;
        srcShape_cmp[1] = tileNum / 8 / sizeof(half);
        dstShape_cmp[0] = tileLength;
        dstShape_cmp[1] = tileNum / 8 / sizeof(half);

        input.SetGlobalBuffer((__gm__ T *)(srcDevice));              
        ms_output0.SetGlobalBuffer((__gm__ T *)(ms0Global));         
        ms_output1.SetGlobalBuffer((__gm__ T *)(ms1Global));            
        e_output0.SetGlobalBuffer((__gm__ T *)(e0Global));            
        mbl_output.SetGlobalBuffer((__gm__ T *)(mblGlobal));          
        e_output1.SetGlobalBuffer((__gm__ T *)(e1Global + bufferSize * blockId)); 
        blockCompSizeOutput.SetGlobalBuffer((__gm__ T *)(blockCompSize + 32 * blockId)); 

        pipe->InitBuffer(mbl_outQueue, BUFFER_NUM, tileLength * tileNum / 8);
    }

    __aicore__ inline void Process()
    {
        pipe->InitBuffer(merge, computeNum * sizeof(T)); 
        pipe->InitBuffer(mask7, 32);                       
        pipe->InitBuffer(temp0, computeNum * sizeof(float)); 
        pipe->InitBuffer(temp1, computeNum * sizeof(float)); 

        LocalTensor<T> tempLocal0 = temp0.Get<T>();
        LocalTensor<T> tempLocal1 = temp1.Get<T>();
        LocalTensor<T> mergeLocal = merge.Get<T>();
        LocalTensor<T> mask7Local = mask7.Get<T>();

        AIV_WITH_BARRIER(Duplicate, mergeLocal, (T)0, computeNum); 
        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)7, 32 / sizeof(T));

        uint64_t outerNum = 0;           
        uint32_t totalouterNum = 0;       
        uint32_t totalcompressedSize = 0; 
        uint32_t cumulated_amount = 0;    
        uint32_t new_cumulated_amount = 0; 
        uint32_t low_write_num = 0;       
        uint32_t high_unwrite_num = 0;    
        uint32_t write_offset = 0;     
        
        for (uint32_t i = blockId; i < datablockNum; i += blockNum)
        {
            Compute(i,
                    cumulated_amount,
                    low_write_num,
                    high_unwrite_num,
                    write_offset,
                    outerNum,
                    tempLocal0,
                    tempLocal1,
                    mergeLocal,
                    mask7Local);
            
            totalouterNum = totalouterNum + outerNum;
            CopyOut(i);
        }
        
        totalouterNum = (totalouterNum + 256 - 1) / 256 * 256;
        totalcompressedSize = totalcompressedSize + totalouterNum * 2 / 8;
        
        cumulated_amount = (cumulated_amount + 256 - 1) / 256 * 256;
        
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)14, cumulated_amount);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)14, cumulated_amount);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 2], mergeLocal[cumulated_amount / 2], (uint16_t)2, cumulated_amount / 2);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 2], cumulated_amount / 2);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4], (uint16_t)4, cumulated_amount / 4);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 4], cumulated_amount / 4);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 8], mergeLocal[cumulated_amount / 8], (uint16_t)8, cumulated_amount / 8);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 8], cumulated_amount / 8);

        AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, cumulated_amount * 2 / 16);

        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)0, 32 / sizeof(T));
        mask7Local.template ReinterpretCast<int32_t>()(0) = totalcompressedSize;
        PipeBarrier<PIPE_ALL>(); 
        AIV_WITH_BARRIER(DataCopy, blockCompSizeOutput, mask7Local, 32 / sizeof(T));
    }

private:
    __aicore__ inline void Compute(uint32_t i,
                                    uint32_t &cumulated_amount,
                                    uint32_t &low_write_num,
                                    uint32_t &high_unwrite_num,
                                    uint32_t &write_offset,
                                    uint64_t &outerNum,
                                    LocalTensor<T> &tempLocal0,
                                    LocalTensor<T> &tempLocal1,
                                    LocalTensor<T> &mergeLocal,
                                    LocalTensor<T> &mask7Local)
    {
        LocalTensor<T> compareMask = mbl_outQueue.AllocTensor<T>();

        uint32_t offset = i * (computeNum * sizeof(uint16_t) / sizeof(T));
        AIV_WITH_BARRIER(DataCopy, tempLocal0, input[offset], computeNum * sizeof(uint16_t) / sizeof(T));

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal0, (uint16_t)6, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)10, computeNum);
        AIV_WITH_BARRIER(Or, tempLocal0, tempLocal0, tempLocal1, computeNum);

        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal0, (uint16_t)5, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1[computeNum], tempLocal1, (uint16_t)8, computeNum);
        
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum + computeNum / 2], tempLocal1[computeNum + computeNum / 2], (uint16_t)3, computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum], tempLocal1[computeNum], tempLocal1[computeNum + computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum + computeNum / 4], tempLocal1[computeNum + computeNum / 4], (uint16_t)6, computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum], tempLocal1[computeNum], tempLocal1[computeNum + computeNum / 4], computeNum / 4);

        AIV_WITH_BARRIER(ShiftRight, tempLocal1[computeNum + computeNum / 4], tempLocal1[computeNum], (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum + computeNum / 4 + computeNum / 8], tempLocal1[computeNum + computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum + computeNum / 4], tempLocal1[computeNum + computeNum / 4], tempLocal1[computeNum + computeNum / 4 + computeNum / 8], computeNum / 8);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal1, (uint16_t)8, computeNum + computeNum / 4 + computeNum / 8);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal1, (uint16_t)8, (computeNum + computeNum / 4 + computeNum / 8) / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[(computeNum + computeNum / 4 + computeNum / 8) / 2], (computeNum + computeNum / 4 + computeNum / 8) / 2);
        
        AIV_WITH_BARRIER(DataCopy, ms_output1[i * (computeNum * 11 / 8 / sizeof(T))], tempLocal1, computeNum * 11 / 8 / sizeof(T));

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint16_t)11, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)11, computeNum);

        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>(), (int16_t)(-11), computeNum);
        AIV_WITH_BARRIER(Muls, tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>(), (int16_t)(-1), computeNum);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint16_t)11, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)11, computeNum);

        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal0, tempLocal0[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 4], computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 8], computeNum / 8);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 16], computeNum / 16);

        AIV_WITH_BARRIER(CompareScalar, compareMask.template ReinterpretCast<uint8_t>(), tempLocal1.template ReinterpretCast<half>(),
                      (mask7Local.template ReinterpretCast<half>())(0), CMPMODE::GT, tileNum);
        AIV_WITH_BARRIER(DataCopy, mbl_output[i * (tileNum / 8 / sizeof(T))], compareMask, tileNum / 8 / sizeof(T));

        AIV_WITH_BARRIER(DataCopy, compareMask[64], compareMask, 64);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 1], compareMask, 64 << 1);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 2], compareMask, 64 << 2);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 3], compareMask, 64 << 3);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal0, (uint16_t)13, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal1, (uint16_t)13, computeNum);
        
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 2], tempLocal1[computeNum / 2], (uint16_t)3, computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 4], tempLocal1[computeNum / 4], (uint16_t)6, computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 4], computeNum / 4);

        AIV_WITH_BARRIER(ShiftRight, tempLocal1[computeNum / 4], tempLocal1, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal1, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal1, (uint16_t)8, computeNum / 4);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 4  + computeNum / 8], tempLocal1[computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum / 4], tempLocal1[computeNum / 4], tempLocal1[computeNum / 4  + computeNum / 8], computeNum / 8);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[(computeNum * 3 / 16)], tempLocal1[(computeNum * 3 / 16)], (uint16_t)8, computeNum * 3 / 16);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[(computeNum * 3 / 16)], computeNum * 3 / 16);

        AIV_WITH_BARRIER(DataCopy, e_output0[i * (computeNum * 3 / 16)], tempLocal1, computeNum * 3 / 16);

        AIV_WITH_BARRIER(GatherMask, tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<half>(),
                   compareMask.template ReinterpretCast<uint16_t>(), true, computeNum, {1, 1, 1, 0}, outerNum);

        if(cumulated_amount + outerNum >= computeNum){
            low_write_num = computeNum - cumulated_amount;
            high_unwrite_num = outerNum - low_write_num;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], tempLocal1, (uint16_t)3, low_write_num);
            
            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 2], mergeLocal[computeNum / 2], (uint16_t)2, computeNum / 2);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 2], computeNum / 2);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 4], mergeLocal[computeNum / 4], (uint16_t)4, computeNum / 4);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 4], computeNum / 4);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 8], mergeLocal[computeNum / 8], (uint16_t)8, computeNum / 8);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 8], computeNum / 8);

            AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, (computeNum * 2 / 16));
            write_offset = write_offset + computeNum * 2 / 16;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal, tempLocal1[low_write_num], (uint16_t)3, high_unwrite_num);
            cumulated_amount = high_unwrite_num;
        }
        else {
            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], tempLocal1, (uint16_t)3, outerNum);
            cumulated_amount = cumulated_amount + outerNum;
        }

        mbl_outQueue.EnQue(compareMask);
    }

    __aicore__ inline void CopyOut(uint32_t i)
    {
        LocalTensor<T> compareMask = mbl_outQueue.DeQue<T>();
        AIV_WITH_BARRIER(DataCopy, mbl_output[i * (tileNum / 8 / sizeof(T))], compareMask, tileNum / 8 / sizeof(T));
        mbl_outQueue.FreeTensor(compareMask);
    }

private:
    TPipe *pipe;                
    TQue<QuePosition::VECOUT, 1> mbl_outQueue;

    TBuf<TPosition::VECCALC> temp0; 
    TBuf<TPosition::VECCALC> temp1; 
    TBuf<TPosition::VECCALC> merge; 
    TBuf<TPosition::VECCALC> mask7;

    GlobalTensor<T> input;     
    GlobalTensor<T> ms_output0;     
    GlobalTensor<T> ms_output1;    
    GlobalTensor<T> e_output0; 
    GlobalTensor<T> mbl_output;        
    GlobalTensor<T> blockCompSizeOutput;
    GlobalTensor<T> e_output1;        

    uint32_t blockId;  
    uint32_t blockNum;         
    uint32_t computeNum;      
    uint32_t tileLength;   
    uint32_t tileNum;         
    uint32_t threadblockNum;   
    uint32_t datablockNum;  
    uint32_t datablockSize; 
    uint32_t bufferSize;    

    uint32_t srcShape_cmp[2];  
    uint32_t dstShape_cmp[2];  
};

template <typename T>
class CompressKernelFP32
{
public:
    __aicore__ inline CompressKernelFP32() {}

    __aicore__ inline void Init(TPipe *pipe,
                                uint32_t datablockNum,
                                uint32_t datablockSize,
                                uint32_t elementNum,
                                uint32_t tileLength,
                                __gm__ uint8_t *srcDevice,          // e_input
                                __gm__ uint8_t *ms0Global,          // ms0_output
                                __gm__ uint8_t *ms1Global,          // ms1_output
                                __gm__ uint8_t *e0Global,           // e0_output
                                __gm__ uint8_t *mblGlobal,          // mbl_output
                                __gm__ uint8_t *e1Global,           // e1_output
                                __gm__ uint8_t *histogramDevice,    // table_input
                                __gm__ uint8_t *blockCompSize)
    {
        this->pipe = pipe;
        this->blockId = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->computeNum = elementNum;
        this->tileLength = tileLength;
        this->tileNum = elementNum / tileLength;
        this->datablockNum = datablockNum;
        this->datablockSize = datablockSize;
        int datablockNumPerBLOCK = (datablockNum + blockNum - 1) / blockNum;
        this->bufferSize = (datablockSize * datablockNumPerBLOCK);

        srcShape_cmp[0] = 1;
        srcShape_cmp[1] = tileNum / 8 / sizeof(half);
        dstShape_cmp[0] = tileLength;
        dstShape_cmp[1] = tileNum / 8 / sizeof(half);

        input.SetGlobalBuffer((__gm__ T *)(srcDevice));
        ms_output0.SetGlobalBuffer((__gm__ T *)(ms0Global));
        ms_output1.SetGlobalBuffer((__gm__ T *)(ms1Global));
        e_output0.SetGlobalBuffer((__gm__ T *)(e0Global));
        mbl_output.SetGlobalBuffer((__gm__ T *)(mblGlobal));
        e_output1.SetGlobalBuffer((__gm__ T *)(e1Global + bufferSize * blockId));
        blockCompSizeOutput.SetGlobalBuffer((__gm__ T *)(blockCompSize + 32 * blockId));

        pipe->InitBuffer(mbl_outQueue, BUFFER_NUM, tileLength * tileNum / 8);// 128B
    }

    __aicore__ inline void Process()
    {
        pipe->InitBuffer(merge, computeNum * sizeof(T));// 32KB
        pipe->InitBuffer(mask7, 32);// 32B
        LocalTensor<T> mergeLocal = merge.Get<T>();
        LocalTensor<T> mask7Local = mask7.Get<T>();

        pipe->InitBuffer(temp0, computeNum * sizeof(float));// 64KB
        pipe->InitBuffer(temp1, computeNum * sizeof(float));// 64KB

        LocalTensor<T> tempLocal0 = temp0.Get<T>();
        LocalTensor<T> tempLocal1 = temp1.Get<T>();
        AIV_WITH_BARRIER(Duplicate, mergeLocal, (T)0, computeNum);
        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)7, 32 / sizeof(T));

        uint64_t outerNum = 0;
        uint32_t totalouterNum = 0;
        uint32_t totalcompressedSize = 0;
        uint32_t cumulated_amount = 0;
        uint32_t new_cumulated_amount = 0;
        uint32_t low_write_num = 0;
        uint32_t high_unwrite_num = 0;
        uint32_t write_offset = 0;
        for (uint32_t i = blockId; i < datablockNum; i += blockNum)
        {
            Compute(i,
                    cumulated_amount,
                    low_write_num,
                    high_unwrite_num,
                    write_offset,
                    outerNum,
                    tempLocal0,
                    tempLocal1,
                    mergeLocal,
                    mask7Local
                );
            totalouterNum = totalouterNum + outerNum;
            CopyOut(i);
        }
        totalouterNum = (totalouterNum + 256 - 1) / 256 * 256;
        totalcompressedSize = totalcompressedSize 
        + totalouterNum * 3 / 8
        ;

        cumulated_amount = (cumulated_amount + 256 - 1) / 256 * 256;
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)13, cumulated_amount);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)13, cumulated_amount);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 2], mergeLocal[cumulated_amount / 2], (uint16_t)3, cumulated_amount / 2);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 2], cumulated_amount / 2);
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4], (uint16_t)6, cumulated_amount / 4);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[cumulated_amount / 4], cumulated_amount / 4);

        AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount / 4], mergeLocal, (uint16_t)8, cumulated_amount / 4);
        AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, cumulated_amount / 4);
        AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, cumulated_amount / 4);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[cumulated_amount / 4  + cumulated_amount / 8], mergeLocal[cumulated_amount / 4 + cumulated_amount / 8], (uint16_t)4, cumulated_amount / 8);
        AIV_WITH_BARRIER(Or, mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4], mergeLocal[cumulated_amount / 4  + cumulated_amount / 8], cumulated_amount / 8);

        AIV_WITH_BARRIER(ShiftLeft, mergeLocal[(cumulated_amount * 3 / 16)], mergeLocal[(cumulated_amount * 3 / 16)], (uint16_t)8, cumulated_amount * 3 / 16);
        AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[(cumulated_amount * 3 / 16)], cumulated_amount * 3 / 16);

        AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, cumulated_amount * 3 / 16);

        AIV_WITH_BARRIER(Duplicate, mask7Local, (T)0, 32 / sizeof(T));
        mask7Local.template ReinterpretCast<int32_t>()(0) = totalcompressedSize;
        AIV_WITH_BARRIER(DataCopy, blockCompSizeOutput, mask7Local, 32 / sizeof(T));
    }

private:

    __aicore__ inline void Compute( uint32_t i,
                                    uint32_t &cumulated_amount,
                                    uint32_t &low_write_num,
                                    uint32_t &high_unwrite_num,
                                    uint32_t &write_offset,
                                    uint64_t &outerNum,
                                    LocalTensor<T> &tempLocal0,
                                    LocalTensor<T> &tempLocal1,// 64KB
                                    LocalTensor<T> &mergeLocal,// 32KB
                                    LocalTensor<T> &mask7Local
                                )// 3， 6
    {   
        LocalTensor<T> compareMask = mbl_outQueue.AllocTensor<T>();// 1024/8 = 128bytes

        uint32_t offset = i * (computeNum * sizeof(uint32_t) / sizeof(T));
        AIV_WITH_BARRIER(DataCopy, tempLocal0, input[offset], computeNum * sizeof(uint32_t) / sizeof(T));

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1.template ReinterpretCast<uint32_t>(), tempLocal0.template ReinterpretCast<uint32_t>(), (uint32_t)16, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1.template ReinterpretCast<uint32_t>(), tempLocal1.template ReinterpretCast<uint32_t>(), (uint32_t)16, computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1.template ReinterpretCast<uint32_t>(), tempLocal1.template ReinterpretCast<uint32_t>(), tempLocal1.template ReinterpretCast<uint32_t>()[computeNum / 2], computeNum / 2 * 2);
        AIV_WITH_BARRIER(DataCopy, ms_output0[i * computeNum], tempLocal1, computeNum);
        
        AIV_WITH_BARRIER(Cast, tempLocal0.template ReinterpretCast<bfloat16_t>(), tempLocal0.template ReinterpretCast<float>(), RoundMode::CAST_TRUNC, computeNum);
        
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal0, (uint16_t)9, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)7, computeNum);
        AIV_WITH_BARRIER(Or, tempLocal0, tempLocal0, tempLocal1, computeNum);

        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal0, (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 2], tempLocal1[computeNum / 2], (uint16_t)8, computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 2], computeNum / 2);

        AIV_WITH_BARRIER(DataCopy, ms_output1[i * (computeNum / sizeof(T))], tempLocal1, computeNum / sizeof(T));

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint16_t)8, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)8, computeNum);

        AIV_WITH_BARRIER(Adds, tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>(), (int16_t)(-121), computeNum);
        AIV_WITH_BARRIER(Muls, tempLocal0.template ReinterpretCast<int16_t>(), tempLocal0.template ReinterpretCast<int16_t>(), (int16_t)(-1), computeNum);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal0, tempLocal0, (uint16_t)10, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal0, tempLocal0, (uint16_t)10, computeNum);

        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal0, tempLocal0[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 4], computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 8], computeNum / 8);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 16], computeNum / 16);

        AIV_WITH_BARRIER(CompareScalar, compareMask.template ReinterpretCast<uint8_t>(), tempLocal1.template ReinterpretCast<half>(),
                      (mask7Local.template ReinterpretCast<half>())(0), CMPMODE::GT, tileNum);
        AIV_WITH_BARRIER(DataCopy, mbl_output[i * (tileNum / 8 / sizeof(T))], compareMask, tileNum / 8 / sizeof(T));

        AIV_WITH_BARRIER(DataCopy, compareMask[64], compareMask, 64);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 1], compareMask, 64 << 1);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 2], compareMask, 64 << 2);
        AIV_WITH_BARRIER(DataCopy, compareMask[64 << 3], compareMask, 64 << 3);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal0, (uint16_t)13, computeNum);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal1, (uint16_t)13, computeNum);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 2], tempLocal1[computeNum / 2], (uint16_t)3, computeNum / 2);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 2], computeNum / 2);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 4], tempLocal1[computeNum / 4], (uint16_t)6, computeNum / 4);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[computeNum / 4], computeNum / 4);

        AIV_WITH_BARRIER(ShiftRight, tempLocal1[computeNum / 4], tempLocal1, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftLeft, tempLocal1, tempLocal1, (uint16_t)8, computeNum / 4);
        AIV_WITH_BARRIER(ShiftRight, tempLocal1, tempLocal1, (uint16_t)8, computeNum / 4);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[computeNum / 4  + computeNum / 8], tempLocal1[computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
        AIV_WITH_BARRIER(Or, tempLocal1[computeNum / 4], tempLocal1[computeNum / 4], tempLocal1[computeNum / 4  + computeNum / 8], computeNum / 8);

        AIV_WITH_BARRIER(ShiftLeft, tempLocal1[(computeNum * 3 / 16)], tempLocal1[(computeNum * 3 / 16)], (uint16_t)8, computeNum * 3 / 16);
        AIV_WITH_BARRIER(Or, tempLocal1, tempLocal1, tempLocal1[(computeNum * 3 / 16)], computeNum * 3 / 16);

        AIV_WITH_BARRIER(DataCopy, e_output0[i * (computeNum * 3 / 16)], tempLocal1, computeNum * 3 / 16);

        AIV_WITH_BARRIER(GatherMask, tempLocal1.template ReinterpretCast<half>(), tempLocal0.template ReinterpretCast<half>(),
                   compareMask.template ReinterpretCast<uint16_t>(), true, computeNum, {1, 1, 1, 0}, outerNum);

        if(cumulated_amount + outerNum >= computeNum){
            low_write_num = computeNum - cumulated_amount;
            high_unwrite_num = outerNum - low_write_num;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], tempLocal1, (uint16_t)3, low_write_num);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 2], mergeLocal[computeNum / 2], (uint16_t)3, computeNum / 2);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 2], computeNum / 2);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 4], mergeLocal[computeNum / 4], (uint16_t)6, computeNum / 4);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[computeNum / 4], computeNum / 4);

            AIV_WITH_BARRIER(ShiftRight, mergeLocal[computeNum / 4], mergeLocal, (uint16_t)8, computeNum / 4);
            AIV_WITH_BARRIER(ShiftLeft, mergeLocal, mergeLocal, (uint16_t)8, computeNum / 4);
            AIV_WITH_BARRIER(ShiftRight, mergeLocal, mergeLocal, (uint16_t)8, computeNum / 4);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[computeNum / 4  + computeNum / 8], mergeLocal[computeNum / 4 + computeNum / 8], (uint16_t)4, computeNum / 8);
            AIV_WITH_BARRIER(Or, mergeLocal[computeNum / 4], mergeLocal[computeNum / 4], mergeLocal[computeNum / 4  + computeNum / 8], computeNum / 8);

            AIV_WITH_BARRIER(ShiftLeft, mergeLocal[(computeNum * 3 / 16)], mergeLocal[(computeNum * 3 / 16)], (uint16_t)8, computeNum * 3 / 16);
            AIV_WITH_BARRIER(Or, mergeLocal, mergeLocal, mergeLocal[(computeNum * 3 / 16)], computeNum * 3 / 16);

            AIV_WITH_BARRIER(DataCopy, e_output1[write_offset], mergeLocal, (computeNum * 3 / 16));
            write_offset = write_offset + computeNum * 3 / 16;

            AIV_WITH_BARRIER(ShiftRight, mergeLocal, tempLocal1[low_write_num], (uint16_t)3, high_unwrite_num);
            cumulated_amount = high_unwrite_num;
        }
        else {
            AIV_WITH_BARRIER(ShiftRight, mergeLocal[cumulated_amount], tempLocal1, (uint16_t)3, outerNum);
     
            cumulated_amount = cumulated_amount + outerNum;
        }

        mbl_outQueue.EnQue(compareMask);
    }

    __aicore__ inline void CopyOut(uint32_t i)
    {
        LocalTensor<T> compareMask = mbl_outQueue.DeQue<T>();

        AIV_WITH_BARRIER(DataCopy, mbl_output[i * (tileNum / 8 / sizeof(T))], compareMask, tileNum / 8 / sizeof(T));

        mbl_outQueue.FreeTensor(compareMask);
    }

private:
    TPipe *pipe;

    TQue<QuePosition::VECOUT, 1> mbl_outQueue;

    TBuf<TPosition::VECCALC> temp0;
    TBuf<TPosition::VECCALC> temp1;
    TBuf<TPosition::VECCALC> merge;
    TBuf<TPosition::VECCALC> mask7;

    GlobalTensor<T> input;
    GlobalTensor<T> ms_output0;
    GlobalTensor<T> ms_output1;
    GlobalTensor<T> e_output0;
    GlobalTensor<T> mbl_output;
    GlobalTensor<T> blockCompSizeOutput;
    GlobalTensor<T> e_output1;

    uint32_t blockId;
    uint32_t blockNum;
    uint32_t computeNum;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t threadblockNum;
    uint32_t datablockNum;
    uint32_t datablockSize;
    uint32_t bufferSize;

    uint32_t srcShape_cmp[2];
    uint32_t dstShape_cmp[2];
};

// BF16
__global__ __aicore__ void compBF16(uint32_t datablockNum,
                                    uint32_t datablockSize,
                                    uint32_t elementNum,
                                    uint32_t tileLength,
                                    __gm__ uint8_t* srcDevice,     
                                    __gm__ uint8_t* msGlobal,
                                    __gm__ uint8_t* e0Global,   
                                    __gm__ uint8_t* mblGlobal,   
                                    __gm__ uint8_t* e1Global,    
                                    __gm__ uint8_t* histogramDevice,
                                    __gm__ uint8_t* blockCompSize)
{
    TPipe pipe;
    CompressKernelBF16<uint16_t> op;
    op.Init(&pipe, datablockNum, datablockSize, elementNum, tileLength, 
            srcDevice, 
            msGlobal, 
            e0Global, 
            mblGlobal, 
            e1Global, 
            histogramDevice, 
            blockCompSize);
    op.Process();
}

// FP16
__global__ __aicore__ void compFP16(uint32_t datablockNum,
                                    uint32_t datablockSize,
                                    uint32_t elementNum,
                                    uint32_t tileLength,
                                    __gm__ uint8_t* srcDevice,  
                                    __gm__ uint8_t* ms0Global,    
                                    __gm__ uint8_t* ms1Global,     
                                    __gm__ uint8_t* e0Global,       
                                    __gm__ uint8_t* mblGlobal,    
                                    __gm__ uint8_t* e1Global,      
                                    __gm__ uint8_t* histogramDevice, 
                                    __gm__ uint8_t* blockCompSize) 
{
    TPipe pipe;
    CompressKernelFP16<uint16_t> op;
    op.Init(&pipe, datablockNum, datablockSize, elementNum, tileLength, 
            srcDevice, 
            ms0Global, 
            ms1Global,
            e0Global, 
            mblGlobal, 
            e1Global, 
            histogramDevice, 
            blockCompSize);
    op.Process();
}

// FP32
__global__ __aicore__ void compFP32(uint32_t datablockNum,
                                    uint32_t datablockSize,
                                    uint32_t elementNum,
                                    uint32_t tileLength,
                                    __gm__ uint8_t* srcDevice,       // e_input
                                    __gm__ uint8_t* ms0Global,        // ms0_output
                                    __gm__ uint8_t* ms1Global,        // ms1_output
                                    __gm__ uint8_t* e0Global,         // e0_output
                                    __gm__ uint8_t* mblGlobal,       // mbl_output
                                    __gm__ uint8_t* e1Global,         // e1_output
                                    __gm__ uint8_t* histogramDevice, // table_input
                                    __gm__ uint8_t* blockCompSize)
{
    TPipe pipe;
    CompressKernelFP32<uint16_t> op;
    op.Init(&pipe, datablockNum, datablockSize, elementNum, tileLength, 
            srcDevice, 
            ms0Global, 
            ms1Global,
            e0Global, 
            mblGlobal, 
            e1Global, 
            histogramDevice, 
            blockCompSize);
    op.Process();
}

extern "C" void enec_compress(Header *cphd, void *stream, uint8_t* srcDevice, uint8_t* compressedDevice, uint8_t* compressedFinal, uint8_t* histogramDevice, uint8_t* blockCompSize)
{
    switch (cphd->dataType)
    {
    case 0: // BF16
    { 
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint16_t);
        compBF16<<<BLOCK_NUM, nullptr, stream>>>(
            cphd->dataBlockNum, cphd->dataBlockSize, elementNum, cphd->tileLength, 
            srcDevice, 
            getMsdata(cphd, compressedFinal), 
            getEdata(cphd, compressedFinal),
            getMbl(cphd, compressedFinal), 
            getCompressed_exp(cphd, compressedDevice), 
            histogramDevice, 
            blockCompSize);
        break;
    }
    case 1: // FP16
    { 
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint16_t);
        compFP16<<<BLOCK_NUM, nullptr, stream>>>(
            cphd->dataBlockNum, cphd->dataBlockSize, elementNum, cphd->tileLength, 
            srcDevice, 
            getMs0data(cphd, compressedFinal), 
            getMs1data(cphd, compressedFinal), 
            getEdata(cphd, compressedFinal),
            getMbl(cphd, compressedFinal), 
            getCompressed_exp(cphd, compressedDevice), 
            histogramDevice, 
            blockCompSize);
        break;
    }
    case 2: // FP32
    { 
        uint32_t elementNum = cphd->dataBlockSize / sizeof(uint32_t);
        compFP32<<<BLOCK_NUM, nullptr, stream>>>(
            cphd->dataBlockNum, cphd->dataBlockSize, elementNum, cphd->tileLength, 
            srcDevice, 
            getMs0data(cphd, compressedFinal), 
            getMs1data(cphd, compressedFinal), 
            getEdata(cphd, compressedFinal),
            getMbl(cphd, compressedFinal), 
            getCompressed_exp(cphd, compressedDevice), 
            histogramDevice, 
            blockCompSize);
        break;
    }
    default:
        return;
    }
}