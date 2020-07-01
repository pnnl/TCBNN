// ---------------------------------------------------------------------------
// File: param.h
// Define basic layer objects.
// ---------------------------------------------------------------------------
// See our arXiv paper for detail: https://arxiv.org/abs/2006.16578
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/TCBNN
// PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850
// BSD Lincese.
// Richland, 99352, WA, USA. June-30-2020.
// ---------------------------------------------------------------------------



#ifndef PARAM_H
#define PARAM_H

#include "utility.h"

const int dev=0;


__global__ void PackFcWeight128(const float* __restrict__ A, uin32* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    GET_WARPID;
    const int gdx = STEP128(A_height);
    const int gdy = STEP8(A_width);
    const int lx = (warpid&0x3);
    const int ly = (warpid>>2);
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid % gdx;
        const int by = bid / gdx;
        float f0 = ( (bx*128+lx*32+laneid<A_height) && (by*8+ly<A_width) )?
            A[(bx*128+lx*32+laneid)*A_width+by*8+ly]:-1.0f;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0?1:0));
        if (laneid==0) B[(by*8+ly)*gdx*4+bx*4+lx] = r0;
    }
}

__global__ void PackFcWeight128FMT(const float* __restrict__ A, uin32* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    GET_WARPID;
    const int gdx = STEP128(A_height);
    const int gdy = STEP8(A_width);
    const int lx = (warpid&0x3);
    const int ly = (warpid>>2);
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid % gdx;
        const int by = bid / gdx;

        float f0 = ( (bx*128+lx*32+laneid<A_height) && (by*8+ly<A_width) )?
            A[(bx*128+lx*32+laneid)*A_width+by*8+ly]:-1.0f;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0?1:0));
        if (laneid==0) B[(by*gdx+bx)*32+warpid] = r0;
    }
}

__global__ void UnPackFcOutput128(const uin32* __restrict__ A, float* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    GET_WARPID;
    const int gdy = STEP128(A_width);
    const int gdx = STEP8(A_height);
    const int lx = (warpid>>2);
    const int ly = (warpid&0x3);
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid / gdy;
        const int by = bid % gdy;
        unsigned r0 = A[(bx*8+lx)*gdy*4+by*4+ly];
        if ((bx*8+lx<A_height) && (by*128+ly*32+laneid<A_width))
            B[(bx*8+lx)*A_width+by*128+ly*32+laneid] = 2*(float)((r0>>(31-laneid)) & 0x1)-1;
    }
}




__global__ void UnPackFcOutput128FMT(const uin32* __restrict__ A, float* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    GET_WARPID;
    const int gdy = STEP128(A_width);
    const int gdx = STEP8(A_height);
    const int lx = (warpid>>2);
    const int ly = (warpid&0x3);
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid / gdy;
        const int by = bid % gdy;
        unsigned r0 = A[(bx*gdy+by)*32+warpid];
        if ((bx*8+lx<A_height) && (by*128+ly*32+laneid<A_width))
            B[(bx*8+lx)*A_width+by*128+ly*32+laneid] = 2*(float)((r0>>(31-laneid)) & 0x1)-1;
    }
}









//Convert floating point input into 1-bit input layer
class In128LayerParam
{
    public:
        In128LayerParam(const char* _name, int _input_height, int _input_width)
            :input_height(_input_height), output_height(_input_height),
            input_width(_input_width), output_width(_input_width),
            input(NULL), input_gpu(NULL), output(NULL), output_gpu(NULL)
        {
            strncpy(name, _name, 8);
        }
        //input utility
        int input_size() { return input_height * input_width; }
        int input_bytes() { return input_size() * sizeof(float);}
        int input_bit_size() { return input_height * input_width; }
        int input_bit_bytes() { return input_bit_size() * sizeof(float);}
        //output utility
        int output_size() { return  output_height * output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        //binarize on row
        int output_bit_size() { return PAD8(output_height)*STEP128(output_width);}
        int output_bit_bytes() { return output_bit_size()*sizeof(uin128);}

        In128LayerParam* initialize(float* input)
        {
            CHECK_NULL_POINTER(input);
            this->input = input;
            SAFE_ALOC_GPU(input_gpu, input_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(input_gpu, input, 
                        input_bytes(), cudaMemcpyHostToDevice) );
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            return this->ready();
        }
        In128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(In128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(In128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        void set_output_gpu(uin32* _output_gpu) 
        { 
            this->output_gpu = _output_gpu; 
        }
        uin32* get_output_gpu()
        { 
            return this->output_gpu; 
        }
        uin32* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            const int size = output_size()*sizeof(float);
            float* full_output = NULL;
            SAFE_ALOC_HOST(full_output, size);
            float* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
            int numThreads = 1024;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            int numBlocksPerSm;
#ifdef NEWFMT
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
                    numThreads, 0);
            UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_gpu, full_output_gpu, output_height, output_width);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, 
                    numThreads, 0);
            UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_gpu, full_output_gpu, output_height, output_width);
#endif

            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, 
                        cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
            return full_output;
        }
        void release() 
        {
            SAFE_FREE_GPU(input_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
            SAFE_FREE_HOST(output);
        }
        ~In128LayerParam() { release(); }
    public:
        float* input;
        float* input_gpu;
        uin32* output;
        uin32* output_gpu;

        int input_width;
        int input_height;
        int output_width;
        int output_height;

        In128LayerParam* gpu;
        char name[8];
};

class Fc128LayerParam
{
    public:
        Fc128LayerParam(const char* name, int _input_height, int _input_width, 
                int _weight_width) : 
            weight_height(_input_width), weight_width(_weight_width), 
            input_height(_input_height), input_width(_input_width),
            output_height(_input_height), output_width(_weight_width),
            bn_width(_weight_width), weight(NULL), weight_gpu(NULL),
            bn(NULL), bn_gpu(NULL), output(NULL), output_gpu(NULL),
            input(NULL), input_gpu(NULL), gpu(NULL)
        {
            strncpy(this->name, name, 8);
        }
        //row major
        int input_size() { return input_height*input_width;}
        int input_bytes() { return input_size()*sizeof(uin128);}
        int input_bit_size() { return PAD8(input_height)*STEP128(input_width);}
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}
        //colum major
        int weight_size() { return weight_height*weight_width;}
        int weight_bytes() { return weight_size()*sizeof(float);}
        int weight_bit_size() { return STEP128(weight_height)*PAD128(weight_width);}
        int weight_bit_bytes() { return weight_bit_size()*sizeof(uin128);}
        //row-major
        int output_size() { return output_height*output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        int output_bit_size() { return PAD8(output_height)*STEP128(output_width);}
        int output_bit_bytes() { return output_bit_size()*sizeof(uin128);}

        //batch-norm
        int bn_size() { return bn_width;}
        int bn_bytes() { return bn_size()*sizeof(float);}
        Fc128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(Fc128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this,
                        sizeof(Fc128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        void set_input_gpu(uin32* _input_gpu)
        {
            this->input_gpu = _input_gpu;
        }
        Fc128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu)
        {
            //Process weight
            SAFE_ALOC_HOST(weight, weight_bytes());
            launch_array(config_file, this->weight, weight_size());
            SAFE_ALOC_GPU(weight_gpu, weight_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->weight_gpu, 0, weight_bit_bytes()) );
            float* weight_float = NULL;
            SAFE_ALOC_GPU(weight_float, weight_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, 
                        weight_bytes(), cudaMemcpyHostToDevice) );
            int numThreads = 1024;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            int numBlocksPerSm;

#ifdef NEWFMT
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128FMT, 
                    numThreads, 0);
            PackFcWeight128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_float, weight_gpu, weight_height, weight_width);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, 
                    numThreads, 0);
            PackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_float, weight_gpu, weight_height, weight_width);
#endif
            CUDA_CHECK_KERNEL();
            SAFE_FREE_GPU(weight_float);
            //Process bn
            SAFE_ALOC_HOST(bn, bn_bytes());
            launch_array(config_file, this->bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );
            //Allocate output gpu
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );

            set_input_gpu(prev_layer_gpu);
            return this->ready();
        }
        uin32* get_output_gpu()
        {
            return this->output_gpu;
        }
        uin32* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            const int size = output_size()*sizeof(float);
            float* full_output = NULL;
            SAFE_ALOC_HOST(full_output, size);
            float* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, size);
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
            int numThreads = 1024;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            int numBlocksPerSm;
#ifdef NEWFMT
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128FMT, 
                    numThreads, 0);
            UnPackFcOutput128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_gpu, full_output_gpu, output_height, output_width);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, 
                    numThreads, 0);
            UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_gpu, full_output_gpu, output_height, output_width);
#endif
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, 
                        cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
            return full_output;
        }
        void release()
        {
            SAFE_FREE_HOST(weight);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(weight_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
        }
        ~Fc128LayerParam() { release(); }

    public:
        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        //Weight
        float* weight;
        uin32* weight_gpu;
        int weight_width;
        int weight_height;
        //Output
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;
        //Batch normalization
        float* bn;
        float* bn_gpu;
        int bn_width;
        //GPU shadow
        Fc128LayerParam* gpu;
        char name[8];
};


class Out128LayerParam
{
    public:
        Out128LayerParam(const char* name, int _input_height, 
                int _input_width, int _weight_width) :
            input_height(_input_height), input_width(_input_width),
            weight_height(_input_width), weight_width(_weight_width),
            output_height(_input_height), output_width(_weight_width),
            input(NULL), input_gpu(NULL), output(NULL), output_gpu(NULL),
            weight(NULL), weight_gpu(NULL)
        {
            strncpy(this->name, name, 8);
        }
        // row major
        int input_size() { return input_height*input_width;}
        int input_bytes() { return input_size()*sizeof(uin128);}
        int input_bit_size() { return PAD8(input_height)*STEP128(input_width);}
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}
        // colum major
        int weight_size() { return weight_height*weight_width;}
        int weight_bytes() { return weight_size()*sizeof(float);}
        int weight_bit_size() { return STEP128(weight_height)*PAD8(weight_width);}
        int weight_bit_bytes() { return weight_bit_size()*sizeof(uin128);}
        // row major
        int output_size() { return output_height * output_width;}
        int output_bytes() { return output_size()*sizeof(float);}
        int output_bit_size() { return output_height * output_width;}
        int output_bit_bytes() { return output_bit_size()*sizeof(float);}

        int bn_size() { return output_width;}
        int bn_bytes() { return output_width*sizeof(float); }
 
        Out128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(gpu, sizeof(Out128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(Out128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_input_gpu(uin32* input_gpu)
        {
            this->input_gpu = input_gpu;
        }
        float* get_output_gpu()
        {
            return this->output_gpu;
        }

        Out128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu)
        {
            SAFE_ALOC_HOST(weight, weight_bytes());
            launch_array(config_file, this->weight, weight_size());
            SAFE_ALOC_GPU(weight_gpu, weight_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->weight_gpu, 0, weight_bit_bytes()) );
            float* weight_float = NULL;
            SAFE_ALOC_GPU(weight_float, weight_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, 
                        weight_bytes(), cudaMemcpyHostToDevice) );
            int numThreads = 1024;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            int numBlocksPerSm;

#ifdef NEWFMT
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128FMT, 
                    numThreads, 0);
            PackFcWeight128FMT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_float, weight_gpu, weight_height, weight_width);
#else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, 
                    numThreads, 0);
            PackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    weight_float, weight_gpu, weight_height, weight_width);

#endif

            CUDA_CHECK_KERNEL();
            //BN
            SAFE_ALOC_HOST(bn_scale, bn_bytes());
            launch_array(config_file, this->bn_scale, bn_size());
            SAFE_ALOC_GPU(bn_scale_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_scale_gpu, bn_scale, 
                        bn_bytes(), cudaMemcpyHostToDevice) );
            SAFE_ALOC_HOST(bn_bias, bn_bytes());
            launch_array(config_file, this->bn_bias, bn_size());
            SAFE_ALOC_GPU(bn_bias_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_bias_gpu, bn_bias, 
                        bn_bytes(), cudaMemcpyHostToDevice) );

            SAFE_ALOC_GPU(output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bytes()) );
            set_input_gpu(prev_layer_gpu);
            return this->ready();
        }
        float* download_output()
        {
            SAFE_ALOC_HOST(output, output_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        void release()
        {
            SAFE_FREE_HOST(weight);
            SAFE_FREE_HOST(output);
            SAFE_FREE_HOST(bn_scale);
            SAFE_FREE_HOST(bn_bias);

            SAFE_FREE_GPU(weight_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(gpu);
            SAFE_FREE_GPU(bn_scale_gpu);
            SAFE_FREE_GPU(bn_bias_gpu);
        }
        ~Out128LayerParam() { release(); }
    public:
        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        //Weight
        float* weight;
        uin32* weight_gpu;
        int weight_width;
        int weight_height;
        //Output
        float* output;
        float* output_gpu;
        int output_height;
        int output_width;
        //Batch normalization
        bool has_bn;
        float* bn_scale;
        float* bn_scale_gpu;
        float* bn_bias;
        float* bn_bias_gpu;
        //GPU shadow
        Out128LayerParam* gpu;
        char name[8];
};



//================================ Convolution ====================================

__global__ void PackFiltersByInChannels128(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int input_channels, const int output_channels, 
        const int filter_width, const int filter_height) 
{
    GET_LANEID;
    GET_WARPID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over output_channels
    const int ins = 4*STEP128(input_channels);//condense C:in_channel into 32bit-unsigned

    for (int c=0; c<ins; c++) //iter over C:input_channels
    {
        // From shape[filter_height, filter_width, input_channels, output_channels] 
        float f0 = ((c*32+laneid)<input_channels)? filter[bx*input_channels*output_channels 
            + (c*32+laneid)*output_channels + by]:-1.0f;
        unsigned r0 = __brev(__ballot(f0>=0));
        if (laneid == 0) //avoid warp conflict
            // To shape[filter_height, filter_width, output_channels, input_channels/32]
            filter_binarized[bx*PAD32(output_channels)*ins+ by*ins + c] = r0;
    }
}


__global__ void PackFiltersByInChannels128FMT(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int input_channels, const int output_channels, 
        const int filter_width, const int filter_height) 
{
    GET_LANEID;
    GET_WARPID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over output_channels
    const int ins = 4*STEP128(input_channels);//condense C:in_channel into 32bit-unsigned

    for (int c=0; c<ins; c++) //iter over C:input_channels
    {
        // From shape[filter_height, filter_width, input_channels, output_channels] 
        float f0 = ((c*32+laneid)<input_channels)? filter[bx*input_channels*output_channels 
            + (c*32+laneid)*output_channels + by]:-1.0f;
        unsigned r0 = __brev(__ballot(f0>=0));
        if (laneid == 0) //avoid warp conflict
            //filter_binarized[bx*PAD32(output_channels)*ins+ by*ins + c] = r0;
            filter_binarized[bx*PAD32(output_channels)*ins
                + ((by/8)*(ins/4)+c/4)*32+(by%8)*4+(c%4)] = r0;
    }
}









/*
__global__ void PackFiltersByOutChannels32(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over input_channels
    const int ots = STEP32(out_channels);//condense K:output_channel into 32bit-unsigned

    for (int k=0; k<ots*4; k++) //iter over K:output_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        float f0 = ((k*32+laneid)<out_channels)? filter[bx*in_channels*out_channels 
            + by*out_channels + k*32 + laneid]:0;
        unsigned r0 = __brev(__ballot(f0>=0));
        // To shape[filter_height, filter_width, in_channels, out_channels/32]
        filter_binarized[bx*ots*in_channels+ by*ots + k] = r0;
    }
}  */


__global__ void PackFiltersByOutChannels32(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    GET_LANEID;
    GET_WARPID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over input_channels
    const int ots = STEP32(out_channels);//condense K:output_channel into 32bit-unsigned

    for (int k=0; k<ots; k++) //iter over K:output_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        float f0 = ((k*32+laneid)<out_channels)? filter[bx*in_channels*out_channels 
            + by*out_channels + k*32 + laneid]:-1.0f;
        unsigned r0 = __brev(__ballot(f0>=0));
        // To shape[filter_height, filter_width, in_channels, out_channels/32]
        filter_binarized[bx*ots*in_channels+ by*ots + k] = r0;
    }
}

__global__ void UnpackConvOutput32(const unsigned* __restrict__ input_binarized, 
        float* input, const int input_height, const int input_width,
        const int input_channels, const int batch) 
{
    GET_LANEID;
    GET_WARPID;
    const int bx = blockIdx.x;//input_width
    const int by = blockIdx.y;//input_height
    const int bz = blockIdx.z;//batch
    /*const int ins = STEP32(input_channels);//condense C:in_channel into 32bit-unsigned*/
    const int ins = STEP128(input_channels);//condense C:in_channel into 32bit-unsigned
    /*const int otb = STEP8(batch);*/
    for (int c=0; c<ins*4; c++) //iter over C:in_channels
    {
        // From shape[input_height, input_width, batch, in_channels/32] 
        unsigned r0 = input_binarized[by*input_width*PAD8(batch)*ins*4 + bx*PAD8(batch)*ins*4 
            + bz*ins*4 + c];
        // To shape[batch, input_height, input_width, in_channels]
        if (c*32+laneid<input_channels)
        {
            input[bz*input_height*input_width*input_channels + by*input_width*input_channels
                + bx*input_channels + c*32 + laneid] = 2*(float)((r0>>(31-laneid)) & 0x1)-1;
        }
    }
}


__global__ void UnpackConvOutput32FMT(const unsigned* __restrict__ input_binarized, 
        float* input, const int input_height, const int input_width,
        const int input_channels, const int batch) 
{
    GET_LANEID;
    GET_WARPID;
    const int bx = blockIdx.x;//input_width
    const int by = blockIdx.y;//input_height
    const int bz = blockIdx.z;//batch
    /*const int ins = STEP32(input_channels);//condense C:in_channel into 32bit-unsigned*/
    const int ins = STEP128(input_channels);//condense C:in_channel into 32bit-unsigned
    /*const int otb = STEP8(batch);*/
    for (int c=0; c<ins*4; c++) //iter over C:in_channels
    {
        // From shape[input_height, input_width, batch, in_channels/32] 
        // unsigned r0 = input_binarized[by*input_width*PAD8(batch)*ins*4 + bx*PAD8(batch)*ins*4 
        //    + bz*ins*4 + c];
        unsigned r0 = input_binarized[by*input_width*PAD8(batch)*ins*4 + bx*PAD8(batch)*ins*4 
            + ((bz/8)*ins+c/4)*32 + (bz%8)*4+(c%4)];

        // To shape[batch, input_height, input_width, in_channels]
        if (c*32+laneid<input_channels)
        {
            input[bz*input_height*input_width*input_channels + by*input_width*input_channels
                + bx*input_channels + c*32 + laneid] = 2*(float)((r0>>(31-laneid)) & 0x1)-1;
        }
    }
}












class InConv128LayerParam
{
    public:
        InConv128LayerParam(const char* name, int _input_height, int _input_width, 
                int _filter_height, int _filter_width, int _input_channels, 
                int _output_channels, int _batch, int _stride_height=1, 
                int _stride_width=1, bool _padding=true, int _pool_height=1, 
                int _pool_width=1, bool _save_residual=false) :
            input_height(_input_height), input_width(_input_width), filter_height(_filter_height),
            filter_width(_filter_width), input_channels(_input_channels),
            output_channels(_output_channels), batch(_batch), stride_height(_stride_height),
            stride_width(_stride_width), pool_height(_pool_height), pool_width(_pool_width),
            save_residual(_save_residual), padding(_padding), 
            bn(NULL), filter(NULL), output(NULL), output_gpu(NULL), input(NULL), 
            input_gpu(NULL), gpu(NULL), output_residual_gpu(NULL)

        {
            strncpy(this->name, name, 8);
            this->pad_h = padding?((( (input_height+stride_height-(input_height%stride_height))
                            /stride_height-1)*stride_height+filter_height-input_height)/2):0;
            this->pad_w = padding?((( (input_width+stride_width-(input_width%stride_width))
                                /stride_width-1)*stride_width+filter_width-input_width)/2):0; 
            int buf_height = padding?(input_height+stride_height-1)/stride_height
                    :((input_height-filter_height)/stride_height+1);
            int buf_width = padding?(input_width+stride_width-1)/stride_width
                    :((input_width-filter_width)/stride_width+1);
            output_height = (buf_height+pool_height-1)/pool_height;//pooling height
            output_width = (buf_width+pool_width-1)/pool_width; //pooling width
        }
        InConv128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            if (save_residual) CHECK_NULL_POINTER(output_residual_gpu);
            SAFE_ALOC_GPU(this->gpu, sizeof(InConv128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(InConv128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        int input_size() { return input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(float);}
        int input_bit_size() { return  input_channels*input_height*input_width*batch;}
        int input_bit_bytes() {return input_bit_size()*sizeof(float);}
        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() {return PAD128(output_channels)*STEP128(input_channels)
            *filter_height*filter_width;}
        int filter_bit_bytes() { return filter_bit_size() * sizeof(uin128);}
        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(uin32);}
        int output_bit_size() { return STEP128(output_channels)*output_height
            *output_width*PAD8(batch); }
        int output_bit_bytes() { return output_bit_size() * sizeof(uin128); }
        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}
        int residual_size() { return PAD128(output_channels)*PAD8(batch)*output_height
            *output_width;}
        int residual_bytes() { return residual_size()*sizeof(int);}

        InConv128LayerParam* initialize(float* input, FILE* config_file)
        {
            //Process input
            CHECK_NULL_POINTER(input);
            this->input = input;
            SAFE_ALOC_GPU(input_gpu, input_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(input_gpu, input, 
                        input_bytes(), cudaMemcpyHostToDevice) );
            //Process weight
            SAFE_ALOC_HOST(filter, filter_bytes());
            launch_array(config_file, this->filter, filter_size());
            SAFE_ALOC_GPU(filter_gpu, filter_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(filter_gpu, 0, filter_bit_bytes()) );
            float* filter_float = NULL;
            SAFE_ALOC_GPU(filter_float, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, 
                        filter_bytes(), cudaMemcpyHostToDevice) );
            //Binarize Filter
            PackFiltersByOutChannels32<<<dim3(filter_height*filter_width, input_channels), 32>>>(
                    filter_float, filter_gpu, input_channels, 
                    output_channels, filter_width, filter_height);
            SAFE_FREE_GPU(filter_float);
            //Process bn
            SAFE_ALOC_HOST(bn, bn_bytes());
            launch_array(config_file, bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(this->bn_gpu, this->bn, 
                        bn_bytes(), cudaMemcpyHostToDevice) );
            //Allocate output
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            //Allocate residual for saving
            if (save_residual)
            {
                SAFE_ALOC_GPU(output_residual_gpu, residual_bytes());
                CUDA_SAFE_CALL( cudaMemset(this->output_residual_gpu, 0, residual_bytes()) );
            }
            return this->ready();
        }
        uin32* get_output_gpu() { return this->output_gpu; }
        int* get_output_residual_gpu() { return this->output_residual_gpu; }

        unsigned* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            float* full_output = NULL;
            SAFE_ALOC_HOST(full_output, output_bytes());
            float* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, output_bytes()) );
#ifdef NEWFMT
            UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
#else
            UnpackConvOutput32<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
#endif
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, 
                        output_bytes(), cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_output_gpu);
            return full_output;
        }
        void release()
        {
            SAFE_FREE_HOST(filter);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(input_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(filter_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(gpu);
            if (save_residual) SAFE_FREE_GPU(output_residual_gpu);
        }
        ~InConv128LayerParam() { release(); }
    public:
        float* input;
        float* input_gpu;
        int input_width;
        int input_height;
        int input_channels;
        float* filter;
        uin32* filter_gpu;
        int filter_width;
        int filter_height;
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;
        int output_channels;
        bool padding;
        float* bn;
        float* bn_gpu;
        int batch;
        int stride_height;
        int stride_width;
        int pad_h;
        int pad_w;
        int pool_width;
        int pool_height;
        bool save_residual;
        int* output_residual_gpu;
        InConv128LayerParam* gpu;
        char name[8];
};

class Conv128LayerParam
{
    public:
        Conv128LayerParam(const char* name, int _input_height, int _input_width, 
                int _filter_height, int _filter_width, int _input_channels, 
                int _output_channels, int _batch, int _stride_height=1, 
                int _stride_width=1, bool _padding=true, int _pool_height=1, 
                int _pool_width=1, bool _ahead_fc=false, bool _save_residual=false,
                bool _inject_residual=false, int _residual_channels=0,
                bool _residual_pool=false) :

            input_height(_input_height), input_width(_input_width), 
            filter_height(_filter_height), filter_width(_filter_width),
            input_channels(_input_channels), output_channels(_output_channels),
            batch(_batch), stride_height(_stride_height), stride_width(_stride_width),
            pool_height(_pool_height), pool_width(_pool_width), ahead_fc(_ahead_fc),
            save_residual(_save_residual), inject_residual(_inject_residual),
            residual_channels(_residual_channels), padding(_padding), 
            residual_pool(_residual_pool),
            bn(NULL), bn_gpu(NULL), filter(NULL), filter_gpu(NULL), output(NULL),
            output_gpu(NULL), input(NULL), input_gpu(NULL), gpu(NULL), 
            output_residual_gpu(NULL), input_residual_gpu(NULL)
                
        {
            strncpy(this->name, name, 8);
            this->pad_h = padding?((( (input_height+stride_height-(input_height%stride_height))
                            /stride_height-1)*stride_height+filter_height-input_height)/2):0;
            this->pad_w = padding?((( (input_width+stride_width-(input_width%stride_width))
                                /stride_width-1)*stride_width+filter_width-input_width)/2):0; 
            int buf_height = padding?(input_height+stride_height-1)/stride_height
                :((input_height-filter_height)/stride_height+1);
            output_height = (buf_height+pool_height-1)/pool_height;//pooling height
            int buf_width = padding?(input_width+stride_width-1)/stride_width
                :((input_width-filter_width)/stride_width+1);
            output_width = (buf_width+pool_width-1)/pool_width; //pooling width
        }
        int input_size() { return input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(uin32);}
        int input_bit_size() { return STEP128(input_channels)*input_height
            *input_width*PAD8(batch); }
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}
        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() { return PAD32(output_channels)*STEP128(input_channels)
            *filter_height*filter_width; }
        int filter_bit_bytes() { return filter_bit_size()*sizeof(uin128);}
        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(uin32);}
        int output_bit_size() { return STEP128(output_channels)*output_height
            *output_width*PAD8(batch); }
        int output_bit_bytes() { return output_bit_size()*sizeof(uin128); }
        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}
        int residual_size() { return PAD128(output_channels)*PAD8(batch)*output_height
            *output_width;}
        int residual_bytes() { return residual_size()*sizeof(int);}

        Conv128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            if (save_residual) CHECK_NULL_POINTER(output_residual_gpu);
            if (inject_residual) CHECK_NULL_POINTER(input_residual_gpu);
            SAFE_ALOC_GPU(this->gpu, sizeof(Conv128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(Conv128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        void set_input_gpu(uin32* input_gpu) { this->input_gpu = input_gpu; }
        void set_input_residual_gpu(int* input_residual_gpu) { 
            this->input_residual_gpu = input_residual_gpu; }

        Conv128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu,
                int* input_residual_gpu = NULL)
        {
            //Process weight
            SAFE_ALOC_HOST(filter, filter_bytes());
            launch_array(config_file, filter, filter_size());
            SAFE_ALOC_GPU(filter_gpu, filter_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(filter_gpu, 0, filter_bit_bytes()) );
            float* filter_float = NULL;
            SAFE_ALOC_GPU(filter_float, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, 
                        filter_bytes(), cudaMemcpyHostToDevice) );

#ifdef NEWFMT
            PackFiltersByInChannels128FMT<<<dim3(filter_height*filter_width, output_channels),
                32>>>(filter_float, filter_gpu, input_channels, output_channels, 
                    filter_width, filter_height);
#else
            PackFiltersByInChannels128<<<dim3(filter_height*filter_width, output_channels),
                32>>>(filter_float, filter_gpu, input_channels, output_channels, 
                    filter_width, filter_height);
#endif
            SAFE_FREE_GPU(filter_float);
            //Process bn
            SAFE_ALOC_HOST(bn, bn_bytes());
            launch_array(config_file, bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );
            //Allocate output gpu
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            set_input_gpu(prev_layer_gpu);
            //Allocate residual for saving
            if (save_residual)
            {
                SAFE_ALOC_GPU(output_residual_gpu, residual_bytes());
                CUDA_SAFE_CALL( cudaMemset(output_residual_gpu, 0, residual_bytes()) );
            }
            //inject residual
            if (inject_residual) set_input_residual_gpu(input_residual_gpu);
            return this->ready();
        }
        uin32* get_output_gpu() { return this->output_gpu; }
        int* get_output_residual_gpu() { return this->output_residual_gpu; }

        unsigned* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            float* full_output = NULL;
            SAFE_ALOC_HOST(full_output, output_bytes());
            float* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, output_bytes()));
#ifdef NEWFMT
            UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
#else
            UnpackConvOutput32<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
#endif
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, 
                        output_bytes(), cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_output_gpu);
            return full_output;
        }
        void release()
        {
            SAFE_FREE_HOST(filter);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(filter_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(gpu);
            if (save_residual) SAFE_FREE_GPU(output_residual_gpu);
        }
        ~Conv128LayerParam() { release(); }

    public:
        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        int input_channels;
        //Weight
        float* filter;
        uin32* filter_gpu;
        int filter_width;
        int filter_height;
        //Output
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;
        int output_channels;
        float* bn;
        float* bn_gpu;
        int batch;
        int stride_height;
        int stride_width;
        bool padding;
        int pad_h;
        int pad_w;
        int pool_width;
        int pool_height;
        bool ahead_fc;
        bool save_residual;
        int* output_residual_gpu;
        bool inject_residual;
        int* input_residual_gpu; 
        int residual_channels;
        bool residual_pool;
        Conv128LayerParam* gpu;
        char name[8];
};








#endif
