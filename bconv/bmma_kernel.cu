/*
 * =====================================================================================
 *
 *       Filename:  bmma_kernel.cu
 *
 *    Description:  Accelerate BConv via TensorCores in Turing GPU
 *
 *        Version:  1.0
 *        Created:  11/04/2019 10:48:58 PM, Richland, WA, USA.
 *       Revision:  none
 *       Compiler:  nvcc -arch=sm_75
 *
 *         Author:  Ang Li
 *        Website:  https://www.angliphd.com
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>

// A faster way to obtain lane id in a warp
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
using namespace nvcuda;
typedef unsigned char uchar;


//======================================================================================    
// TensorCore BMMA for 2D Convolution.
//   @Input Feature Map: [H,W,N,C/128] or [in_height, in_width, batch, in_channels/128] as 
//    specified by tensorflow.
//   @Filter: [K,K,O,C/128] or [filter_height, filter_width, out_channels, in_channels/128]
//   @Output Feature Map: [P,Q,N,O] or [out_height, out_width, batch, out_channels] as
//     the same as input feature map for propogation
// Algorithm Description: We exploit internal Matrix Multiplication inside Conv2d
// from T[H,W,N,C]*T[K,K,O,C]=T[P,Q,N,O] to exploit M[N,C]*M[O,C]
// per warp execute M[N=8,C=128]*M[O=8,C=128]
// P(out_height) => Grid-Y
// Q(out_width) => Grid-X
// N(batch/8 or O/8) => Grid-Z
//======================================================================================    


template <typename T>
__global__ void BTC_bin_filter(const T* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    GET_LANEID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over out_channels
    const int ins = (in_channels >> 5);//condense C:in_channel into 32bit-unsigned

    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        T f0 = filter[bx*in_channels*out_channels + (c*32+laneid)*out_channels + by];
        unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);
        // To shape[out_channels, filter_height, filter_width, out_channels, in_channels/32]
        //filter: [K,K,O,C]
        if (laneid == 0) //avoid warp conflict
        {      
            filter_binarized[bx*ins*out_channels+ by*ins + c] = r0;
        }
    }
}

template <typename T>
__global__ void BTC_bin_input(const T* __restrict__ input, unsigned* input_binarized,
    const int in_channels, const int batch, const int in_width, const int in_height)
{
    GET_LANEID;
    //input: [H,W,N,C]
    const int bx = blockIdx.x;//iter over in_width
    const int by = blockIdx.y;//iter over in_height
    const int bz = blockIdx.z;///iter over batch
    const int ins = (in_channels >> 5);//condense C:in_channel into 32bit-unsigned

    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
        // From shape[batch, in_height, in_width, in_channels]
        T f0 = input[bz*in_height*in_width*in_channels + by*in_width*in_channels
            + bx*in_channels + c*32 + laneid];
        unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);
        // To shape[input_height, input_width, batch, in_channels/32]
        //input: [H,W,N,C]
        if (laneid == 0) //avoid warp conflict
        {
            input_binarized[(by*in_width+bx)*batch*ins + bz*ins + c] = r0;
            //printf("-%u-",r0);
        }
    }
}

//workload balance
__global__ void BTC_Conv2d_balance(const unsigned* __restrict__ input, 
        const unsigned* __restrict__ filter,
        int* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    using namespace nvcuda::wmma::experimental;
    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over O: out_channel/8 * batch/8
    const int ins = (in_channels >> 7);//number of steps in C: in_channels
    const int bn = bz / (out_channels >> 3);
    const int bo = bz % (out_channels >> 3);

    //coord (ax,ay) in Input from bx,by in Output
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //track the number of filter entries that are masked off
    int exclude = 0;
    wmma::fill_fragment(c_frag, 0);

    //load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r; //y-coord in Input
            const int ax = ax0 + s; //x-coord in Input
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) ) //within Input frame
            {
                for (int c=0; c<ins; c++)//per 128-bit in C:in_channels
                {
                    //input: [H,W,N,C]
                    load_matrix_sync(a_frag, &input[(ay*in_width+ax)*batch*ins*4+ 
                            bn*8*ins*4 + c*4], in_channels);
                    //filter: [K,K,O,C]
                    load_matrix_sync(b_frag, &filter[(r*filter_width+s)*out_channels*ins*4+
                            bo*8*ins*4 + c*4], in_channels);
                    bmma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }
            else //not in frame
            {
                exclude++; //accumulate
            }
        }
    }
    for (int i=0; i<c_frag.num_elements; i++)
    {
        c_frag.x[i] = in_channels*filter_height*filter_width // C*R*S
            - exclude*in_channels // eliminate distoration
            - (2*c_frag.x[i]); 
    }
    //output: [P,Q,N,O] => [by,bx,bn,bo]
    store_matrix_sync(&output[(by*out_width+bx)*batch*out_channels + bn*8*out_channels + bo*8], 
            c_frag, out_channels, wmma::mem_row_major);

    //output should be N,P,Q,O
}


__global__ void BTC_Conv2d_balance_bin(const unsigned* __restrict__ input, 
        const unsigned* __restrict__ filter,
        int* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    using namespace nvcuda::wmma::experimental;
    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over O: out_channel/8 * batch/8
    const int ins = (in_channels >> 7);//number of steps in C: in_channels
    const int ots = (out_channels >> 7);//number of steps in O: out_channels
    const int bn = bz / (out_channels >> 3);
    const int bo = bz % (out_channels >> 3);

    //coord (ax,ay) in Input from bx,by in Output
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //track the number of filter entries that are masked off
    int exclude = 0;
    wmma::fill_fragment(c_frag, 0);

    //load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r; //y-coord in Input
            const int ax = ax0 + s; //x-coord in Input
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) ) //within Input frame
            {
                for (int c=0; c<ins; c++)//per 128-bit in C:in_channels
                {
                    //input: [H,W,N,C]
                    load_matrix_sync(a_frag, &input[(ay*in_width+ax)*batch*ins*4+ 
                            bn*8*ins*4 + c*4], in_channels);
                    //filter: [K,K,O,C]
                    load_matrix_sync(b_frag, &filter[(r*filter_width+s)*out_channels*ins*4+
                            bo*8*ins*4 + c*4], in_channels);
                    bmma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }
            else //not in frame
            {
                exclude++; //accumulate
            }
        }
    }
    __shared__ int Cs[64];
    store_matrix_sync(Cs, c_frag, 8, wmma::mem_row_major);
    union{ unsigned data; uchar elements[4];} p0, p1;
    int tmp = in_channels*filter_height*filter_width // C*R*S
            - exclude*in_channels; // eliminate distoration
    p0.data =  __ballot_sync(0xffffffff, (tmp-2*Cs[laneid]>=0)?1:0);
    p1.data =  __ballot_sync(0xffffffff, (tmp-2*Cs[32+laneid]>=0)?1:0);

    uchar* output_uc = (uchar*)(&output[(by*out_width+bx)*batch*ots]);
    if (laneid < 4)
    {
        output_uc[(bn*8+laneid)*(out_channels/8)+bo] = p0.elements[laneid]; 
        output_uc[(bn*8+4+laneid)*(out_channels/8)+bo] = p1.elements[laneid]; 
    }
    ////output: [P,Q,N,O] => [by,bx,bn,bo]
    //store_matrix_sync(&output[(by*out_width+bx)*batch*out_channels + bn*8*out_channels + bo*8], 
    //c_frag, out_channels, wmma::mem_row_major);

    //output should be N,P,Q,O
}





__global__ void BTC_trans_output(const int* __restrict__ output_buf, int* output,
    const int out_channels, const int batch, const int out_width, const int out_height)
{
    //convert from [P,Q,N,O] to [N,P,Q,O]
    GET_LANEID;
    const int bx = blockIdx.x;//iter over out_width
    const int by = blockIdx.y;//iter over out_height
    const int bz = blockIdx.z;///iter over batch

    for (int c=laneid; c<out_channels; c+=32) //iter over C:in_channels
    {
        int f0 = output_buf[by*out_width*out_channels*batch + bx*out_channels*batch
            + bz*out_channels+c];
        output[bz*out_height*out_width*out_channels + by*out_width*out_channels
            + bx*out_channels+c] = f0;
    }
}





template <typename T>
__global__ void BTC_bin_filter_128(const T* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    GET_LANEID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over out_channels
    const int ins = (in_channels >> 5);//condense C:in_channel into 32bit-unsigned

    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        T f0 = filter[bx*in_channels*out_channels + (c*32+laneid)*out_channels + by];
        unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);
        // To shape[out_channels, filter_height, filter_width, out_channels, in_channels/32]
        //filter: [K,K,O,C]
        if (laneid == 0) //avoid warp conflict
        {      
            //filter_binarized[bx*ins*out_channels  
            //+ ((by/8)*(ins/4) + c/4)*32 + (by%8)*4 + (c%4)] = r0;
            filter_binarized[bx*ins*out_channels  
                + (((by>>3)*(ins>>2) + (c>>2))<<5) + ((by&0x7)<<2) + (c&0x3)] = r0;
        }
    }
}

template <typename T>
__global__ void BTC_bin_input_128(const T* __restrict__ input, unsigned* input_binarized,
    const int in_channels, const int batch, const int in_width, const int in_height)
{
    GET_LANEID;
    //input: [H,W,N,C]
    const int bx = blockIdx.x;//iter over in_width
    const int by = blockIdx.y;//iter over in_height
    const int bz = blockIdx.z;///iter over batch
    const int ins = (in_channels >> 5);//condense C:in_channel into 32bit-unsigned

    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
        // From shape[batch, in_height, in_width, in_channels]
        T f0 = input[bz*in_height*in_width*in_channels + by*in_width*in_channels
            + bx*in_channels + c*32 + laneid];
        unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);
        // To shape[input_height, input_width, batch, in_channels/32]
        //input: [H,W,N,C]
        if (laneid == 0) //avoid warp conflict
        {
            //input_binarized[(by*in_width+bx)*batch*ins 
            //+ ((bz/8)*(ins/4) + (c/4))*32 + (bz%8)*4 + (c%4)] = r0;
            input_binarized[(by*in_width+bx)*batch*ins 
                + (((bz>>3)*(ins>>2) + (c>>2))<<5) + ((bz&0x7)<<2) + (c&0x3)] = r0;
        }
    }
}

//workload balance
__global__ void BTC_Conv2d_balance_128(const unsigned* __restrict__ input, 
        const unsigned* __restrict__ filter,
        int* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    using namespace nvcuda::wmma::experimental;
    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over O: out_channel/8 * batch/8
    const int ins = (in_channels >> 7);//number of steps in C: in_channels
    const int bn = bz / (out_channels >> 3);
    const int bo = bz % (out_channels >> 3);

    //coord (ax,ay) in Input from bx,by in Output
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //track the number of filter entries that are masked off
    int exclude = 0;
    wmma::fill_fragment(c_frag, 0);

    //load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r; //y-coord in Input
            const int ax = ax0 + s; //x-coord in Input
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) ) //within Input frame
            {
                for (int c=0; c<ins; c++)//per 128-bit in C:in_channels
                {
                    //input: [H,W,N,C]
                    load_matrix_sync(a_frag, &input[(ay*in_width+ax)*batch*ins*4+ 
                            bn*8*ins*4 + c*4*8], 128);
                    //filter: [K,K,O,C]
                    load_matrix_sync(b_frag, &filter[(r*filter_width+s)*out_channels*ins*4+
                            bo*8*ins*4 + c*4*8], 128);

                    bmma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }
            else //not in frame
            {
                exclude++; //accumulate
            }
        }
    }
    for (int i=0; i<c_frag.num_elements; i++)
    {
        c_frag.x[i] = in_channels*filter_height*filter_width // C*R*S
            - exclude*in_channels // eliminate distoration
            - (2*c_frag.x[i]); 
    }
    //output: [P,Q,N,O] => [by,bx,bn,bo]
    store_matrix_sync(&output[(by*out_width+bx)*batch*out_channels + bn*8*out_channels + bo*8], 
            c_frag, out_channels, wmma::mem_row_major);

    //output should be N,P,Q,O

}

//workload balance
__global__ void BTC_Conv2d_balance_128_bin(const unsigned* __restrict__ input, 
        const unsigned* __restrict__ filter,
        int* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    using namespace nvcuda::wmma::experimental;
    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over O: out_channel/8 * batch/8
    const int ins = (in_channels >> 7);//number of steps in C: in_channels
    const int ots = (out_channels >> 7);//number of steps in O: out_channels
    const int bn = bz / (out_channels >> 3);
    const int bo = bz % (out_channels >> 3);

    //coord (ax,ay) in Input from bx,by in Output
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //track the number of filter entries that are masked off
    int exclude = 0;
    wmma::fill_fragment(c_frag, 0);

    //load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r; //y-coord in Input
            const int ax = ax0 + s; //x-coord in Input
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) ) //within Input frame
            {
                for (int c=0; c<ins; c++)//per 128-bit in C:in_channels
                {
                    //input: [H,W,N,C]
                    load_matrix_sync(a_frag, &input[(ay*in_width+ax)*batch*ins*4+ 
                            bn*8*ins*4 + c*4*8], 128);
                    //filter: [K,K,O,C]
                    load_matrix_sync(b_frag, &filter[(r*filter_width+s)*out_channels*ins*4+
                            bo*8*ins*4 + c*4*8], 128);

                    bmma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }
            else //not in frame
            {
                exclude++; //accumulate
            }
        }
    }

    __shared__ int Cs[64];
    store_matrix_sync(Cs, c_frag, 8, wmma::mem_row_major);

    union{ unsigned data; uchar elements[4];} p0, p1;
    int tmp = in_channels*filter_height*filter_width // C*R*S
            - exclude*in_channels; // eliminate distoration
    p0.data =  __ballot_sync(0xffffffff, (tmp-2*Cs[laneid]>=0)?1:0);
    p1.data =  __ballot_sync(0xffffffff, (tmp-2*Cs[32+laneid]>=0)?1:0);

    uchar* output_uc = (uchar*)(&output[(by*out_width+bx)*batch*ots]);
    if (laneid < 4)
    {
        //output_uc[(bn*8+laneid)*(out_channels/8)+bo] = p0.elements[laneid]; 
        //output_uc[(bn*8+4+laneid)*(out_channels/8)+bo] = p1.elements[laneid]; 
        output_uc[(bo*batch/8+(bn/16))*8*128/32+laneid*128/32+bn%16] = p0.elements[laneid]; 
        output_uc[(bo*batch/8+(bn/16))*8*128/32+(32+laneid)*128/32+bn%16] = p1.elements[laneid]; 
    }

    ////output: [P,Q,N,O] => [by,bx,bn,bo]
    //store_matrix_sync(&output[(by*out_width+bx)*batch*out_channels + bn*8*out_channels + bo*8], 
    //c_frag, out_channels, wmma::mem_row_major);
    //output should be N,P,Q,O
}









/*



template <typename T>
__global__ void BTC_bin_filter_new(const T* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over out_channels/8
    const int ins = (in_channels >> 7);//C:in_channel into 32bit-unsigned
    
    const int laneid = threadIdx.x
    const int ty = threadIdx.y;//iter over 4 of C-block
    const int tz = threadIdx.z;//iter over 8 of out_channel

    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
    ////// Start from here!!!!!!!! Nov-30
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        T f0 = filter[bx*in_channels*out_channels + (c*128+ty*4+laneid)*out_channels + by*8+tz];
        unsigned r0 = __ballot(f0>=0);
        // To shape[out_channels, filter_height, filter_width, out_channels, in_channels/32]
        //filter: [K,K,O,C]
        if (laneid == 0) //avoid warp conflict
            filter_binarized[bx*ins*out_channels+ by*ins + c] = r0;
    }
}

template <typename T>
__global__ void BTC_bin_input_new(const T* __restrict__ input, unsigned* input_binarized,
    const int in_channels, const int batch, const int in_width, const int in_height)
{
    //input: [H,W,N,C]
    const int bx = blockIdx.x;//iter over in_width
    const int by = blockIdx.y;//iter over in_height
    const int bz = blockIdx.z///iter over batch
    const int ins = (in_channels >> 5);//condense C:in_channel into 32bit-unsigned

    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
        // From shape[batch, in_height, in_width, in_channels]
        T f0 = input[bz*in_height*in_width*in_channels + by*in_width*in_channels
            + bx*in_channels + c*32 + laneid];
        unsigned r0 = __ballot(f0>=0);
        // To shape[input_height, input_width, batch, in_channels/32]
        //input: [H,W,N,C]
        if (laneid == 0) //avoid warp conflict
            input_binarized[(by*in_width+bx)*batch*ins + bz*ins + c] = r0;
    }
}











template <typename T, const int BATCH_TILE>
__global__ void BTC_Conv2d_reuse_N(const T* __restrict__ input, 
        const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over O: out_channel by step=8
    const int ins = (in_channels >> 7);//number of steps in C: in_channels/128
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    //extern __shared__ int Csub[];
    //const int ty = threadIdx.y;
    //volatile int* Ck = &Csub[ty*64*BATCH_TILE];
    //for (int k=0; k<BATCH_TILE; k++) 
    //Ck[k*32+laneid] = 0;

    int exclude = 0;
    for (int r=0; r<filter_height; r++)
    {
        const int ay = ay0 + r;
        for (int s=0; s<filter_width; s++)
        {
            const int ax = ax0 + s;
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) )
            {
                for (int c=0; c<ins; c++)
                {
                    load_matrix_sync(a_frag, , 128);
                    load_matrix_sync(b_frag, &Bs[wy*8], 128);
                    bmma_sync(c_frag, a_frag, b_frag, c_frag);



                    // Filter [filter_height, filter_width, in_channels/64,out_channels]
                    unsigned r1 = filter[(r*filter_width+s)*ins*out_channels+c*out_channels+bz+laneid];
                    for (int k=0; k<BATCH_TILE; k++)
                    {
                        //input in shape[batch, in_height, in_width, in_channels] with left largest
                        T f0 = input[(ty*BATCH_TILE+k)*in_width*in_height*in_channels
                            +(ay*in_width+ax)*in_channels+c*32+laneid];

                        unsigned r0 = __ballot(f0>=0);//binarize
                        Ck[k*32+laneid] +=  __popc(r0 ^ r1);
                    }
                }
            }
            else
            {
                exclude++;
            }
        }
    }
    for (int k=0; k<BATCH_TILE; k++)
    {
        output[((ty*BATCH_TILE+k)*out_height*out_width*out_channels) + (by*out_width*out_channels) 
            + (bx*out_channels) + bz+laneid] = in_channels*filter_width*filter_height 
            -2*Ck[k*32+laneid] - exclude*in_channels;
    }
}





//Reuse batch
template <typename T>
__global__ void BTC_Conv2d_reuse_K(const T* __restrict__ input, 
        const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over O: out_channel
    const int ins = (in_channels >> 7);//number of steps in C: in_channels

    //coord (ax,ay) in Input from bx,by in Output
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //track the number of filter entries that are masked off
    int exclude = 0;

    //load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r; //y-coord in Input
            const int ax = ax0 + s; //x-coord in Input
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) ) //within Input frame
            {
                for (int c=0; c<ins; c++)//per 128-bit in C:in_channels
                {
                    //[H,W,N,C]
                    T f0 = input[bz*in_width*in_height*in_channels
                        +(ay*in_width+ax)*in_channels+c*32+laneid];
                    unsigned r0 = __ballot(f0>=0);//binarize

                    for (int i=laneid; i<out_channels; i+=32)
                    {
                        unsigned r1 = filter[(r*filter_width+s)*ins*out_channels+c*out_channels+i];
                        Csub[i] += __popc(r0 ^ r1);
                    }
                }
            }
            else //not in frame
            {
                exclude++; //accumulate
            }
        }
    }

    for (int i=laneid; i<out_channels; i+=32) //iter over K by lane
    {
        //output shape [N,P,Q,K] left largest, corresponding to [bz,by,bx,i]
        output[(bz*out_height*out_width*out_channels) //N
            + (by*out_width*out_channels)//P 
            + (bx*out_channels) + i] //Q
            = in_channels*filter_width*filter_height //C*R*S
              - exclude*in_channels //eliminate padding distoration 
              - (2*Csub[i]);//n-2acc(a^b) for 0/1 to simulate acc(a*b) for +1/-1
    }

}








template <typename T>
__global__ void Conv2d_32bit_reuse_K(const T* __restrict__ input, const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over N: batch
    const int ins = (in_channels >> 5);//number of steps in C: in_channels
    extern __shared__ int Csub[];
    //coord (ax,ay) in Input from bx,by in Output
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //track the number of filter entries that are masked off
    int exclude = 0;

    for (int i=laneid; i<out_channels; i+=32) Csub[i] = 0;

    //load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r; //y-coord in Input
            const int ax = ax0 + s; //x-coord in Input
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) ) //within Input frame
            {
                for (int c=0; c<ins; c++)//per 32-bit in C:in_channels
                {
                    //Input[batch, in_height, in_width, in_channels] with left largest
                    T f0 = input[bz*in_width*in_height*in_channels
                        +(ay*in_width+ax)*in_channels+c*32+laneid];
                    unsigned r0 = __ballot(f0>=0);//binarize

                    for (int i=laneid; i<out_channels; i+=32)
                    {
                        unsigned r1 = filter[(r*filter_width+s)*ins*out_channels+c*out_channels+i];
                        Csub[i] += __popc(r0 ^ r1);
                    }
                }
            }
            else //not in frame
            {
                exclude++; //accumulate
            }
        }
    }

    for (int i=laneid; i<out_channels; i+=32) //iter over K by lane
    {
        //output shape [N,P,Q,K] left largest, corresponding to [bz,by,bx,i]
        output[(bz*out_height*out_width*out_channels) //N
            + (by*out_width*out_channels)//P 
            + (bx*out_channels) + i] //Q
            = in_channels*filter_width*filter_height //C*R*S
              - exclude*in_channels //eliminate padding distoration 
              - (2*Csub[i]);//n-2acc(a^b) for 0/1 to simulate acc(a*b) for +1/-1
    }

}

*/
