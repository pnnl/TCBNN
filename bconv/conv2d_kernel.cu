/*
 * =====================================================================================
 *
 *       Filename:  conv2d_kernel.cu
 *
 *    Description:  Binarized Soft-Tensor-Core for Bit Convolution. See our paper for detail.
 *
 *        Version:  1.0
 *        Created:  01/16/2018 11:35:18 AM, Richland, WA, USA.
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Ang Li
 *        Website:  https://www.angliphd.com
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

// A faster way to obtain lane id in a warp
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
#define PREFETCH_L1(x) asm("prefetch.global.L1 [%0];"::"l"(&x));
#define PREFETCH_L1T(x) __ldg((int*)&x);

//======================================================================================    
// BSTC32bit Verion for filter conversion. It converts the raw filter from Tensorflow
// in Shape[filter_height, filter_width, in_channels, out_channels] with left the bigger
// to Shape[filter_height, filter_width, in_channels/32, out_channels] with left the bigger
//======================================================================================    
template <typename T>
__global__ void ConvertFilters_32bit(const T* __restrict__ filter, 
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
        // To shape[out_channels, filter_height, filter_width, in_channels/32]
        //if (laneid == 0) //avoid warp conflict
        filter_binarized[bx*ins*out_channels+ c*out_channels + by] = r0;
    }
}

//======================================================================================    
// BSTC64bit Verion for filter conversion. It converts the raw filter from Tensorflow
// in Shape[filter_height, filter_width, in_channels, out_channels] with left the bigger
// to Shape[filter_height, filter_width, in_channels/64, out_channels] with left the bigger
//======================================================================================    
template <typename T>
__global__ void ConvertFilters_64bit(const T* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, 
        const int out_channels, const int filter_width, const int filter_height)
{
    GET_LANEID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over out_channels
    const int ins = (in_channels >> 6);//condense C:in_channel into 64bit-unsigned-long-long
    unsigned long long* filter_binarized_64bits = (unsigned long long*)filter_binarized;

    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] with left largest
        // Fetch two int32/float32 per warp-lane
        T f0 = filter[bx*in_channels*out_channels + (c*64+laneid)*out_channels + by];
        T f1 = filter[bx*in_channels*out_channels + (c*64+32+laneid)*out_channels + by];
        // Ballot and reverse to binarize, see our paper
        unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);
        unsigned r1 = __ballot_sync(0xffffffff, f1>=0?1:0);
        // Merge into 64bit
        unsigned long long l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r1),"r"(r0)); //(low,high)
        //We tune filter to [filter_height, filter_width, in_channels/64,out_channels]
        //if (laneid == 0)
        filter_binarized_64bits[bx*ins*out_channels+ c*out_channels + by] = l0;
    }

}

//======================================================================================    
// BSTC32bit Verion for 2D Convolution.
//   @Input Feature Map: [N,H,W,C] or [batch, in_height, in_width, in_channels] as 
//    specified by tensorflow.
//   @Filter: [P,Q,C/32,K] or [filter_height, filter_width, in_channels/32, out_channels]
//   @Output Feature Map: [N,P,Q,K] or [batch, out_height, out_width, out_channels] as
//     the same as input feature map for propogation
// Algorithm Description: We convert 2D Convolution into Matrix Multiplication
// from T[N,H,W,C]*T[K,P,Q,C]=T[N,P,Q,K] into M[NPQ,CRS]xM[CRS,K], then the mapping is
// K(out_channels) => warp-lane;
// C(in_channels) => bit-op per lane
// RS(filter_width*filter_height) => flat and stored in shared-mem to share among diff K
// P(out_height) => Grid-Y
// Q(out_width) => Grid-X
// N(batch) => Grid-Z
//======================================================================================    

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
                    unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);//binarize

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

template <typename T, const int BATCH_TILE>
__global__ void Conv2d_32bit_reuse_N(const T* __restrict__ input, const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = (blockIdx.z<<5); //over K/32: out_channel/lanes
    const int ins = (in_channels >> 5);//number of steps in C: in_channels
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    extern __shared__ int Csub[];
    const int ty = threadIdx.y;
    volatile int* Ck = &Csub[ty*32*BATCH_TILE];
    int exclude = 0;
    for (int k=0; k<BATCH_TILE; k++) 
        Ck[k*32+laneid] = 0;

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
                    // Filter [filter_height, filter_width, in_channels/64,out_channels]
                    unsigned r1 = filter[(r*filter_width+s)*ins*out_channels+c*out_channels+bz+laneid];
                    for (int k=0; k<BATCH_TILE; k++)
                    {
                        //input in shape[batch, in_height, in_width, in_channels] with left largest
                        T f0 = input[(ty*BATCH_TILE+k)*in_width*in_height*in_channels
                            +(ay*in_width+ax)*in_channels+c*32+laneid];

                        unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);//binarize
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


//======================================================================================    
// BSTC64bit Verion for 2D Convolution. Similar to BSTC32bit. The difference is that
// now each warp-lane process 64 in_channels rather than 32 per op.
//   @Filter: [P,Q,C/64,K] or [filter_height, filter_width, in_channels/64, out_channels]
//======================================================================================    
template <typename T>
__global__ void Conv2d_64bit_reuse_K(const T* __restrict__ input, const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over N: batch
    const int ins = (in_channels >> 6);
    extern __shared__ int Csub[];
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //We would like to use the same format for filter param, so need type conversion here
    const unsigned long long* filter_64bits = (unsigned long long*)filter;
    int exclude = 0;

    for (int i=laneid; i<out_channels; i+=32) Csub[i] = 0;
    ////load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r;
            const int ax = ax0 + s;
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) )
            {
                for (int c=0; c<ins; c++)
                {
                    //input in shape[batch, in_height, in_width, in_channels] with left largest
                    T f0 = input[bz*in_width*in_height*in_channels
                        +(ay*in_width+ax)*in_channels+c*64+laneid];
                    T f1 = input[bz*in_width*in_height*in_channels
                        +(ay*in_width+ax)*in_channels+c*64+32+laneid];
                    unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);
                    unsigned r1 = __ballot_sync(0xffffffff, f1>=0?1:0);
                    unsigned long long l0;
                    asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r1),"r"(r0)); //low,high
                    for (int i=laneid; i<out_channels; i+=32)
                    {
                        // Filter [filter_height, filter_width, in_channels/64,out_channels]
                        unsigned long long l1 = filter_64bits[(r*filter_width+s)*ins*out_channels
                                                              +c*out_channels+i];
                        Csub[i] +=  __popcll(l0 ^ l1);//popcll for 64bit population-count
                    }
                }
            }
            else
            {
                exclude++;
            }
        }
    }
    for (int i=laneid; i<out_channels; i+=32)
        output[(bz*out_height*out_width*out_channels) + (by*out_width*out_channels) 
            + (bx*out_channels) + i] = in_channels*filter_width*filter_height 
                                        -2*Csub[i] - exclude*in_channels;


}



//==============================================================================================


template <typename T>
__global__ void Conv2d_32bit_bin(const T* __restrict__ input, const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    GET_LANEID;
    const int wid = (threadIdx.x >> 5);
    const int bx = blockIdx.x*32+wid;//over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over N: batch
    const int ins = (in_channels >> 5);//number of steps in C: in_channels
    const int ots = (out_channels >> 5);
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
                for (int c=0; c<ins; c++)
                {
                    unsigned r0 = input[bz*in_width*in_height*ins
                        +(ay*in_width+ax)*ins+c]; //coalesced access

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

    for (int k=0; k<ots; k++) //iter over K by lane
    {
        int res = in_channels*filter_width*filter_height //C*R*S
                - exclude*in_channels //eliminate padding distoration 
                - (2*Csub[k*32+laneid]);//n-2acc(a^b) for 0/1 to simulate acc(a*b) for +1/-1
        unsigned C = __brev(__ballot_sync(0xffffffff, res>=0?1:0));
        if (laneid==0)
            output[(bz*out_height*out_width*ots) //N
                + (by*out_width*ots)//P 
                + (bx*ots) + k] = C; //Q
    }
}





template <typename T>
__global__ void Conv2d_64bit_bin(const T* __restrict__ input, const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    GET_LANEID;
    const int wid = (threadIdx.x >> 5);
    const int bx = blockIdx.x*32+wid;//over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over N: batch
    const int ins = (in_channels >> 6);
    const int ots = (out_channels >> 6);

    extern __shared__ int Csub[];
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //We would like to use the same format for filter param, so need type conversion here
    const unsigned long long* filter_64bits = (unsigned long long*)filter;
    int exclude = 0;

    for (int i=laneid; i<out_channels; i+=32) Csub[i] = 0;
    ////load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r;
            const int ax = ax0 + s;
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) )
            {
                for (int c=0; c<ins; c++)
                {
                    ullong l0 = input[bz*in_width*in_height*ins
                        +(ay*in_width+ax)*ins+c]; //coalesced access

                    for (int i=laneid; i<out_channels; i+=32)
                    {
                        // Filter [filter_height, filter_width, in_channels/64,out_channels]
                        unsigned long long l1 = filter_64bits[(r*filter_width+s)*ins*out_channels
                                                              +c*out_channels+i];
                        Csub[i] +=  __popcll(l0 ^ l1);//popcll for 64bit population-count
                    }
                }
            }
            else
            {
                exclude++;
            }
        }
    }
    
    for (int k=0; k<ots; k++)
    {
        int a0 = in_channels*filter_width*filter_height
            - exclude*in_channels - (2*Csub[k*64+laneid]);
        int a1 =  in_channels*filter_width*filter_height
            - exclude*in_channels - (2*Csub[k*64+32+laneid]);
        unsigned r0 = __ballot_sync(0xffffffff, a0>=0?1:0);
        unsigned r1 = __ballot_sync(0xffffffff, a1>=0?1:0);
        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //(low,high)
        ullong C = __brevll(l0);
        if (laneid==0)
            output[(bz*(out_height)*(out_width)*ots) //N
                + (by*(out_width)*ots)//P 
                + (bx*ots) + k] //Q
                = C;
    }

}


template <typename T>
__global__ void Conv2d_32bit_bin_finegrained(const T* __restrict__ input, 
        const unsigned* __restrict__ filter,
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
    const int ots = (out_channels >> 5);
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
                    unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);//binarize

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

/*
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
*/

    for (int k=0; k<ots; k++) //iter over K by lane
    {
        int res = in_channels*filter_width*filter_height //C*R*S
                - exclude*in_channels //eliminate padding distoration 
                - (2*Csub[k*32+laneid]);//n-2acc(a^b) for 0/1 to simulate acc(a*b) for +1/-1
        unsigned C = __brev(__ballot_sync(0xffffffff, res>=0?1:0));
        if (laneid==0)
            output[(bz*out_height*out_width*ots) //N
                + (by*out_width*ots)//P 
                + (bx*ots) + k] = C; //Q
    }





}


template <typename T>
__global__ void Conv2d_64bit_bin_finegrained(const T* __restrict__ input, 
        const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{

    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = blockIdx.z; //over N: batch
    const int ins = (in_channels >> 6);
    const int ots = (out_channels >> 6);
    extern __shared__ int Csub[];
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    //We would like to use the same format for filter param, so need type conversion here
    const unsigned long long* filter_64bits = (unsigned long long*)filter;
    int exclude = 0;

    for (int i=laneid; i<out_channels; i+=32) Csub[i] = 0;
    ////load a window of data from Input
    for (int r=0; r<filter_height; r++)
    {
        for (int s=0; s<filter_width; s++)
        {
            const int ay = ay0 + r;
            const int ax = ax0 + s;
            if ( (ay>=0) && (ay<in_height) && (ax>=0) && (ax<in_width) )
            {
                for (int c=0; c<ins; c++)
                {
                    //input in shape[batch, in_height, in_width, in_channels] with left largest
                    T f0 = input[bz*in_width*in_height*in_channels
                        +(ay*in_width+ax)*in_channels+c*64+laneid];
                    T f1 = input[bz*in_width*in_height*in_channels
                        +(ay*in_width+ax)*in_channels+c*64+32+laneid];
                    unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);
                    unsigned r1 = __ballot_sync(0xffffffff, f1>=0?1:0);
                    unsigned long long l0;
                    asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r1),"r"(r0)); //low,high
                    for (int i=laneid; i<out_channels; i+=32)
                    {
                        // Filter [filter_height, filter_width, in_channels/64,out_channels]
                        unsigned long long l1 = filter_64bits[(r*filter_width+s)*ins*out_channels
                                                              +c*out_channels+i];
                        Csub[i] +=  __popcll(l0 ^ l1);//popcll for 64bit population-count
                    }
                }
            }
            else
            {
                exclude++;
            }
        }
    }
    //for (int i=laneid; i<out_channels; i+=32)
    //output[(bz*out_height*out_width*out_channels) + (by*out_width*out_channels) 
    //+ (bx*out_channels) + i] = in_channels*filter_width*filter_height 
    //-2*Csub[i] - exclude*in_channels;

    for (int k=0; k<ots; k++)
    {
        int a0 = in_channels*filter_width*filter_height
            - exclude*in_channels - (2*Csub[k*64+laneid]);
        int a1 =  in_channels*filter_width*filter_height
            - exclude*in_channels - (2*Csub[k*64+32+laneid]);
        unsigned r0 = __ballot_sync(0xffffffff, a0>=0?1:0);
        unsigned r1 = __ballot_sync(0xffffffff, a1>=0?1:0);
        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //(low,high)
        ullong C = __brevll(l0);
        if (laneid==0)
            output[(bz*(out_height)*(out_width)*ots) //N
                + (by*(out_width)*ots)//P 
                + (bx*ots) + k] //Q
                = C;
    }



}



//==============================================================================================














template <typename T, const int BATCH_TILE>
__global__ void Conv2d_64bit_reuse_N(const T* __restrict__ input, const unsigned* __restrict__ filter,
        T* output, const int in_channels, const int out_channels, const int in_width, 
        const int in_height, const int filter_width, const int filter_height, 
        const int batch, const int stride_vertical, const int stride_horizontal,
        const int out_width, const int out_height, const int pad_h, const int pad_w)
{
    GET_LANEID;
    const int bx = blockIdx.x; //over Q: out_width 
    const int by = blockIdx.y; //over P: out_height
    const int bz = (blockIdx.z<<5); //over K/32: out_channel/lanes
    const int ins = (in_channels >> 6);
    const int ax0 = bx*stride_horizontal-pad_w;
    const int ay0 = by*stride_vertical-pad_h;
    const unsigned long long* filter_64bits = (unsigned long long*)filter;
    extern __shared__ int Csub[];
    const int ty = threadIdx.y;
    volatile int* Ck = &Csub[ty*32*BATCH_TILE];
    int exclude = 0;

    //#pragma unroll
    for (int k=0; k<BATCH_TILE; k++) 
        Ck[k*32+laneid] = 0;

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
                    // Filter [filter_height, filter_width, in_channels/64,out_channels]
                    unsigned long long l1 = filter_64bits[(r*filter_width+s)*ins*out_channels
                        +c*out_channels+bz+laneid];
                    for (int k=0; k<BATCH_TILE; k++)
                    {
                        //input in shape[batch, in_height, in_width, in_channels] with left largest
                        T f0 = input[(ty*BATCH_TILE+k)*in_width*in_height*in_channels
                            +(ay*in_width+ax)*in_channels+c*64+laneid];
                        T f1 = input[(ty*BATCH_TILE+k)*in_width*in_height*in_channels
                            +(ay*in_width+ax)*in_channels+c*64+32+laneid];
                        unsigned r0 = __ballot_sync(0xffffffff, f0>=0?1:0);
                        unsigned r1 = __ballot_sync(0xffffffff, f1>=0?1:0);
                        unsigned long long l0;
                        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r1),"r"(r0)); //low,high
                        Ck[k*32+laneid] +=  __popcll(l0 ^ l1);//popcll for 64bit population-count
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

//======================================================================================    
// The real API for tensorflow wraper to call
//======================================================================================    
template<typename T>
double Conv2dFunctor(
        const T* input, const T* filter, T* output, 
        unsigned* filter_binarized, const int batch, const int in_width, 
        const int in_height, const int in_channels, 
        const int out_channels, const int filter_width, const int filter_height,
        const int stride_vertical, const int stride_horizontal, 
        const int out_width, const int out_height, const int pad_h, const int pad_w,
        bool use_64bit) 
{
#ifdef DEBUG        
    printf("\n\nExecuting Conv2dFunctor.\n");
    //printf("\nKernel Setting: Grid(%d,%d,%d), Block(%d).\n",out_width, out_height, batch, 32);
#endif        
    unsigned* filter_binarized_unsigned = (unsigned*)filter_binarized;
    //All this work (BSTC) follows the warp-consolidation model, see our ics-18 paper
    dim3 blockDim(32);
    //by iter over K, bx iter over P*Q
    dim3 convertDim(filter_height*filter_width, out_channels);
    const unsigned BATCH_TILE = 16;//the optimal depends on batch, GPU, and 32/64bit

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (use_64bit) //C:in_channels can be divided by 64, call 64-bit version
    {
        //bx iter over Q, by iter over P, bz iter over N
        dim3 gridDim(out_width, out_height, batch);
        int smem_size = out_channels * sizeof(int);

        cudaEventRecord(start);

        for (int i=0; i<TEST_TIMES; i++)
        {
            ConvertFilters_64bit<T><<<convertDim, blockDim>>>(filter, filter_binarized_unsigned, 
                    in_channels, out_channels, filter_width, filter_height);
            //==================================================== 
            Conv2d_64bit_reuse_K<T><<<gridDim, blockDim, smem_size>>>(input, filter_binarized_unsigned,
                    output, in_channels, out_channels, in_width, in_height, filter_width, 
                    filter_height, batch, stride_vertical, stride_horizontal, 
                    out_width, out_height, pad_h, pad_w);
            //==================================================== 
        }
        cudaEventRecord(stop);
    }
    else //BSTC-32
    {
        //bx iter over Q, by iter over P, bz iter over N
        dim3 gridDim(out_width, out_height, batch);
        int smem_size = out_channels * sizeof(int);

        cudaEventRecord(start);
        for (int i=0; i<TEST_TIMES; i++)
        {
            ConvertFilters_32bit<T><<<convertDim, blockDim>>>(filter, filter_binarized_unsigned, 
                    in_channels, out_channels, filter_width, filter_height);
            //==================================================== 
            Conv2d_32bit_reuse_K<T><<<gridDim, blockDim, smem_size>>>(input, filter_binarized_unsigned,
                    output, in_channels, out_channels, in_width, in_height, filter_width, 
                    filter_height, batch, stride_vertical, stride_horizontal, 
                    out_width, out_height, pad_h, pad_w);
            //==================================================== 
        }
        cudaEventRecord(stop);
    }

#ifdef DEBUG        
    {
        cudaThreadSynchronize();
        cudaError err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,"cudaCheckError() failed at Conv2dFunctor: %s\n", 
                    cudaGetErrorString(err));
            exit(-1);
        }
        printf("\n\nFinished Conv2dFunctor.\n\n");
    }
#endif        

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);

    return (milliseconds*1e3)/TEST_TIMES;
}



template<typename T>
double Conv2dFunctorBin(
        const T* input, const float* filter, T* output, 
        unsigned* filter_binarized, const int batch, const int in_width, 
        const int in_height, const int in_channels, 
        const int out_channels, const int filter_width, const int filter_height,
        const int stride_vertical, const int stride_horizontal, 
        const int out_width, const int out_height, const int pad_h, const int pad_w,
        bool use_64bit) 
{
#ifdef DEBUG        
    printf("\n\nExecuting Conv2dFunctor.\n");
    //printf("\nKernel Setting: Grid(%d,%d,%d), Block(%d).\n",out_width, out_height, batch, 32);
#endif        
    unsigned* filter_binarized_unsigned = (unsigned*)filter_binarized;
    //by iter over K, bx iter over P*Q
    const unsigned BATCH_TILE = 16;//the optimal depends on batch, GPU, and 32/64bit
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (use_64bit) //C:in_channels can be divided by 64, call 64-bit version
    {
        //bx iter over Q, by iter over P, bz iter over N
        dim3 gridDim(out_width/32, out_height, batch);
        int smem_size = out_channels * sizeof(int)*32;
        cudaEventRecord(start);
        for (int i=0; i<TEST_TIMES; i++)
        {
            ConvertFilters_64bit<float><<<dim3(filter_height*filter_width, out_channels), 32>>>(filter, 
                    filter_binarized_unsigned, 
                    in_channels, out_channels, filter_width, filter_height);
            //==================================================== 
            Conv2d_64bit_bin<T><<<gridDim, 1024, smem_size>>>(input, filter_binarized_unsigned,
                    output, in_channels, out_channels, in_width, in_height, filter_width, 
                    filter_height, batch, stride_vertical, stride_horizontal, 
                    out_width, out_height, pad_h, pad_w);
            //==================================================== 
        }
        cudaEventRecord(stop);
    }
    else //BSTC-32
    {
        //bx iter over Q, by iter over P, bz iter over N
        dim3 gridDim(out_width/32, out_height, batch);
        int smem_size = out_channels * sizeof(int)*32;
        cudaEventRecord(start);
        for (int i=0; i<TEST_TIMES; i++)
        {
            ConvertFilters_32bit<float><<<dim3(filter_height*filter_width, out_channels), 32>>>(filter, 
                    filter_binarized_unsigned, 
                    in_channels, out_channels, filter_width, filter_height);
            //==================================================== 
            Conv2d_32bit_bin<T><<<gridDim, 1024, smem_size>>>(input, filter_binarized_unsigned,
                    output, in_channels, out_channels, in_width, in_height, filter_width, 
                    filter_height, batch, stride_vertical, stride_horizontal, 
                    out_width, out_height, pad_h, pad_w);
            //==================================================== 
        }
        cudaEventRecord(stop);
    }

#ifdef DEBUG        
    {
        cudaThreadSynchronize();
        cudaError err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,"cudaCheckError() failed at Conv2dFunctor: %s\n", 
                    cudaGetErrorString(err));
            exit(-1);
        }
        printf("\n\nFinished Conv2dFunctor.\n\n");
    }
#endif        
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    return (milliseconds*1e3)/TEST_TIMES;
}



template<typename T>
double Conv2dFunctorBinFineGrined(
        const T* input, const float* filter, T* output, 
        unsigned* filter_binarized, const int batch, const int in_width, 
        const int in_height, const int in_channels, 
        const int out_channels, const int filter_width, const int filter_height,
        const int stride_vertical, const int stride_horizontal, 
        const int out_width, const int out_height, const int pad_h, const int pad_w,
        bool use_64bit) 
{
#ifdef DEBUG        
    printf("\n\nExecuting Conv2dFunctor.\n");
    //printf("\nKernel Setting: Grid(%d,%d,%d), Block(%d).\n",out_width, out_height, batch, 32);
#endif        
    unsigned* filter_binarized_unsigned = (unsigned*)filter_binarized;
    //by iter over K, bx iter over P*Q
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 blockDim(32);
    dim3 convertDim(filter_height*filter_width, out_channels);

    if (use_64bit) //C:in_channels can be divided by 64, call 64-bit version
    {
        //bx iter over Q, by iter over P, bz iter over N
        dim3 gridDim(out_width, out_height, batch);
        int smem_size = out_channels * sizeof(int);

        cudaEventRecord(start);
        for (int i=0; i<TEST_TIMES; i++)
        {
            ConvertFilters_64bit<float><<<dim3(filter_height*filter_width, out_channels), 32>>>(filter, 
                    filter_binarized_unsigned, 
                    in_channels, out_channels, filter_width, filter_height);
            //==================================================== 
            Conv2d_64bit_bin_finegrained<T><<<gridDim, blockDim, smem_size>>>(input, filter_binarized_unsigned,
                    output, in_channels, out_channels, in_width, in_height, filter_width, 
                    filter_height, batch, stride_vertical, stride_horizontal, 
                    out_width, out_height, pad_h, pad_w);
            //==================================================== 
        }
        cudaEventRecord(stop);
    }
    else //BSTC-32
    {
        //bx iter over Q, by iter over P, bz iter over N
        dim3 gridDim(out_width, out_height, batch);
        int smem_size = out_channels * sizeof(int);
        cudaEventRecord(start);
        for (int i=0; i<TEST_TIMES; i++)
        {
            ConvertFilters_32bit<float><<<dim3(filter_height*filter_width, out_channels), 32>>>(
                    filter, 
                    filter_binarized_unsigned, 
                    in_channels, out_channels, filter_width, filter_height);
            //==================================================== 
            Conv2d_32bit_bin_finegrained<T><<<gridDim, blockDim, smem_size>>>(
                    input, filter_binarized_unsigned,
                    output, in_channels, out_channels, in_width, in_height, filter_width, 
                    filter_height, batch, stride_vertical, stride_horizontal, 
                    out_width, out_height, pad_h, pad_w);
            //==================================================== 
        }
        cudaEventRecord(stop);
    }

#ifdef DEBUG        
    {
        cudaThreadSynchronize();
        cudaError err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,"cudaCheckError() failed at Conv2dFunctor: %s\n", 
                    cudaGetErrorString(err));
            exit(-1);
        }
        printf("\n\nFinished Conv2dFunctor.\n\n");
    }
#endif        
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    return (milliseconds*1e3)/TEST_TIMES;
}







//template void Conv2dFunctor<float>(const float* input, 
        //const float* filter, 
        //float* output, int* filter_binarized, const int batch, const int in_width, 
        //const int in_height, const int in_channels, 
        //const int out_channels, const int filter_width, const int filter_height,
        //const int stride_vertical, const int stride_horizontal, 
        //const int out_width, const int out_height, const int pad_h, const int pad_w); 

//template void Conv2dFunctor<int>(const int* input, 
        //const int* filter, 
        //int* output, int* filter_binarized, const int batch, const int in_width, 
        //const int in_height, const int in_channels, 
        //const int out_channels, const int filter_width, const int filter_height,
        //const int stride_vertical, const int stride_horizontal, 
        //const int out_width, const int out_height, const int pad_h, const int pad_w);

