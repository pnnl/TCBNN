/*
 * =====================================================================================
 *
 *       Filename:  matmul_kernel.cu
 *
 *    Description:  Binarized Soft-Tensor-Core for Bit Matmul. See our paper for detail.
 *
 *        Version:  1.0
 *        Created:  01/17/2018 10:24:21 AM, Richland, WA, USA.
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

typedef unsigned long long ullong;
// A faster way to obtain lane id in a warp
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));

//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store64(const void* addr, T a, T b)
{
    *((float2*)addr) = make_float2(*(float*)(&a),*(float*)(&b));
}
//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store128(const void* addr, T a, T b, T c, T d)
{
    *((float4*)addr) = make_float4(*(float*)(&a),*(float*)(&b),*(float*)(&c),*(float*)(&d));
}

//======================================================================================    
// From row-major normal array to column-major 32-bit-array. This func assumes A_width
// can be divided by 32. So no padding is required.
//======================================================================================  
template <typename T>
__global__ void ToBit32Col(const T* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; 
    const unsigned bx = blockIdx.x;
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++) 
    {
        T f0 = A[(bx*32+i)*A_width+by*32 +laneid];
        unsigned r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
        if (laneid == i) Bval = r0;
    }
    B[by*A_height+bx*32+laneid] = Bval;
}

//======================================================================================    
// From row-major normal array to column-major 32-bit-array. This func is general which
// allows padding when A_width cannot divide 32.
//======================================================================================  
template <typename T>
__global__ void ToBit32ColUd(const T* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; 
    const unsigned bx = blockIdx.x;
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++) 
    {
        T f0 = ( (by*32+laneid<A_width) && (bx*32+i<A_height) )?
            A[(bx*32+i)*A_width+by*32 +laneid]:(T)-1;
        unsigned r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
        if (laneid == i) Bval = r0;
    }
    if (laneid < A_height*A_width)
        B[by*gridDim.x*32+bx*32+laneid] = Bval;
}

//======================================================================================    
// From row-major normal array to row-major 32-bit-array. This func assumes A_width
// can be divided by 32. So no padding is required.
//======================================================================================  
template <typename T>
__global__ void ToBit32Row(const T* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    unsigned Bval=0;
#pragma unroll
    for (int i=0; i<32; i++) 
    {
        T f0 = A[(bx*32+i)*A_width+by*32+laneid];
        Bval = (Bval<<1) + (f0>=0?1:0);
    }
    B[bx*A_width+by*32+laneid] = Bval;
}

//======================================================================================    
// From row-major normal array to row-major 32-bit-array. This func is general which
// allows padding when A_width cannot divide 32.
//======================================================================================  
template <typename T>
__global__ void ToBit32RowUd(const T* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    unsigned Bval=0;
#pragma unroll
    for (int i=0; i<32; i++) 
    {
        T f0 = ( (by*32+laneid<A_width) && (bx*32+i<A_height) )?  A[(bx*32+i)*A_width+by*32+laneid]:(T)-1;
        Bval = (Bval<<1) + (f0>=0?1:0);
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}

//======================================================================================    
// From column-major 32-bit-array to row-major normal array. No padding. 
//======================================================================================  
template <typename T>
__global__ void Bit32ColTo(const unsigned* __restrict__  A, T* B,
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; 
    const unsigned bx = blockIdx.x;
    unsigned Aval = A[by*A_height+bx*32+laneid];
#pragma unroll
    for (int i=0; i<32; i++) 
    {
        unsigned r0 = __shfl_sync(0xffffffff, Aval, i); //from lane-i
        B[(32*bx+i)*A_width*32+by*32+laneid] = (T)((r0>>(31-laneid)) & 0x1);
    }
}

//======================================================================================    
// From row-major 32-bit-array to row-major normal array. No padding. 
//======================================================================================  
template <typename T>
__global__ void Bit32RowTo(const unsigned* __restrict__ A, T* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    unsigned Aval = A[by*A_width+bx*32+laneid];
#pragma unroll
    for (int i=0; i<32; i++) 
        B[(by*32+i)*A_width*32+bx*32+laneid] = (T)((Aval >> (31-i)) & 0x1);
}

//======================================================================================    
// From row-major normal array to column-major 64-bit-array. This func assumes A_width
// can be divided by 64. So no padding is required.
//======================================================================================  
template <typename T>
__global__ void ToBit64Col(const T* __restrict__ A, ullong* B, const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; 
    const unsigned bx = blockIdx.x;
    ullong Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = A[(bx*32+i)*A_width+by*64+laneid];
        T f1 = A[(bx*32+i)*A_width+by*64+32+laneid];
        unsigned r0 = __ballot_sync(0xffffffff, f0>=0);
        unsigned r1 = __ballot_sync(0xffffffff, f1>=0);
        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //lo,hi
        if (laneid == i) Bval = __brevll(l0);
    }
    B[by*A_height+bx*32 + laneid] = Bval;
}

//======================================================================================    
// From row-major normal array to column-major 64-bit-array. This func is general which
// allows padding when A_width cannot divide 64.
//======================================================================================  
template <typename T>
__global__ void ToBit64ColUd(const T* __restrict__ A, ullong* B, const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; 
    const unsigned bx = blockIdx.x;
    ullong Bval = 0;
#pragma unroll
    for (int i=0; i<32; i++) 
    {
        T f0 = ( (by*64+laneid<A_width) && (bx*32+i<A_height) )?
            A[(bx*32+i)*A_width+by*64 +laneid]:(T)-1;
        T f1 = ( (by*64+32+laneid<A_width) && (bx*32+i<A_height) )?
            A[(bx*32+i)*A_width+by*64 + 32 +laneid]:(T)-1;

        unsigned r0 = __ballot_sync(0xffffffff, f0>=0);
        unsigned r1 = __ballot_sync(0xffffffff, f1>=0);
        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //lo,hi
        if (laneid == i) Bval = __brevll(l0);
    }
    if (laneid < A_height*A_width)
        B[by*gridDim.x*32+bx*32+laneid] = Bval;
}

//======================================================================================    
// From row-major normal array to row-major 64-bit-array. This func assumes A_width
// can be divided by 64. So no padding is required.
//======================================================================================  
template <typename T>
__global__ void ToBit64Row(const T* __restrict__  A, ullong* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    ullong Bval = 0;
#pragma unroll
    for (int i=0; i<64; i++)
    {
        T f0 = A[(bx*64+i)*A_width+by*32+laneid];
        Bval = (Bval<<1) | (f0>=0?1:0);
    }
    B[bx*A_width+by*32+laneid] = Bval;
}

//======================================================================================    
// From row-major normal array to row-major 64-bit-array. This func is general which
// allows padding when A_width cannot divide 64.
//======================================================================================  
template <typename T>
__global__ void ToBit64RowUd(const T* __restrict__ A, ullong* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    ullong Bval=0;
#pragma unroll
    for (int i=0; i<64; i++) 
    {
        T f0 = ( (by*32+laneid<A_width) && (bx*64+i<A_height) )?
            A[(bx*64+i)*A_width+by*32+laneid]:(T)-1;
        Bval = (Bval<<1) | (f0>=0?1:0);
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}

//======================================================================================    
// From column-major 64-bit-array to row-major normal array. No padding. 
//======================================================================================  
template <typename T>
__global__ void Bit64ColTo(const ullong* __restrict__ A, T* B, const int A_height, const int A_width)
{
    const unsigned by = blockIdx.y; 
    const unsigned bx = blockIdx.x;
    GET_LANEID;
    ullong Aval = A[by*A_height+bx*32+laneid];
    unsigned r0, r1;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(r1),"=r"(r0):"l"(Aval)); //lo,hi
#pragma unroll
    for (int i=0; i<32; i++)
    {
        unsigned r2 = __shfl_sync(0xffffffff, r0, i); //from lane-i, only 32-bit shuffle is allowed
        unsigned r3 = __shfl_sync(0xffffffff, r1, i); //r2 left, r3 right
        B[(32*bx+i)*A_width*64+by*64+laneid] = (T)((r2>>(31-laneid)) & 0x1);
        B[(32*bx+i)*A_width*64+by*64+32+laneid] = (T)((r3>>(31-laneid)) & 0x1);
    }
}

//======================================================================================    
// From row-major 64-bit-array to row-major normal array. No padding. 
//======================================================================================  
template <typename T>
__global__ void Bit64RowTo(const ullong* __restrict__ A, T* B, const int A_height, const int A_width)
{
    const unsigned by = blockIdx.y; 
    const unsigned bx = blockIdx.x;
    GET_LANEID;
    ullong Aval = A[by*A_width+bx*32+laneid];
    #pragma unroll
    for (int i=0; i<64; i++)
        B[(by*64+i)*A_width*32+bx*32+laneid] = (T)((Aval >> (63-i)) & 0x1);
}

//======================================================================================    
// This function performs 32-bit Matmul and assumes no padding. A and B are 32-bit-array,
// C is normal array in row-major. The dot product is among A-row and B-col.
// A(A_height, A_width) * B(B_height, B_width) = C(A_height, B_width), A_width = B_height
//======================================================================================  
template <typename T>
__global__ void BMM32_Arow_Bcol(const unsigned* __restrict__ A, const unsigned* __restrict__  B, 
        T* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*A_width*32];
    T* Csub = &C[blockIdx.x*B_width*32+blockIdx.y*32];
    register unsigned Cm[32] = {0};
    for (int i = 0; i < A_width; i++)
    {
        unsigned r0 = __brev(Asub[i*A_height+laneid]); 
        unsigned r1 = Bsub[i*32+laneid]; 
        #pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __ballot_sync(0xffffffff, (r1>>j) & 0x1);
            Cm[j] += __popc(r0 ^ r2) ;
        }
    }
    //This is for saving as st.128 to improve memory access efficiency
    //It is to optimize the following original version:
    //for (int i=0; i<32; i++)
    //   Csub[laneid*B_width+i] = 32*A_width - (T)(Cm[31-i])*2;
#pragma unroll
    for (int i=0; i<32; i+=4)
    {
        store128((void*)&Csub[laneid*B_width+i], 
                32*A_width-(T)(Cm[31-i])*2, 
                32*A_width-(T)(Cm[30-i])*2, 
                32*A_width-(T)(Cm[29-i])*2, 
                32*A_width-(T)(Cm[28-i])*2);
    }
}

//======================================================================================    
// This function performs 32-bit Matmul and assumes no padding. A and B are 32-bit-array,
// C is normal array in row-major. The dot product is among A-row and B-row.
// A(A_width, A_height) * B(B_height, B_width) = C(A_height, B_width), A_width = B_height
//======================================================================================  
template <typename T>
__global__ void  __launch_bounds__(32,32)
BMM32_Arow_Brow(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
        T* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    T* Csub = &C[blockIdx.x*B_width*32+blockIdx.y*32];
    register unsigned Cm[32] = {0};
    
    for (int i = 0; i < A_width; i++)
    {
        unsigned r0 = Asub[i*A_height+laneid]; 
        unsigned r1 = Bsub[i*B_width+laneid];

        #pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __shfl_sync(0xffffffff, r1, j); //from lane-j, r1 of matrix B
            Cm[j] += __popc(r0 ^ r2); //can remove C to exploit register reuse
        }
    }
    //This is for saving as st.128 to improve memory access efficiency
    //It is to optimize the following original version:
    //for (int i=0; i<32; i++)
    //Csub[laneid*B_width+i] = 32*A_width - (T)(Cm[i])*2;
    for (int i=0; i<32; i+=4)
    {
        store128((void*)&Csub[laneid*B_width+i], 
                32*A_width-(T)(Cm[i+0])*2, 
                32*A_width-(T)(Cm[i+1])*2, 
                32*A_width-(T)(Cm[i+2])*2, 
                32*A_width-(T)(Cm[i+3])*2); 
    }

}

//======================================================================================    
// This function performs 32-bit Matmul and assumes no padding. A and B are 32-bit-array,
// C is normal array in row-major. The dot product is among A-col and B-col.
// A(A_height, A_width) * B(B_width, B_height) = C(A_height, B_width), A_width = B_height
//======================================================================================  
template <typename T>
__global__ void BMM32_Acol_Bcol(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
        T* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*32*A_width];
    const unsigned* Bsub = &B[blockIdx.y*32*A_width];
    T* Csub = &C[blockIdx.x*B_width*32+blockIdx.y*32];
    register unsigned Cm[32] = {0};
    for (int i = 0; i < A_width; i++)
    {
        unsigned r0 = Asub[i*32+laneid]; 
        unsigned r1;
        unsigned r2 = Bsub[i*32+laneid];
#pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r3 = __ballot_sync(0xffffffff, (r0>>j) & 0x1);
            if (laneid == 31-j) r1 = r3;
        }
#pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r4 = __ballot_sync(0xffffffff, (r2>>(j)) & 0x1);
            Cm[j] += __popc(r1 ^ r4); 
        }
    }
    for (int i=0; i<32; i++)
        Csub[laneid*B_width+i] = 32*A_width - (T)(Cm[31-i])*2;
}

//======================================================================================    
// This function performs 32-bit Matmul and assumes no padding. A and B are 32-bit-array,
// C is normal array in row-major. The dot product is among A-col and B-row.
// A(A_width, A_height) * B(B_width, B_height) = C(A_height, B_width), A_width = B_height
//======================================================================================  
template <typename T>
__global__ void BMM32_Acol_Brow(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
        T* C, int A_height, int A_width, int B_width)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*A_width*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    T* Csub = &C[blockIdx.x*B_width*32+blockIdx.y*32];
    register unsigned Cm[32] = {0};

    for (int i = 0; i < A_width; i++)
    {
        unsigned r0 = Asub[i*32+laneid]; 
        unsigned r1 = __brev(Bsub[i*B_width+laneid]); 
        #pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __ballot_sync(0xffffffff, (r0>>j) & 0x1);
            Cm[j] += __popc(r1 ^ r2) ;
        }
    }
    for (int i=0; i<32; i++)
        Csub[i*A_height+laneid] = 32*A_width - (T)(Cm[31-i])*2;
}

//======================================================================================    
// This function performs 32-bit Matmul with padding. A and B are 32-bit-array,
// C is normal array in row-major. The dot product is among A-row and B-row.
// A(A_width, A_height) * B(B_height, B_width) = C(A_height, B_width), A_width = B_height
//======================================================================================  
template <typename T>
__global__ void BMM32_Arow_Brow_UD(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
        T* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    T* Csub = &C[blockIdx.x*B_width*32+blockIdx.y*32];
    register unsigned Cm[32] = {0};

    for (int i = 0; (i*32) < A_width; i++)
    {
        unsigned r0 = Asub[i*32*gridDim.x+laneid]; 
        unsigned r1 = Bsub[i*32*gridDim.y+laneid];
#pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __shfl(0xffffffff, r1, j); //from lane-j, r1 of matrix B
            Cm[j] += __popc(r0 ^ r2); //can remove C to exploit register reuse
        }
    }
    if ( (blockIdx.x*32+laneid)<A_height )
    {
        for (int i=0; i<32; i++)
            if (blockIdx.y*32+i<B_width)
                Csub[laneid*B_width+i] = A_width - (T)(Cm[i])*2;
    }
}

//======================================================================================    
// This function performs 64-bit Matmul and assumes no padding. A and B are 64-bit-array,
// C is normal array in row-major. The dot product is among A-row and B-col.
// A(A_height, A_width) * B(B_height, B_width) = C(A_height, B_width), A_width = B_height
//======================================================================================  
template <typename T>
__global__ void BMM64_Arow_Bcol(const ullong* __restrict__ A, const ullong* __restrict__ B, 
        T* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const ullong* Asub = &A[blockIdx.x*64];
    const ullong* Bsub = &B[blockIdx.y*A_width*64];
    float* Csub = (float*)&C[blockIdx.x*B_width*64+blockIdx.y*64];
    register unsigned Cm[64] = {0};

    for (int i = 0; i < A_width; i++) {
        ullong b0 = (Bsub[i*64+laneid]);
        ullong b1 = (Bsub[i*64+32+laneid]);

        ullong a0 = __brevll(Asub[i*A_height+laneid]);
        ullong a1 = __brevll(Asub[i*A_height+32+laneid]);
#pragma unroll
        for (int j=0; j<64; j++) {
            unsigned u0 = __ballot_sync(0xffffffff, (b0>>j) & 0x1);
            unsigned u1 = __ballot_sync(0xffffffff, (b1>>j) & 0x1);
            ullong l0;
            asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(u0),"r"(u1));
            Cm[j] += (__popcll(a0^l0)<<16) + __popcll(a1^l0);
        }
    }
    for (int i=0; i<64; i+=2)
    {
        short t0,t1,t2,t3;
        asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[64-i-1]));
        asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t3),"=h"(t2):"r"(Cm[64-i-2]));
        store64(&Csub[laneid*B_width+i],64*A_width - 2*(T)(t0), 64*A_width - 2*(T)(t2));
        store64(&Csub[(laneid+32)*B_width+i],64*A_width - 2*(T)(t1), 64*A_width - 2*(T)(t3));
    }
}

//======================================================================================    
// This function performs 64-bit Matmul and assumes no padding. A and B are 64-bit-array,
// C is normal array in row-major. The dot product is among A-row and B-col.
// A(A_height, A_width) * B(B_width, B_height) = C(A_height, B_width), A_width = B_height
//======================================================================================  
template <typename T> 
__global__ void BMM64_Arow_Brow(const ullong* __restrict__ A, const ullong* __restrict__ B, 
        T* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const ullong* Asub = &A[blockIdx.x*64];
    const ullong* Bsub = &B[blockIdx.y*64];
    T* Csub = &C[blockIdx.x*B_width*64+blockIdx.y*64];
    register unsigned Cm[64] = {0};

    for (int i = 0; i < A_width; i++)
    {
        ullong a0 = (Asub[i*A_height+laneid]);
        ullong a1 = (Asub[i*A_height+32+laneid]);
        ullong b0 = (Bsub[i*B_width+laneid]);
        ullong b1 = (Bsub[i*B_width+32+laneid]);
        #pragma unroll
        for (int j=0; j<32; j++)
        {
            ullong l0 = __shfl_sync(0xffffffff,b0,j);
            ullong l1 = __shfl_sync(0xffffffff,b1,j);
            Cm[j] += (__popcll(a0^l0)<<16) + __popcll(a1^l0);
            Cm[32+j] += (__popcll(a0^l1)<<16) + __popcll(a1^l1);
        }
    }
    #pragma unroll
    for (int i=0; i<64; i+=2)
    {
        short t0,t1,t2,t3;
        asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[i]));
        asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t3),"=h"(t2):"r"(Cm[i+1]));
        store64(&Csub[laneid*B_width+i],64*A_width - 2*(T)(t0), 64*A_width - 2*(T)(t2));
        store64(&Csub[(laneid+32)*B_width+i],64*A_width - 2*(T)(t1), 64*A_width - 2*(T)(t3));
    }
}

//======================================================================================    
// This function performs 64-bit Matmul with padding. A and B are 64-bit-array,
// C is normal array in row-major. The dot product is among A-row and B-row.
// A(A_width, A_height) * B(B_height, B_width) = C(A_height, B_width), A_width = B_height
//======================================================================================  
template <typename T>
__global__ void BMM64_Arow_Brow_UD(const ullong* __restrict__ A, const ullong* __restrict__ B, 
        T* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const ullong* Asub = &A[blockIdx.x*64];
    const ullong* Bsub = &B[blockIdx.y*64];
    T* Csub = &C[blockIdx.x*B_width*64+blockIdx.y*64];
    register unsigned Cm[64] = {0};

    for (int i = 0; (i*64) < A_width; i++)
    {
        ullong a0 = (Asub[i*64*gridDim.x+laneid]);
        ullong a1 = (Asub[i*64*gridDim.x+32+laneid]);
        ullong b0 = (Bsub[i*64*gridDim.y+laneid]);
        ullong b1 = (Bsub[i*64*gridDim.y+32+laneid]);
        #pragma unroll
        for (int j=0; j<32; j++)
        {
            ullong l0 = __shfl_sync(0xffffffff,b0,j);
            ullong l1 = __shfl_sync(0xffffffff,b1,j);
            Cm[j] += (__popcll(a0^l0)<<16) + __popcll(a1^l0);
            Cm[32+j] += (__popcll(a0^l1)<<16) + __popcll(a1^l1);

        }
    }
    for (int i=0; i<64; i++)
    {
        if (blockIdx.y*64+i<B_width)
        {
            short t0,t1;
            asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[i]));
            if (blockIdx.x*64+laneid<A_height)
                Csub[laneid*B_width+i] = A_width - (T)(t0)*2;
            if (blockIdx.x*64+32+laneid<A_height)
                Csub[(laneid+32)*B_width+i] = A_width - (T)(t1)*2;

        }
    }

}

__global__ void BMM32_MT_M_S(const unsigned* __restrict__ A, const unsigned* __restrict__ B, float* C, int m, int n, int k)
{
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >>5);
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    float* Csub = &C[blockIdx.x*k*32+blockIdx.y*32];
    register unsigned Cc = 0;

    for (int i=0; i<n; i++)
    {
        unsigned r0 = Asub[i*m+warpid];
        unsigned r1 = Bsub[i*k+laneid];
        Cc += __popc(r0 ^ r1);
    }
    Csub[warpid*k+laneid] = 32*n - (float)Cc*2;
 }


__global__ void BMM64_MT_M_S(const ullong* __restrict__ A, const ullong* __restrict__ B, float* C, int m, int n, int k)
{
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    const ullong* Asub = &A[blockIdx.x*64];
    const ullong* Bsub = &B[blockIdx.y*64];
    float* Csub = &C[blockIdx.x*k*64+blockIdx.y*64];
    register unsigned C0=0, C1=0, C2=0, C3=0;

    for (int i = 0; i < n; i++)
    {
        ullong a0 = Asub[i*m+warpid];
        ullong a1 = Asub[i*m+32+warpid];
        ullong b0 = Bsub[i*k+laneid];
        ullong b1 = Bsub[i*k+32+laneid];

        C0 += __popcll(a0 ^ b0);
        C1 += __popcll(a1 ^ b0);
        C2 += __popcll(a0 ^ b1);
        C3 += __popcll(a1 ^ b1);
    }
    Csub[warpid*k+laneid] = 64*n - 2*float(C0);
    Csub[(warpid+32)*k+laneid] = 64*n - 2*float(C1);
    Csub[warpid*k+laneid+32] = 64*n - 2*float(C2);
    Csub[(warpid+32)*k+laneid+32] = 64*n - 2*float(C3);
}


//=============

__global__ void BMM32_BIN(const unsigned* __restrict__ A, const unsigned* __restrict__ B, unsigned* C, int m, int n, int k)
{
    GET_LANEID;
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    unsigned* Csub = &C[blockIdx.y*m+blockIdx.x*32];
    register int Cm[32] = {0};
    for (int i=0; (i*32) < n; i++)
    {
        unsigned r0 = Asub[i*m+laneid];
        unsigned r1 = Bsub[i*k+laneid];
#pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __shfl_sync(0xffffffff, r1, j); //from lane-j, r1 of weight matrix
            Cm[j] += __popc(r0 ^ r2);
        }
    }
    unsigned c = 0;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        float t = n - 2*(float)Cm[i];
        c = c+c+(t>=0);
    }
    Csub[laneid] = c;
}


__global__ void BMM32S_BIN(const unsigned* __restrict__ A, const unsigned* __restrict__ B, unsigned* C, int m, int n, int k)
{
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >>5);
    const unsigned* Asub = &A[blockIdx.x*32];
    const unsigned* Bsub = &B[blockIdx.y*32];
    unsigned* Csub = &C[blockIdx.x*k*32+blockIdx.y*32];
    register unsigned Cc = 0;

    for (int i=0; (32*i)<n; i++)
    {
        unsigned r0 = Asub[i*m+warpid];
        unsigned r1 = Bsub[i*k+laneid];
        Cc += __popc(r0 ^ r1);
    }
    unsigned r2 = __ballot_sync(0xffffffff, n-2*(float)Cc>=0);
    Csub[warpid] = __brev(r2);
}


__global__ void BMM64_BIN(const ullong* __restrict__ A, const ullong* __restrict__ B, 
        ullong* C, const int A_height, const int A_width, const int B_width)
{
    GET_LANEID;
    const ullong* Asub = &A[blockIdx.x*64];
    const ullong* Bsub = &B[blockIdx.y*64];
    ullong* Csub = &C[blockIdx.x*B_width*64+blockIdx.y*64];
    register unsigned Cm[64] = {0};

    for (int i = 0; (i*64) < A_width; i++)
    {
        ullong a0 = (Asub[i*64*gridDim.x+laneid]);
        ullong a1 = (Asub[i*64*gridDim.x+32+laneid]);
        ullong b0 = (Bsub[i*64*gridDim.y+laneid]);
        ullong b1 = (Bsub[i*64*gridDim.y+32+laneid]);
        #pragma unroll
        for (int j=0; j<32; j++)
        {
            ullong l0 = __shfl_sync(0xffffffff, b0,j);
            ullong l1 = __shfl_sync(0xffffffff, b1,j);
            Cm[j] += (__popcll(a0^l0)<<16) + __popcll(a1^l0);
            Cm[32+j] += (__popcll(a0^l1)<<16) + __popcll(a1^l1);

        }
    }
    ullong C0 = 0;
    ullong C1 = 0;
    for (int i=0; i<64; i++)
    {
        short t0,t1;
        asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[i])); //lo, hi
        //if (bx*64+laneid<(p->input_height))
        C0 |= (((((float)A_width)-2*(float)t0>=0)?(ullong)1:(ullong)0)<<(63-i));
        C1 |= (((((float)A_width)-2*(float)t1>=0)?(ullong)1:(ullong)0)<<(63-i));
    }
    Csub[laneid] = C0;
    Csub[laneid+32] = C1;
}


__global__ void BMM64S_BIN(const ullong* __restrict__ A, const ullong* __restrict__ B, ullong* C, int m, int n, int k)
{
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    const ullong* Asub = &A[blockIdx.x*64];
    const ullong* Bsub = &B[blockIdx.y*64];
    ullong* Csub = &C[blockIdx.x*k*64+blockIdx.y*64];
    register unsigned C0=0, C1=0, C2=0, C3=0;

    for (int i = 0; i < n; i++)
    {
        ullong a0 = Asub[i*m+warpid];
        ullong a1 = Asub[i*m+32+warpid];
        ullong b0 = Bsub[i*k+laneid];
        ullong b1 = Bsub[i*k+32+laneid];

        C0 += __popcll(a0 ^ b0);
        C1 += __popcll(a1 ^ b0);
        C2 += __popcll(a0 ^ b1);
        C3 += __popcll(a1 ^ b1);
    }
    unsigned r0 = __ballot_sync(0xffffffff, (((float)64*n)-2*(float)C0>=0));
    unsigned r1 = __ballot_sync(0xffffffff, (((float)64*n)-2*(float)C2>=0));
    unsigned r2 = __ballot_sync(0xffffffff, (((float)64*n)-2*(float)C1>=0));
    unsigned r3 = __ballot_sync(0xffffffff, (((float)64*n)-2*(float)C3>=0));

    ullong l0,l1;
    asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1));//lo,hi
    asm volatile("mov.b64 %0, {%1,%2};":"=l"(l1):"r"(r2),"r"(r3));//lo,hi

    Csub[warpid] = __brevll(l0);
    Csub[warpid+32] = __brevll(l1);

}













