/*
 * =====================================================================================
 *
 *       Filename:  tensorcore_kernel.cu
 *
 *    Description:  Accelerate BMM via TensorCores in Turing GPU
 *                  Please see our TPDS paper "Accelerating Binarized Neural 
 *                  Networks via Bit-Tensor-Cores in Turing GPUs" for detail.
 *                  https://arxiv.org/abs/2006.16578
 *
 *        Version:  1.0
 *        Created:  11/04/2019 11:43:58 AM, Richland, WA, USA.
 *       Revision:  none
 *       Compiler:  nvcc -arch=sm_75
 *
 *      PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850.
 * 
 *         Author:  Ang Li
 *        Website:  https://www.angliphd.com
 *
 * =====================================================================================
 */


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>


#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));

using namespace nvcuda;

typedef unsigned char uchar;

void BMMAcpu(const unsigned* A, const unsigned* B, int* C, const unsigned m, const unsigned n, const unsigned k)
{
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
        {
            for (int t=0; t<k/32; t++)
            {
                unsigned tmp = (A[i*(k/32)+t] ^ B[j*(k/32)+t]);
                unsigned s = 0;
                for(int u=0; u<32; u++) 
                {
                    s += (tmp & 0x1);
                    tmp = (tmp >> 1);
                }
                C[i*n+j] += s;
            }
        }
    }
}


__global__ void float_to_half(const float* __restrict__ A, half* __restrict__ B, const int n)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) B[idx] = __float2half(A[idx]);
}

__global__ void half_to_float(const half* __restrict__ A, float* __restrict__ B, const int n)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < n) B[idx] = __half2float(A[idx]);
}



template <typename T>
__global__ void BMMA_toBit32Col(const T* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    T f0 = A[(bx*32+laneid)*A_width+by];
    unsigned r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
    if (laneid == 0) B[(by*A_height/32)+bx] = r0;
}


template <typename T>
__global__ void BMMA_toBit32Row(const T* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    T f0 = A[bx*A_width+by*32+laneid];
    unsigned r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
    if (laneid == 0) B[(bx*A_width/32)+by] = r0;
}



template <typename T>
__global__ void BMMA_toBit32Col_new(const T* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 

    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;

    T f0 = A[(bx*128+wx*32+laneid)*A_width+(by*8)+wy];
    unsigned r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
    //if (laneid == 0) B[((by*8+wy)*A_height/32)+wx] = r0;
    if (laneid == 0) B[(by*gridDim.x+bx)*8*(128/(4*8))+wy*128/(4*8)+wx] = r0;
}


template <typename T>
__global__ void BMMA_toBit32Row_new(const T* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;

    T f0 = A[(bx*8+wx)*A_width+by*128+wy*32+laneid];
    unsigned r0 = __brev(__ballot_sync(0xffffffff, f0>=0?1:0));
    //if (laneid == 0) B[(bx*8+wx)*A_width/32+wy] = r0;
    if (laneid == 0) B[(bx*gridDim.y+by)*8*128/(4*8)+wx*128/(4*8)+wy] = r0;
}




template<int M, int N>
__global__ void BMMA(const unsigned *A, const unsigned *B, int *C, const unsigned A_height,
const unsigned B_height, const unsigned A_width)
{
    using namespace nvcuda::wmma::experimental;
    __shared__ uint4 As[32];
    __shared__ uint4 Bs[32];

    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;
    const unsigned bx = blockIdx.x; // gridIdx.x = A_height / (8*M)
    const unsigned by = blockIdx.y; // gridIdx.y = B_height / (8*N)

    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (unsigned k=0; k<A_width; k++) //each step processes 128B
    {
        if (wx ==0 && wy == 0)
        {
            As[laneid] = ((uint4*)A)[(bx*M*8+laneid)*A_width+k];
            Bs[laneid] = ((uint4*)B)[(by*N*8+laneid)*A_width+k];
        }
        __syncthreads();

        load_matrix_sync(a_frag, &As[wx*8], 128);
        load_matrix_sync(b_frag, &Bs[wy*8], 128);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    int* bC = &C[(bx*M+wx)*8*B_height+(by*N+wy)*8];
    for (int i=0; i<c_frag.num_elements; i++)
        c_frag.x[i] = (A_width*128) - (2*c_frag.x[i]); 
    store_matrix_sync(bC, c_frag, B_height, wmma::mem_row_major);
}


template<int M, int N>
__global__ void BMMApipe(const unsigned *A, const unsigned *B, int *C, const unsigned A_height,
const unsigned B_height, const unsigned A_width)
{
    using namespace nvcuda::wmma::experimental;
    __shared__ uint4 As[64];
    __shared__ uint4 Bs[64];

    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;
    const unsigned bx = blockIdx.x; // gridIdx.x = A_height / (8*M)
    const unsigned by = blockIdx.y; // gridIdx.y = B_height / (8*N)

    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> aa_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> bb_frag;

    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    if (wx ==0 && wy == 0)
    {
        As[laneid] = ((uint4*)A)[(bx*M*8+laneid)*A_width];
        Bs[laneid] = ((uint4*)B)[(by*N*8+laneid)*A_width];
    }
    __syncthreads();

    for (unsigned k=0; k<A_width/2; k++)
    {
        load_matrix_sync(a_frag, &As[wx*8], 128);
        load_matrix_sync(b_frag, &Bs[wy*8], 128);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);

        if (wx ==0 && wy == 0 )
        {
            As[laneid+32] = ((uint4*)A)[(bx*M*8+laneid)*A_width+A_width/2+k];
            Bs[laneid+32] = ((uint4*)B)[(by*N*8+laneid)*A_width+A_width/2+k];
        }
        __syncthreads();

        if (wx ==0 && wy == 0 && k<(A_width/2-1))
        {
            As[laneid] = ((uint4*)A)[(bx*M*8+laneid)*A_width+k+1];
            Bs[laneid] = ((uint4*)B)[(by*N*8+laneid)*A_width+k+1];
        }
        load_matrix_sync(aa_frag, &As[32+wx*8], 128);
        load_matrix_sync(bb_frag, &Bs[32+wy*8], 128);
        bmma_sync(c_frag, aa_frag, bb_frag, c_frag);
        __syncthreads();
    }


    int* bC = &C[(bx*M+wx)*8*B_height+(by*N+wy)*8];
    for (int i=0; i<c_frag.num_elements; i++)
        c_frag.x[i] = (A_width*128) - (2*c_frag.x[i]); 
    store_matrix_sync(bC, c_frag, B_height, wmma::mem_row_major);
}












//A_width is the number of unsigned, which is 4B*8bit/B*A_width bits
template<int M, int N>
__global__ void BMMA_bin(const unsigned *A, const unsigned *B, unsigned *C, 
        const unsigned A_height, const unsigned B_height, const unsigned A_width)
{
    using namespace nvcuda::wmma::experimental;
    __shared__ uint4 As[32];
    __shared__ uint4 Bs[32];

    const unsigned laneid = threadIdx.x;
    const unsigned wx = threadIdx.y;
    const unsigned wy = threadIdx.z;
    const unsigned bx = blockIdx.x; // gridIdx.x = A_height / (8*M)
    const unsigned by = blockIdx.y; // gridIdx.y = B_height / (8*N)
    const unsigned wid = threadIdx.y * N + threadIdx.z;

    wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (unsigned k=0; k<A_width; k++) //each step processes 128B
    {
        if (wx ==0 && wy == 0)
        {
            As[laneid] = ((uint4*)A)[(bx*M*8+laneid)*A_width+k];
            Bs[laneid] = ((uint4*)B)[(by*N*8+laneid)*A_width+k];
        }

        __syncthreads();

        load_matrix_sync(a_frag, &As[wx*8], 128);
        load_matrix_sync(b_frag, &Bs[wy*8], 128);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    __shared__ int Cs[8*M][8*N]; //asume N%4==0 for C access coalescing
    int* bCs = &Cs[8*(wx)][8*(wy)];
    store_matrix_sync(bCs, c_frag, 8*N, wmma::mem_row_major);

    __syncthreads();

    for (unsigned i=wid; i<8*M; i+=(M*N))
    {
        for (unsigned j=0; j<(N/4); j++)
        {
            unsigned r0 = __ballot_sync(0xffffffff, (((float)A_width*128)-2*(float)Cs[i][j]>=0));
            if (laneid == 0) C[(bx*8*M+i)*(B_height/32)+by*(N/4)+j] = r0;
        }
    }
}

template<int M>
__global__ void BMMAS(const unsigned *A, const unsigned *B, int *C, const unsigned m, const unsigned n, const unsigned k)
{
    using namespace nvcuda::wmma::experimental;
    int bx = blockIdx.x * blockDim.y + threadIdx.y; 
    int by = blockIdx.y;
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (int i=0; i<(k/128); i+=M)
    {
        #pragma unroll
        for (int j=i; j<i+M; j++)
        {
            if (j==(k/128)) goto endloop;
            load_matrix_sync(a_frag, A + bx*8*k/32 + j*128/32, k); //M
            load_matrix_sync(b_frag, B + by*8*k/32 + j*128/32, k); //N
            bmma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
endloop:

    #pragma unroll
    for (int i=0; i<c_frag.num_elements; i++)
        c_frag.x[i] = k - 2*c_frag.x[i]; 
    store_matrix_sync(C+(bx*8*n+by*8), c_frag, n, wmma::mem_row_major);
}


__global__ void BMMAS_new(const unsigned *A, const unsigned *B, int *C, const unsigned m, const unsigned n, const unsigned k)
{
    using namespace nvcuda::wmma::experimental;
    int bx = blockIdx.x * blockDim.y + threadIdx.y; 
    int by = blockIdx.y;
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);
    
    for (int j=0; j<(k/128); j++)
    {
        load_matrix_sync(a_frag, A + bx*8*k/32 + j*128*8/32, 128);
        load_matrix_sync(b_frag, B + by*8*k/32 + j*128*8/32, 128);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    #pragma unroll
    for (int i=0; i<c_frag.num_elements; i++)
        c_frag.x[i] = k - 2*c_frag.x[i]; 
    store_matrix_sync(C+(bx*8*n+by*8), c_frag, n, wmma::mem_row_major);
}


template<int M>
__global__ void BMMAS_bin(const unsigned *A, const unsigned *B, unsigned *C, const unsigned A_height, const unsigned B_height, const unsigned A_width)
{
    using namespace nvcuda::wmma::experimental;
    __shared__ int Cs[64];
    uchar* Cb = (uchar*)(&C[0]);

    GET_LANEID;

    int bx = blockIdx.x * blockDim.y + threadIdx.y; 
    int by = blockIdx.y;
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (int i=0; i<(A_width/128); i+=M)
    {
#pragma unroll
        for (int j=i; j<i+M; j++)
        {
            if (j==(A_width/128)) goto endloop;
            load_matrix_sync(a_frag, A + bx*8*A_width/32 + j*128/32, A_width);
            load_matrix_sync(b_frag, B + by*8*A_width/32 + j*128/32, A_width);
            bmma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
endloop:
    //store_matrix_sync(C+(bx*8*n+by*8), c_frag, n, wmma::mem_row_major);
    store_matrix_sync(Cs, c_frag, 8, wmma::mem_row_major);

    union{ unsigned data; uchar elements[4];} p0, p1;
    p0.data =  __ballot_sync(0xffffffff, (((float)A_width)-2*(float)Cs[laneid]>=0));
    p1.data =  __ballot_sync(0xffffffff, (((float)A_width)-2*(float)Cs[32+laneid]>=0));

    if (laneid < 4)
    {
        Cb[(bx*8+laneid)*(B_height/8)+by] = p0.elements[laneid]; 
        Cb[(bx*8+4+laneid)*(B_height/8)+by] = p1.elements[laneid]; 
    }
}



__global__ void BMMAS_bin_new(const unsigned *A, const unsigned *B, unsigned *C, const unsigned A_height, const unsigned B_height, const unsigned A_width)
{
    using namespace nvcuda::wmma::experimental;
    __shared__ int Cs[64];
    uchar* Cb = (uchar*)(&C[0]);

    GET_LANEID;

    int bx = blockIdx.x * blockDim.y + threadIdx.y; 
    int by = blockIdx.y;
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    const unsigned* Asub = A + (bx*A_width>>2);
    const unsigned* Bsub = B + (by*A_width>>2);
    for (int i=0; i<(A_width>>2); i+=32)
    {
        load_matrix_sync(a_frag, Asub + i, 128);
        load_matrix_sync(b_frag, Bsub + i, 128);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    /*
    for (int i=0; i<(A_width/128); i++)
    {
        load_matrix_sync(a_frag, A + bx*8*A_width/32 + i*128*8/32, 128);
        load_matrix_sync(b_frag, B + by*8*A_width/32 + i*128*8/32, 128);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    */

    //store_matrix_sync(C+(bx*8*n+by*8), c_frag, n, wmma::mem_row_major);
    
    //store_matrix_sync(Cs, c_frag, 8, wmma::mem_row_major);
    store_matrix_sync(Cs, c_frag, 8, wmma::mem_col_major); //This can be row or col major

    union{ unsigned data; uchar elements[4];} p0, p1;
    p0.data = __ballot_sync(0xffffffff, ((A_width)-2*Cs[laneid]>=0));
    p1.data = __ballot_sync(0xffffffff, ((A_width)-2*Cs[32+laneid]>=0));

    if (laneid < 4)
    {
        //Cb[(bx*8+laneid)*(B_height/8)+by] = p0.elements[laneid]; 
        //Cb[(bx*8+4+laneid)*(B_height/8)+by] = p1.elements[laneid]; 
        
        Cb[(by*gridDim.x+(bx/16))*8*128/32+laneid*128/32+bx%16] = p0.elements[laneid]; 
        Cb[(by*gridDim.x+(bx/16))*8*128/32+(32+laneid)*128/32+bx%16] = p1.elements[laneid]; 
    }
}



