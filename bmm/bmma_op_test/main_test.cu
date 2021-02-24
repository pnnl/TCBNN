/*
 * =====================================================================================
 *    Description:  Accelerate BNN via TensorCores in Turing/Ampere GPU
 *                  Please see our TPDS paper "Accelerating Binarized Neural 
 *                  Networks via Bit-Tensor-Cores in Turing GPUs" for detail.
 *                  https://arxiv.org/abs/2006.16578
 *
 *        Version:  1.0
 *        Created:  11/04/2019 11:43:58 AM, Richland, WA, USA.
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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32
#define THREADS_PER_CTA 64

using namespace nvcuda;

__global__ void read_global(const unsigned *A, const unsigned *B, const int *C, int *D, const unsigned m, const unsigned n, const unsigned k, const unsigned t, int* time)
{
    using namespace nvcuda::wmma::experimental;
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int* Dg = D+(bx*8*n+by*8);
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;


    //wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> aa_frag;
    //wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> bb_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> cc_frag;
    //wmma::fragment<wmma::accumulator, 8, 8, 128, int> ccc_frag;
    //wmma::fragment<wmma::accumulator, 8, 8, 128, int> cccc_frag;
    //wmma::fragment<wmma::accumulator, 8, 8, 128, int> ccccc_frag;

    load_matrix_sync(c_frag, C+(bx*8*n+by*8), n, wmma::mem_row_major);
    int start = 0;
    int sum = 0;

    for (int i=0; i<(k/128); i++)
    {
        const unsigned* Ag = A + bx*8*k/32 + i*128/32;
        __syncthreads();
        
        load_matrix_sync(a_frag, Ag, t);
        //load_matrix_sync(aa_frag, B, t+128);
        __threadfence_block();

        //load_matrix_sync(a_frag, A + bx*8*k/32 + i*128/32, k);
        //load_matrix_sync(b_frag, B + by*8*k/32 + i*128/32, k);
        //load_matrix_sync(bb_frag, B + by*8*k/32, k);

        __threadfence_block();
        __syncthreads();


        start = clock();

        bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(cc_frag, a_frag, b_frag, cc_frag);
        //bmma_sync(ccc_frag, a_frag, b_frag, ccc_frag);
        //bmma_sync(cccc_frag, a_frag, b_frag, cccc_frag);
        //bmma_sync(ccccc_frag, a_frag, b_frag, ccccc_frag);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);

        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);

        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);

        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);
        //bmma_sync(c_frag, a_frag, b_frag, c_frag);

        
        sum += (clock() - start);
    }
    store_matrix_sync(Dg, c_frag, n, wmma::mem_row_major);
    //store_matrix_sync(Dg, cc_frag, n, wmma::mem_row_major);
    //store_matrix_sync(Dg, ccc_frag, n, wmma::mem_row_major);
    //store_matrix_sync(Dg, cccc_frag, n, wmma::mem_row_major);
    //store_matrix_sync(Dg, ccccc_frag, n, wmma::mem_row_major);
    time[(bx*gridDim.y + by)*32+threadIdx.x] = sum / (k/128);
}

__global__ void write_global(const unsigned *A, const unsigned *B, const int *C, int *D, const unsigned m, const unsigned n, const unsigned k, const unsigned t, int* time)
{
    using namespace nvcuda::wmma::experimental;
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int* Dg = D+(bx*8*n+by*8);
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    load_matrix_sync(c_frag, C+(bx*8*n+by*8), n, wmma::mem_row_major);
    int start = 0;
    int sum = 0;

    for (int i=0; i<(k/128); i++)
    {
        load_matrix_sync(a_frag, A + bx*8*k/32 + i*128/32, k);
        load_matrix_sync(b_frag, B + by*8*k/32 + i*128/32, k);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();

    start = clock();
    store_matrix_sync(Dg, c_frag, t, wmma::mem_row_major);
    __threadfence_block();
    sum += (clock() - start);
    
    time[(bx*gridDim.y + by)*32+threadIdx.x] = sum;
}

__global__ void read_shared(const unsigned *A, const unsigned *B, const int *C, int *D, const unsigned m, const unsigned n, const unsigned k, const unsigned t, int* time)
{
    using namespace nvcuda::wmma::experimental;
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int* Dg = D+(bx*8*n+by*8);
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    load_matrix_sync(c_frag, C+(bx*8*n+by*8), n, wmma::mem_row_major);
    int start = 0;
    int sum = 0;
    __shared__ unsigned sbuf[8192];

    for (int i=threadIdx.x; i<8192; i+=32)
        sbuf[i] = A[i];

    __syncthreads();
    
    for (int i=0; i<(k/128); i++)
    {
        start = clock();
        load_matrix_sync(a_frag, sbuf, t);
        __threadfence_block();
        sum += (clock() - start);

        //load_matrix_sync(a_frag, A + bx*8*k/32 + i*128/32, k);
        load_matrix_sync(b_frag, B + by*8*k/32 + i*128/32, k);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(Dg, c_frag, n, wmma::mem_row_major);
    time[(bx*gridDim.y + by)*32+threadIdx.x] = sum / (k/128);
}

__global__ void write_shared(const unsigned *A, const unsigned *B, const int *C, int *D, const unsigned m, const unsigned n, const unsigned k, const unsigned t, int* time)
{
    using namespace nvcuda::wmma::experimental;
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int* Dg = D+(bx*8*n+by*8);
	wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
    load_matrix_sync(c_frag, C+(bx*8*n+by*8), n, wmma::mem_row_major);
    int start = 0;
    int sum = 0;
    __shared__ int sbuf[8192];

    for (int i=0; i<(k/128); i++)
    {
        load_matrix_sync(a_frag, A + bx*8*k/32 + i*128/32, k);
        load_matrix_sync(b_frag, B + by*8*k/32 + i*128/32, k);
        bmma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
    
    start = clock();
    store_matrix_sync(sbuf, c_frag, t, wmma::mem_row_major);
    __threadfence_block();
    sum += (clock() - start);

    time[(bx*gridDim.y + by)*32+threadIdx.x] = sum;
}











void WMMABOOLCPU(const unsigned* A, const unsigned* B, const int* C, int* D, const unsigned m, const unsigned n, const unsigned k)
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
                D[i*n+j] += s;
            }
        }
    }
    for (int i=0; i<m; i++)
        for (int j=0; j<n; j++)
            D[i*n+j] += C[i*n+j];
}


int main(int argc, char* argv[])
{
	cudaError_t cuda_status;
	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
		printf("CudaSetDevice failed! ");
        exit(1);
	}

    if (argc != 5) 
    {
        printf("./main M N K T\n");
        exit(1);
    }

    unsigned m = atoi(argv[1]);
    unsigned n = atoi(argv[2]);
    unsigned k = atoi(argv[3]);
    unsigned t = atoi(argv[4]);

    assert((m%8)==0);
    assert((n%8)==0);
    assert((k%128)==0);

    unsigned *A = NULL;
    unsigned *B = NULL;
    int *C = NULL;
    int *D = NULL;
    int *E = NULL;

    cudaMallocHost((void**)&A, m*k*sizeof(unsigned)/32);
    cudaMallocHost((void**)&B, k*n*sizeof(unsigned)/32);
    cudaMallocHost((void**)&C, m*n*sizeof(int));
    cudaMallocHost((void**)&D, m*n*sizeof(int));
    cudaMallocHost((void**)&E, m*n*sizeof(int));

    srand(4);
    for (int i=0; i<m*k/32; i++) A[i] = rand();
    for (int i=0; i<k*n/32; i++) B[i] = rand();
    for (int i=0; i<m*n; i++) C[i] = rand();
    for (int i=0; i<m*n; i++) D[i] = 0;
    for (int i=0; i<m*n; i++) E[i] = 0;

    unsigned *A_gpu = NULL;
    unsigned *B_gpu = NULL;
    int *C_gpu = NULL;
    int *D_gpu = NULL;

    cudaMalloc((void**)&A_gpu, m*k*sizeof(unsigned)/32);
    cudaMalloc((void**)&B_gpu, k*n*sizeof(unsigned)/32);
    cudaMalloc((void**)&C_gpu, m*n*sizeof(int));
    cudaMalloc((void**)&D_gpu, m*n*sizeof(int));

    cudaMemcpy(A_gpu, A, m*k*sizeof(unsigned)/32, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, k*n*sizeof(unsigned)/32, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, m*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(D_gpu, D, m*n*sizeof(int), cudaMemcpyHostToDevice);

	// computing bgemm using tensor core
	printf("[*] Computing D = A * B + C with Tensor Cores...\n");
	dim3 gridDim, blockDim;
    blockDim.x = 32*1;
    //gridDim.x = m/8;
    //gridDim.y = n/8;
    gridDim.x = 1;
    gridDim.y = 1;

    int total_threads = blockDim.x * gridDim.x * gridDim.y;
    int* time = NULL;
    int* time_gpu = NULL;
    cudaMallocHost((void**)&time, total_threads*sizeof(int));
    cudaMalloc((void**)&time_gpu, total_threads*sizeof(int));
    memset(time,0,total_threads*sizeof(int));
    cudaMemset(time_gpu,0,total_threads*sizeof(int));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaDeviceSynchronize();
    
    //start testing
	cudaEventRecord(start);

    //read_global<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, D_gpu, m, n, k, t, time_gpu);
    //write_global<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, D_gpu, m, n, k, t, time_gpu);
    //read_shared<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, D_gpu, m, n, k, t, time_gpu);
    write_shared<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, D_gpu, m, n, k, t, time_gpu);


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

    cudaMemcpy(D, D_gpu, m*n*sizeof(int), cudaMemcpyDeviceToHost);
    WMMABOOLCPU(A, B, C, E, m, n, k);
    //validate
    bool valid = true;
    for (int i=0; i<m*n; i++) if (D[i] != E[i]) { valid=false; break;}
    
    /*
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
            printf("(%d) ",D[i*n+j] - E[i*n+j]);
        printf("\n");

    }
    */
	
    float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[+] GPU(with Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
	printf("[+] TOPS: %.3lf\n", ((double)m*n*k*2) / milliseconds / 1e9);
	printf("[+] Validation: %s.\n", valid?"True":"False");

    cudaMemcpy(time, time_gpu, total_threads * sizeof(int), cudaMemcpyDeviceToHost); 
    double avg_time = 0; 
    for (int i=0; i<total_threads; i++) avg_time += time[i];
    avg_time /= total_threads;
    printf("[+] Avg Time: %.0lf cycles.\n", avg_time);

	// for Performance Metrics
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(D);
    cudaFreeHost(E);
    cudaFreeHost(time);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(D_gpu);
    cudaFree(time_gpu);

	return 0;
}
