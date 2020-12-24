/*
 * =====================================================================================
 *
 *       Filename:  benchmark-bmm.cu
 *
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


#include <iostream>
#include <cublas_v2.h>
#include <sys/time.h>

#include "binary_kernels.cu"
#include "matmul_kernel.cu"
#include "tensorcore_kernel.cu"

#define TEST_TIMES 10

using namespace std;

bool check_result(float* p1, float* p2, const int N) 
{
    bool flag = true;
    for (int i = 0; i < N * N; i ++) {
        //printf("(%f,%f)",p1[i],p2[i]);
        float diff = p1[i] - p2[i];
        if (fabs(diff) > 1e-6) {
            flag = false;
        }
    }
    return flag;
}

bool check_result(float* p1, int* p2, const int N) 
{
    bool flag = true;
    for (int i = 0; i < N * N; i ++) {
        //printf("(%.0f,%d)",p1[i],p2[i]);
        float diff = p1[i] - (float)p2[i];
        if (fabs(diff) > 1e-6) {
            flag = false;
        }
    }
    return flag;
}



int main(int argc, char* argv[]) 
{
    bool trans_A = false;
    bool trans_B = false;

    cudaSetDevice(0);

    if (argc != 2) 
    {
        printf("./exe N\n");
        exit(1);
    }
    int N = atoi(argv[1]);
    srand(time(0));
	// prepare data
	float *A = (float*)malloc(N * N * sizeof(float));
	float *B = (float*)malloc(N * N * sizeof(float));
	for (int i = 0; i < N * N; i ++) 
    {
        double x = (double)rand() / RAND_MAX;
        A[i] = (x > 0.5) ? 1 : -1;
        x = (double)rand() / RAND_MAX;
        B[i] = (x > 0.5) ? 1 : -1;
    }

	// copy to cuda
	float *fA, *fB, *fC;
    unsigned *uC;
    ullong *ullC;
	cudaMalloc(&fA, N * N * sizeof(float));
	cudaMalloc(&fB, N * N * sizeof(float));
	cudaMalloc(&fC, N * N * sizeof(float));
	cudaMalloc(&uC, N * N * sizeof(unsigned));
	cudaMalloc(&ullC, N * N * sizeof(unsigned long long));
	cudaMemcpy(fA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fB, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    //============================================= XNOR
    unsigned int *Aconc, *Bconc;
    cudaMalloc(&Aconc, N * N);
    cudaMalloc(&Bconc, N * N);
    cudaMemset(fC, 0, N * N * sizeof(float));
    int block = 64, grid = N * N / (block * 32)  + 1;
    int grid1 = N / block + 1;
    dim3 blockDim(16, 16);
    dim3 gridDim(N / 16 + 1, N / 16 + 1);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        concatenate_rows_kernel<<<grid, block>>>(fA, Aconc, N * N / 32);
        concatenate_cols_kernel<<<grid1, block>>>(fB, Bconc, N, N);
        xnor_gemm<<<gridDim, blockDim>>>(Aconc, Bconc, fC, N, N / 32, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double baseline_time = (milliseconds*1e3)/double(TEST_TIMES);
    //----------------------- 

    cudaFree(Aconc);
    cudaFree(Bconc);
    float* result_xnor = (float*)malloc(N * N * sizeof(float));
    cudaMemcpy(result_xnor, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    //============================================= CUBLAS
    cudaMemset(fC, 0, N * N * sizeof(int));
    cublasOperation_t cublas_trans_A = trans_A?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t cublas_trans_B = trans_B?CUBLAS_OP_T:CUBLAS_OP_N;
    __half* hfA = NULL; 
    cudaMalloc(&hfA, N*N*sizeof(__half));
    cudaMemset(hfA, 0, N*N*sizeof(__half));
    __half* hfB = NULL;
    cudaMalloc(&hfB, N*N*sizeof(__half));
    cudaMemset(hfB, 0, N*N*sizeof(__half));
    __half* hfC = NULL;
    cudaMalloc(&hfC, N*N*sizeof(__half));
    cudaMemset(hfC, 0, N*N*sizeof(__half));


    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0, beta = 0.0;
    __half hf_alpha = __float2half(alpha);
    __half hf_beta = __float2half(beta);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cublas use column-major
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        float_to_half<<<N*N/256,256>>>(fA, hfA, N*N);
        float_to_half<<<N*N/256,256>>>(fB, hfB, N*N);
        //cublasSgemm(handle, cublas_trans_A, cublas_trans_B, N, N, N,
        //&alpha, fB, N, fA, N, &beta, fC, N);
        cublasHgemm(handle, cublas_trans_A, cublas_trans_B, N, N, N,
                &hf_alpha, hfB, N, hfA, N, &hf_beta, hfC, N);
        half_to_float<<<N*N/256,256>>>(hfC, fC, N*N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double cublas_time = (milliseconds*1e3)/double(TEST_TIMES);
    //----------------------- 

    float* result_cublas = (float*)malloc(N * N * sizeof(float));
    cudaMemcpy(result_cublas, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    
    //============================================= BSTC-32
    cudaMemset(fC, 0, N * N * sizeof(float));
    
    unsigned *tA, *tB;
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));
    dim3 bmmDim(N/32, N/32);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        ToBit32Row<float><<<dim3(N/32,N/32), 32>>>(fB, tB, N, N);
        ToBit32Col<float><<<dim3(N/32,N/32), 32>>>(fA, tA, N, N);
        BMM32_Arow_Brow<float><<<dim3(N/32,N/32), 32>>>(tA, tB, fC, N, N/32, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bblas_time = (milliseconds*1e3)/double(TEST_TIMES);

    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 

    float* result_bblas = (float*)malloc(N * N * sizeof(float));
    cudaMemcpy(result_bblas, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    //============================================= BBLAS-64
    cudaMemset(fC, 0, N * N * sizeof(float));

    ullong *llA, *llB;
	cudaMalloc(&llA, N * N/64 * sizeof(ullong));
	cudaMalloc(&llB, N * N/64 * sizeof(ullong));

    dim3 bmm64Dim(N/64, N/64);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        ToBit64Row<float><<<dim3(N/64,N/32), 32>>>(fB, llB, N, N);
        ToBit64Col<float><<<dim3(N/32,N/64), 32>>>(fA, llA, N, N);
        BMM64_Arow_Brow<float><<<dim3(N/64,N/64), 32>>>(llA, llB, fC, N, N/64, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double b64blas_time = (milliseconds*1e3)/double(TEST_TIMES);
    //----------------------- 
    float* result_b64blas = (float*)malloc(N * N * sizeof(float));
    cudaMemcpy(result_b64blas, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(llA);
    cudaFree(llB);

    //============================================= BSTC-32-Small
    cudaMemset(fC, 0, N * N * sizeof(float));
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        ToBit32Row<float><<<dim3(N/32,N/32), 32>>>(fB, tB, N, N);
        ToBit32Col<float><<<dim3(N/32,N/32), 32>>>(fA, tA, N, N);
        BMM32_MT_M_S<<<dim3(N/32,N/32), 1024>>>(tA, tB, fC, N, N/32, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmm32s_time = (milliseconds*1e3)/double(TEST_TIMES);
    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 
    float* result_bmm32s = (float*)malloc(N * N * sizeof(float));
    cudaMemcpy(result_bmm32s, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    //============================================= BSTC-64-Small
    cudaMemset(fC, 0, N * N * sizeof(float));
	cudaMalloc(&llA, N * N/64 * sizeof(ullong));
	cudaMalloc(&llB, N * N/64 * sizeof(ullong));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        ToBit64Row<float><<<dim3(N/64,N/32), 32>>>(fB, llB, N, N);
        ToBit64Col<float><<<dim3(N/32,N/64), 32>>>(fA, llA, N, N);
        BMM64_MT_M_S<<<dim3(N/64,N/64), 1024>>>(llA, llB, fC, N, N/64, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmm64s_time = (milliseconds*1e3)/double(TEST_TIMES);
    //----------------------- 
    float* result_bmm64s = (float*)malloc(N * N * sizeof(float));
    cudaMemcpy(result_bmm64s, fC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(llA);
    cudaFree(llB);

    //============================================= BSTC-32-Bin
    cudaMemset(uC, 0, N * N * sizeof(unsigned));
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));
    ToBit32Row<float><<<dim3(N/32,N/32), 32>>>(fB, tB, N, N);
    ToBit32Col<float><<<dim3(N/32,N/32), 32>>>(fA, tA, N, N);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMM32_BIN<<<dim3(N/32,N/32), 32>>>(tA, tB, uC, N, N, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmm32_bin_time = (milliseconds*1e3)/double(TEST_TIMES);
    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 
    unsigned* result_bmm32_bin = (unsigned*)malloc(N * N * sizeof(unsigned));
    cudaMemcpy(result_bmm32_bin, uC, N * N * sizeof(unsigned), cudaMemcpyDeviceToHost);

    //============================================= BSTC-64-Bin
    cudaMemset(ullC, 0, N * N * sizeof(ullong));
	cudaMalloc(&llA, N * N/64 * sizeof(ullong));
	cudaMalloc(&llB, N * N/64 * sizeof(ullong));
    ToBit64Row<float><<<dim3(N/64,N/32), 32>>>(fB, llB, N, N);
    ToBit64Col<float><<<dim3(N/32,N/64), 32>>>(fA, llA, N, N);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMM64_BIN<<<dim3(N/64,N/64), 32>>>(llA, llB, ullC, N, N, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmm64_bin_time = (milliseconds*1e3)/double(TEST_TIMES);
    //----------------------- 
    ullong* result_bmm64_bin = (ullong*)malloc(N * N * sizeof(ullong));
    cudaMemcpy(result_bmm64_bin, ullC, N * N * sizeof(ullong), cudaMemcpyDeviceToHost);
    cudaFree(llA);
    cudaFree(llB);

    //============================================= BSTC-32-Small-Bin
    cudaMemset(uC, 0, N * N * sizeof(unsigned));
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));
    ToBit32Row<float><<<dim3(N/32,N/32), 32>>>(fB, tB, N, N);
    ToBit32Col<float><<<dim3(N/32,N/32), 32>>>(fA, tA, N, N);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMM32S_BIN<<<dim3(N/32,N/32), 1024>>>(tA, tB, uC, N, N, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmm32s_bin_time = (milliseconds*1e3)/double(TEST_TIMES);
    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 
    unsigned* result_bmm32s_bin = (unsigned*)malloc(N * N * sizeof(unsigned));
    cudaMemcpy(result_bmm32s_bin, uC, N * N * sizeof(unsigned), cudaMemcpyDeviceToHost);



    //============================================= BSTC-64-Small-Bin
    cudaMemset(ullC, 0, N * N * sizeof(ullong));
	cudaMalloc(&llA, N * N/64 * sizeof(ullong));
	cudaMalloc(&llB, N * N/64 * sizeof(ullong));
    ToBit64Row<float><<<dim3(N/64,N/32), 32>>>(fB, llB, N, N);
    ToBit64Col<float><<<dim3(N/32,N/64), 32>>>(fA, llA, N, N);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMM64S_BIN<<<dim3(N/64,N/64), 1024>>>(llA, llB, ullC, N, N/64, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmm64s_bin_time = (milliseconds*1e3)/double(TEST_TIMES);
    //----------------------- 
    ullong* result_bmm64s_bin = (ullong*)malloc(N * N * sizeof(ullong));
    cudaMemcpy(result_bmm64s_bin, ullC, N * N * sizeof(ullong), cudaMemcpyDeviceToHost);
    cudaFree(llA);
    cudaFree(llB);

    //============================================= TensorCore
    int* tC = NULL;
    cudaMalloc(&tC, N * N * sizeof(int));
    cudaMemset(tC, 0, N * N * sizeof(int));
    
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));
    const unsigned BMMA_M = 4;
    const unsigned BMMA_N = 4;
    dim3 tensorcoreBlk(32, BMMA_M, BMMA_N);
    dim3 tensorcoreDim(N/(8*BMMA_M), N/(8*BMMA_N));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMMA_toBit32Row<float><<<dim3(N,N/32), 32>>>(fA, tA, N, N);
        BMMA_toBit32Col<float><<<dim3(N/32,N), 32>>>(fB, tB, N, N);
        BMMApipe<BMMA_M,BMMA_N><<<tensorcoreDim, tensorcoreBlk>>>(tA, tB, tC, N, N, N/128);
        //BMMA<BMMA_M,BMMA_N><<<tensorcoreDim, tensorcoreBlk>>>(tA, tB, tC, N, N, N/128);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double tensorcore_time = (milliseconds*1e3)/double(TEST_TIMES);

    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 

    int* result_tensorcore = (int*)malloc(N * N * sizeof(int));
    memset(result_tensorcore, 0, N*N*sizeof(int));
    cudaMemcpy(result_tensorcore, tC, N * N * sizeof(int), cudaMemcpyDeviceToHost);


    //============================================= TensorCore_Small
    cudaMemset(tC, 0, N * N * sizeof(int));
    
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));
    const unsigned BMMAS_M = 4;

    dim3 tensorcoreSBlk(32, 2);
    dim3 tensorcoreSDim(N/16, N/8);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMMA_toBit32Row<float><<<dim3(N,N/32), 32>>>(fA, tA, N, N);
        BMMA_toBit32Col<float><<<dim3(N/32,N), 32>>>(fB, tB, N, N);
        BMMAS<BMMAS_M><<<tensorcoreSDim, tensorcoreSBlk>>>(tA, tB, tC, N, N, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double tensorcore_s_time = (milliseconds*1e3)/double(TEST_TIMES);

    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 

    int* result_tensorcore_s = (int*)malloc(N * N * sizeof(int));
    cudaMemcpy(result_tensorcore_s, tC, N * N * sizeof(int), cudaMemcpyDeviceToHost);


    //============================================= TensorCore_Bin
    cudaMemset(uC, 0, N * N * sizeof(unsigned));
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));

    //const unsigned BMMA_M = 4;
    //const unsigned BMMA_N = 4;
    //dim3 tensorcoreBlk(32, BMMA_M, BMMA_N);
    //dim3 tensorcoreDim(N/(8*BMMA_M), N/(8*BMMA_N));
    BMMA_toBit32Row<float><<<dim3(N,N/32), 32>>>(fA, tA, N, N);
    BMMA_toBit32Col<float><<<dim3(N/32,N), 32>>>(fB, tB, N, N);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMMA_bin<BMMA_M,BMMA_N><<<tensorcoreDim, tensorcoreBlk>>>(tA, tB, uC, N, N, N/128);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmma_bin_time = (milliseconds*1e3)/double(TEST_TIMES);
    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 
    unsigned* result_bmma_bin = (unsigned*)malloc(N * N * sizeof(unsigned));
    cudaMemcpy(result_bmma_bin, uC, N * N * sizeof(unsigned), cudaMemcpyDeviceToHost);



    //============================================= TensorCore_Small_Bin
    cudaMemset(uC, 0, N * N * sizeof(unsigned));
    
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));
    
    BMMA_toBit32Row<float><<<dim3(N,N/32), 32>>>(fA, tA, N, N);
    BMMA_toBit32Col<float><<<dim3(N/32,N), 32>>>(fB, tB, N, N);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMMAS_bin<BMMAS_M><<<tensorcoreSDim, tensorcoreSBlk>>>(tA, tB, uC, N, N, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmmas_bin_time = (milliseconds*1e3)/double(TEST_TIMES);

    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 

    int* result_bmmas_bin = (int*)malloc(N * N * sizeof(int));
    cudaMemcpy(result_bmmas_bin, uC, N * N * sizeof(int), cudaMemcpyDeviceToHost);


    //============================================= TensorCore_Small_New_format
    cudaMemset(tC, 0, N * N * sizeof(int));
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));

    dim3 tensorcoreSNBlk(32, 2);
    dim3 tensorcoreSNDim(N/16, N/8);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMMA_toBit32Row_new<float><<<dim3(N/8,N/128), dim3(32,8,4)>>>(fA, tA, N, N);
        BMMA_toBit32Col_new<float><<<dim3(N/128,N/8), dim3(32,4,8)>>>(fB, tB, N, N);
        BMMAS_new<<<tensorcoreSNDim, tensorcoreSNBlk>>>(tA, tB, tC, N, N, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmma_sn_time = (milliseconds*1e3)/double(TEST_TIMES);

    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 

    int* result_bmma_sn = (int*)malloc(N * N * sizeof(int));
    cudaMemcpy(result_bmma_sn, tC, N * N * sizeof(int), cudaMemcpyDeviceToHost);


    //============================================= TensorCore_Small_Bin_New_Format
    cudaMemset(uC, 0, N * N * sizeof(unsigned));
    
	cudaMalloc(&tA, N * N/32 * sizeof(unsigned));
	cudaMalloc(&tB, N * N/32 * sizeof(unsigned));
    
    BMMA_toBit32Row_new<float><<<dim3(N/8,N/128), dim3(32,8,4)>>>(fA, tA, N, N);
    BMMA_toBit32Col_new<float><<<dim3(N/128,N/8), dim3(32,4,8)>>>(fB, tB, N, N);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //----------------------- 
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        BMMAS_bin_new<<<tensorcoreSDim, tensorcoreSBlk>>>(tA, tB, uC, N, N, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmmasn_bin_time = (milliseconds*1e3)/double(TEST_TIMES);

    cudaFree(tA);
    cudaFree(tB);
    //----------------------- 

    int* result_bmmasn_bin = (int*)malloc(N * N * sizeof(int));
    cudaMemcpy(result_bmmasn_bin, uC, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    //============================================= CHECK RESULT
    printf("XNOR success: %d\n", check_result(result_cublas, result_xnor, N));
    printf("CUBLAS success: %d\n", check_result(result_cublas, result_cublas, N));
    printf("BMM-32 success: %d\n", check_result(result_cublas, result_bblas, N));
    printf("BMM-64 success: %d\n", check_result(result_cublas, result_b64blas, N));
    printf("BMMS-32 success: %d\n", check_result(result_cublas, result_bmm32s, N));
    printf("BMMS-64 success: %d\n", check_result(result_cublas, result_bmm64s, N));
    printf("BMMA success: %d\n", check_result(result_cublas, result_tensorcore, N));
    printf("BMMAS success: %d\n", check_result(result_cublas, result_tensorcore_s, N));


    //for (int i=0; i<10; i++)
    //printf("bmm32_bin:%x,bmm32s_bin:%x,bmm64_bin:%llx,bmm64s_bin:%llx\n", result_bmm32_bin[i],
    //result_bmm32s_bin[i], result_bmm64_bin[i], result_bmm64s_bin[i]);

    printf("CUBLAS:%.3lf, BNN:%.3lf, BMM-32:%.3lf, BMM-64:%.3lf, BMMS-32:%.3lf, BMMS-64:%.3lf, BMM-32-Bin:%.3lf, BMM-64-Bin:%.3lf, BMMS-32-Bin:%.3lf, BMMS-64-Bin:%.3lf, BMMA:%.3lf, BMMAS:%.3lf, BMMA-Bin:%.3lf, BMMAS-Bin:%.3lf, BMMASN:%.3lf, BMMASN-Bin:%.3lf \n", 
            cublas_time, baseline_time, bblas_time, b64blas_time, bmm32s_time, bmm64s_time,
            bmm32_bin_time, bmm64_bin_time, bmm32s_bin_time, bmm64s_bin_time, tensorcore_time,
            tensorcore_s_time, bmma_bin_time, bmmas_bin_time, bmma_sn_time, bmmasn_bin_time);


    cudaFree(fA);
    cudaFree(fB);
    cudaFree(fC);
    cudaFree(uC);
    cudaFree(ullC);
    free(result_xnor);
    free(result_cublas);
    free(result_bblas);
    free(result_b64blas);

    free(result_tensorcore);
    free(result_tensorcore_s);
    free(result_bmma_bin);
    free(result_bmmas_bin);
    free(result_bmma_sn);
    free(result_bmmasn_bin);

}
