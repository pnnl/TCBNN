#include <stdio.h>
#include <cuda.h>

using namespace std;

#define N 2048*2048
#define M (N/8)
#define TEST_TIMES 10

__global__ void getmaxmin_kernel(float *a, float *c, int n)
{
    int tid = threadIdx.x;
    __shared__ float maxa[1024];
    __shared__ float mina[1024];

    maxa[tid] = a[tid];
    mina[tid] = a[tid];
    __syncthreads();
    for (int i=tid; i<n; i+=1024)
    {
        if (maxa[tid] < a[i]) maxa[tid] = a[i];
        if (mina[tid] > a[i]) mina[tid] = a[i];
    }
    __syncthreads();
    for (int k=512; k>0; k>>=1)
    {
        if (tid < k)
        {
            if (maxa[tid] < maxa[tid+k]) maxa[tid] = maxa[tid+k];
            if (mina[tid] > mina[tid+k]) mina[tid] = mina[tid+k];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        c[0] = maxa[0];
        c[1] = mina[0];
    }
}

__global__ void fp32toint4_kernel(float* a, unsigned* b, float* c, int m)
{ 
    float maxv = c[0];
    float minv = c[1];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < m) 
    {
        unsigned outv = 0;
#pragma unroll
        for (int i=0; i<7; i++)
        {
            outv = outv | ((static_cast<int>((a[id*4+i] - minv)*15.0/(maxv-minv)))<<(28-i*4));
        }
        b[id] = outv; 
    }
}


int main(int argc, char* argv[]) 
{
    cudaSetDevice(0);
    srand(time(0));
	float *A = (float*)malloc(N * sizeof(float));
    unsigned *B = (unsigned*)malloc(M*sizeof(unsigned));
    cudaEvent_t start, stop;
 	
    for (int i=0; i<N; i++) 
    {
        //A[i] = float((double)rand() / RAND_MAX);
        A[i] = (float)rand() - float(0.5*RAND_MAX);
    }
    for (int i=0; i<M; i++) B[i] = 0;

	float *fA = NULL;
    unsigned *fB = NULL;
    float *fC = NULL;
	cudaMalloc(&fA, N * sizeof(float));
	cudaMalloc(&fB, M * sizeof(unsigned));
	cudaMalloc(&fC, 2 * sizeof(float));
	cudaMemcpy(fA, A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fB, B, M * sizeof(unsigned), cudaMemcpyHostToDevice);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++)
    {
        getmaxmin_kernel<<<1,1024>>>(fA,fC,N);
        fp32toint4_kernel<<<dim3((M+1023)/1024), 1024>>>(fA, fB, fC, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double baseline_time = (milliseconds*1e3)/double(TEST_TIMES);

    cudaMemcpy(B, fB, M * sizeof(unsigned), cudaMemcpyDeviceToHost);
    printf("\n==========\n");

    for (int i=0; i<N; i++)
    {
        int bid = i / 8;
        int sid = i % 8;
        int val = ((B[bid]>>((7-sid)*4)) & 15) - 8;
        printf("%d ",val);
    }
    printf("\nBaseline time is:%.3lf us\n",baseline_time);

    cudaFree(fA);
    cudaFree(fB);
    free(A);
    free(B);
 
}
