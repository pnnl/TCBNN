// ---------------------------------------------------------------------------
// File: utility.h
// Utility functions.
// ---------------------------------------------------------------------------
// See our arXiv paper for detail: https://arxiv.org/abs/2006.16578
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/TCBNN
// PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850
// BSD Lincese.
// Richland, 99352, WA, USA. June-30-2020.
// ---------------------------------------------------------------------------



#ifndef UTILITY_H
#define UTILITY_H

#include <sys/time.h>
#include <stdio.h>

#define NEWFMT

//================ Type Definition ===============
typedef unsigned char uin8;
typedef unsigned short uin16;
typedef unsigned int uin32;
typedef unsigned long long uin64;
typedef uint4 uin128;

//================ Macro Definition ===============
//how many bits steps to go
#define STEP4(X) (((X)+3)>>2) 
#define STEP8(X) (((X)+7)>>3) 
#define STEP16(X) (((X)+15)>>4) 
#define STEP32(X) (((X)+31)>>5) 
#define STEP64(X) (((X)+63)>>6) 
#define STEP128(X) (((X)+127)>>7) 
//total bits covers after padding
#define PAD4(X) (STEP4(X)<<2)
#define PAD8(X) (STEP8(X)<<3)
#define PAD16(X) (STEP16(X)<<4)
#define PAD32(X) (STEP32(X)<<5)
#define PAD64(X) (STEP64(X)<<6)
#define PAD128(X) (STEP128(X)<<7)
//get lane id
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid)); 
//get warp id
#define GET_WARPID unsigned warpid; asm("mov.u32 %0, %%warpid;":"=r"(warpid)); 

//flip last b bits of n
#define FLIPBITS(n,b) ((n)^((1u<<(b))-1))


//================ ERROR CHECKING ===============
//check CUDA API calls
#define CUDA_SAFE_CALL( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if ( cudaSuccess != err )
    {   
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);    
    }       
    return;
}

//check last CUDA kernel function
#define CUDA_CHECK_KERNEL()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        exit(-1);
    }
    return;
}

//check null pointer
#define CHECK_NULL_POINTER(X) __checkNullPointer( __FILE__, __LINE__, (void**)&(X))
inline void __checkNullPointer( const char *file, const int line, void** ptr)
{
    if ((*ptr) == NULL)
    {
        fprintf( stderr, "Error: NULL pointer at %s:%i.\n");
        exit(-1);
    }
}

//check nccl error
#define CHECK_NCCL(cmd) do { \
        ncclResult_t r=cmd;\
        if (r!=ncclSuccess){\
                    printf("NCCL error %s: %d '%s'\n",__FILE__,__LINE__,ncclGetErrorString(r));\
                    exit(EXIT_FAILURE);\
                }\
} while(0) \



//================ Allocation and Free ===============
//CPU host allocation
#define SAFE_ALOC_HOST(X,Y) CUDA_SAFE_CALL(cudaMallocHost((void**)&(X),(Y)));
//GPU device allocation
#define SAFE_ALOC_GPU(X,Y) CUDA_SAFE_CALL(cudaMalloc((void**)&(X),(Y)));
//CPU host free
#define SAFE_FREE_HOST(X) if ((X) != NULL) { \
               CUDA_SAFE_CALL( cudaFreeHost((X))); \
               (X) = NULL;}
//GPU device free
#define SAFE_FREE_GPU(X) if ((X) != NULL) { \
               CUDA_SAFE_CALL( cudaFree((X))); \
               (X) = NULL;}

//================ Timer ===============
double get_cpu_timer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    //get current timestamp in milliseconds
    return (double)tp.tv_sec * 1e3 + (double)tp.tv_usec * 1e-3;
}
// CPU Timer object definition
typedef struct CPU_Timer
{
    CPU_Timer() { start = stop = 0.0; }
    void start_timer() { start = get_cpu_timer(); }
    void stop_timer() { stop = get_cpu_timer(); }
    double measure() { double millisconds = stop - start; return millisconds; }
    double start;
    double stop;
} cpu_timer;
// GPU Timer object definition
typedef struct GPU_Timer
{
    GPU_Timer()
    {
        CUDA_SAFE_CALL( cudaEventCreate(&this->start) );
        CUDA_SAFE_CALL( cudaEventCreate(&this->stop) );
    }
    void start_timer() { CUDA_SAFE_CALL( cudaEventRecord(this->start) ); }
    void stop_timer() { CUDA_SAFE_CALL( cudaEventRecord(this->stop) ); }
    double measure()
    {
        CUDA_SAFE_CALL( cudaEventSynchronize(this->stop) );
        float millisconds = 0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&millisconds, this->start, this->stop) ); 
        return (double)millisconds;
    }
    cudaEvent_t start;
    cudaEvent_t stop;
} gpu_timer;



//Start Timer
#define START_TIMER cudaEvent_t start, stop;\
    cudaEventCreate(&start);\
    cudaEventCreate(&stop);\
    cudaEventRecord(start);

//Stop Timer
#define STOP_TIMER cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    float milliseconds = 0; \
    cudaEventElapsedTime(&milliseconds, start, stop); \
/*printf("\n============================\n"); \*/
/*printf("Inference_Time: %.3f ms.",milliseconds);\*/
/*printf("\n============================\n");*/




//start gpu kernel timer
#define SET_KERNEL_TIMER uin64 t0 = clock64();
//stop gpu kernel timer
/*#define TICK_KERNEL_TIMER(X)  grid.sync(); \*/
/*if (threadIdx.x==0 && blockIdx.x == 0) printf("Layer-%s takes %lld cycles.\n", (X)->name, clock64()-t0);*/
#define TICK_KERNEL_TIMER(X) if (threadIdx.x==0 && blockIdx.x == 0) printf("Layer-%s takes %lld cycles.\n", (X)->name, clock64()-t0);


//================ Utility Functions ===============
//load weights from file
void launch_array(FILE* cf, float* array, unsigned array_size)
{
    if (cf == NULL)
    {
        fprintf(stderr, "NULL pointer to the network configuration file.\n");
        exit(1);
    }
    for (int i=0; i<array_size; i++) fscanf(cf, "%f", &array[i]); 
}
//validation
void validate_prediction(float* prediction, unsigned* labels, const unsigned categories, const unsigned batch)
{
    printf("======Label======\n");
    for(int i=0; i<batch; i++) printf("%u ",labels[i]);
    printf("\n=====Predict=======\n");
    int corrects = 0;
    int corrects_top5 = 0;
    for (int i=0; i<batch; i++)
    {
        int pos = 0;
        float max = prediction[i*categories];
        float max1 = prediction[i*categories];
        float max2 = prediction[i*categories];
        float max3 = prediction[i*categories];
        float max4 = prediction[i*categories];
        int pos1 = 0; int pos2 = 0; int pos3 = 0; int pos4 = 0;

        for (int j=0; j<categories; j++)
        {
            float val = prediction[i*categories+j];
            if (val>max) { max = val; pos = j; }
            else if (val > max1) { max1 = val; pos1 = j; }
            else if (val > max2) { max2 = val; pos2 = j; }
            else if (val > max3) { max3 = val; pos3 = j; }
            else if (val > max4) { max4 = val; pos4 = j; }
        }
        printf("%d ", pos);
        if (pos == labels[i]) corrects += 1;
        if (pos == labels[i] || pos1 == labels[i] || pos2 == labels[i]
                || pos3 == labels[i] || pos4 == labels[i] ) { corrects_top5 += 1; }
    }
    printf("\n=====Batch Accuracy:%d=======\n", batch);
    printf("Top-1:%f\%\n",float(corrects)/batch*100);
    printf("Top-5:%f\%\n",float(corrects_top5)/batch*100);
}

//================ Efficient Store ===============
//store 64 bits
template <typename T>
__device__ __inline__ void store64(const void* addr, T a, T b)
{
    *((float2*)addr) = make_float2(*(float*)(&a),*(float*)(&b));
}
//store 128 bits
template <typename T>
__device__ __inline__ void store128(const void* addr, T a, T b, T c, T d)
{
    *((float4*)addr) = make_float4(*(float*)(&a),*(float*)(&b),*(float*)(&c),*(float*)(&d));
}


#endif

