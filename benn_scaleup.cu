// ---------------------------------------------------------------------------
// File: benn_scaleup.cu
// A ResNet-18 BNN for ImageNet on multi-GPUs in a single node via NCCL.
// ---------------------------------------------------------------------------
// See our arXiv paper for detail: https://arxiv.org/abs/2006.16578
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/TCBNN
// PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850
// BSD Lincese.
// Richland, 99352, WA, USA. June-30-2020.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <nccl.h>
#include <omp.h>
#include "utility.h"
#include "param.h"
#include "kernel.cuh"
#include "data.h"

using namespace cooperative_groups;
using namespace std;


#ifdef NEWFMT

__global__ void resnet128(
        InConv128LayerParam* bconv1, 
        Conv128LayerParam* l1b1c1, 
        Conv128LayerParam* l1b1c2,
        Conv128LayerParam* l1b2c1, 
        Conv128LayerParam* l1b2c2,
        Conv128LayerParam* l2b1c1, 
        Conv128LayerParam* l2b1c2,
        Conv128LayerParam* l2b2c1, 
        Conv128LayerParam* l2b2c2,
        Conv128LayerParam* l3b1c1, 
        Conv128LayerParam* l3b1c2,
        Conv128LayerParam* l3b2c1, 
        Conv128LayerParam* l3b2c2,
        Conv128LayerParam* l4b1c1, 
        Conv128LayerParam* l4b1c2,
        Conv128LayerParam* l4b2c1, 
        Conv128LayerParam* l4b2c2,
        Fc128LayerParam* bfc1, 
        Out128LayerParam* bout)
{
    //SET_KERNEL_TIMER;
    grid_group grid = this_grid();
    //========= Conv1 ============
    InConv128LayerFMT(bconv1);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv1);
    //========= L1B1 ============
    Conv128LayerFMT(l1b1c1);
    grid.sync();
    //TICK_KERNEL_TIMER(l1b1c1);
    Conv128LayerFMT(l1b1c2);
    grid.sync();
    //TICK_KERNEL_TIMER(l1b1c2);
    //========= L1B2 ============
    Conv128LayerFMT(l1b2c1);
    grid.sync();
    //TICK_KERNEL_TIMER(l1b2c1);
    Conv128LayerFMT(l1b2c2);
    grid.sync();
    //TICK_KERNEL_TIMER(l1b2c2);
    //========= L2B1 ============
    Conv128LayerFMT(l2b1c1);
    grid.sync();
    //TICK_KERNEL_TIMER(l2b1c1);
    Conv128LayerFMT(l2b1c2);
    grid.sync();
    //TICK_KERNEL_TIMER(l2b1c2);
    //========= L2B2 ============
    Conv128LayerFMT(l2b2c1);
    grid.sync();
    //TICK_KERNEL_TIMER(l2b2c1);
    Conv128LayerFMT(l2b2c2);
    grid.sync();
    //TICK_KERNEL_TIMER(l2b2c2);
    //========= L3B1 ============
    Conv128LayerFMT(l3b1c1);
    grid.sync();
    //TICK_KERNEL_TIMER(l3b1c1);
    Conv128LayerFMT(l3b1c2);
    grid.sync();
    //TICK_KERNEL_TIMER(l3b1c2);
    //========= L3B2 ============
    Conv128LayerFMT(l3b2c1);
    grid.sync();
    //TICK_KERNEL_TIMER(l3b2c1);
    Conv128LayerFMT(l3b2c2);
    grid.sync();
    //TICK_KERNEL_TIMER(l3b2c2);
    //========= L4B1 ============
    Conv128LayerFMT(l4b1c1);
    grid.sync();
    //TICK_KERNEL_TIMER(l4b1c1);
    Conv128LayerFMT(l4b1c2);
    grid.sync();
    //TICK_KERNEL_TIMER(l4b1c2);
    //========= L4B2 ============
    Conv128LayerFMT(l4b2c1);
    grid.sync();
    //TICK_KERNEL_TIMER(l4b2c1);
    Conv128LayerFMT(l4b2c2);
    grid.sync();
    //TICK_KERNEL_TIMER(l4b2c2);
    //========= Fc1 ============
    Fc128LayerFMT(bfc1);
    grid.sync();
    //TICK_KERNEL_TIMER(bfc1);
    //========== Output ===========
    Out128LayerFMT(bout);
    //grid.sync();
    //TICK_KERNEL_TIMER(bout);
}

#else

__global__ void resnet128(
        InConv128LayerParam* bconv1, 
        Conv128LayerParam* l1b1c1, 
        Conv128LayerParam* l1b1c2,
        Conv128LayerParam* l1b2c1, 
        Conv128LayerParam* l1b2c2,
        Conv128LayerParam* l2b1c1, 
        Conv128LayerParam* l2b1c2,
        Conv128LayerParam* l2b2c1, 
        Conv128LayerParam* l2b2c2,
        Conv128LayerParam* l3b1c1, 
        Conv128LayerParam* l3b1c2,
        Conv128LayerParam* l3b2c1, 
        Conv128LayerParam* l3b2c2,
        Conv128LayerParam* l4b1c1, 
        Conv128LayerParam* l4b1c2,
        Conv128LayerParam* l4b2c1, 
        Conv128LayerParam* l4b2c2,
        Fc128LayerParam* bfc1, 
        Out128LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Conv1 ============
    InConv128Layer(bconv1);
    grid.sync();
    //========= L1B1 ============
    Conv128Layer(l1b1c1);
    grid.sync();
    Conv128Layer(l1b1c2);
    grid.sync();
    //========= L1B2 ============
    Conv128Layer(l1b2c1);
    grid.sync();
    Conv128Layer(l1b2c2);
    grid.sync();
    //========= L2B1 ============
    Conv128Layer(l2b1c1);
    grid.sync();
    Conv128Layer(l2b1c2);
    grid.sync();
    //========= L2B2 ============
    Conv128Layer(l2b2c1);
    grid.sync();
    Conv128Layer(l2b2c2);
    grid.sync();
    //========= L3B1 ============
    Conv128Layer(l3b1c1);
    grid.sync();
    Conv128Layer(l3b1c2);
    grid.sync();
    //========= L3B2 ============
    Conv128Layer(l3b2c1);
    grid.sync();
    Conv128Layer(l3b2c2);
    grid.sync();
    //========= L4B1 ============
    Conv128Layer(l4b1c1);
    grid.sync();
    Conv128Layer(l4b1c2);
    grid.sync();
    //========= L4B2 ============
    Conv128Layer(l4b2c1);
    grid.sync();
    Conv128Layer(l4b2c2);
    grid.sync();
    //========= Fc1 ============
    Fc128Layer(bfc1);
    grid.sync();
    //========== Output ===========
    Out128Layer(bout);
}

#endif

void run(unsigned batch, float* pcomp_time, float* pcomm_time)
{
    const int n_gpu = 2;
    float comp_times[8] = {0};
    float comm_times[8] = {0};
    //printf("EBNN with ResNet-18 on ImageNet for Scaling-Up using %d GPUs!\n", n_gpu);

    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm[n_gpu];

    //const unsigned batch = 64;
    const unsigned output_size = 1000;
    const unsigned image_height = 224;
    const unsigned image_width = 224;
    const unsigned image_channel = 3;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);




//Start OpenMP
#pragma omp parallel num_threads (n_gpu) shared (comm, id, batch, images, image_labels)
{
    int i_gpu = omp_get_thread_num();
    CUDA_SAFE_CALL(cudaSetDevice(i_gpu));
    cudaEvent_t comp_start, comp_stop, comm_start, comm_stop;
    CUDA_SAFE_CALL(cudaEventCreate(&comp_start));
    CUDA_SAFE_CALL(cudaEventCreate(&comp_stop));
    CUDA_SAFE_CALL(cudaEventCreate(&comm_start));
    CUDA_SAFE_CALL(cudaEventCreate(&comm_stop));
    CHECK_NCCL(ncclCommInitRank(&comm[i_gpu], n_gpu, id, i_gpu));

    int dev = i_gpu;
    //================ Get Weight =================
    FILE* config_file = fopen("./resnet_imagenet.csv","r");

    //================ Set Network =================
    //Layer-0
    InConv128LayerParam* bconv1 = new InConv128LayerParam("Conv1", image_height, image_width, 
            7, 7, 3, 64, batch,4,4,true,1,1,true);//save residual 
    InConv128LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);

    //Layer-1, basic-block-1, conv1
    Conv128LayerParam* l1b1c1 = new Conv128LayerParam("L1B1C1", bconv1->output_height, 
            bconv1->output_width, 3, 3, 64, 64, batch);
    Conv128LayerParam* l1b1c1_gpu = l1b1c1->initialize(config_file, bconv1->get_output_gpu());

    //Layer-1, basic-block-1, conv2
    Conv128LayerParam* l1b1c2 = new Conv128LayerParam("L1B1C2", l1b1c1->output_height, 
            l1b1c1->output_width, 3, 3, 64, 64, batch,1,1,true,1,1,false,true,true,64);
    Conv128LayerParam* l1b1c2_gpu = l1b1c2->initialize(config_file, l1b1c1->get_output_gpu(),
            bconv1->get_output_residual_gpu());

    //Layer-1, basic-block-2, conv1
    Conv128LayerParam* l1b2c1 = new Conv128LayerParam("L1B2C1", l1b1c2->output_height, 
            l1b1c2->output_width, 3, 3, 64, 64, batch);
    Conv128LayerParam* l1b2c1_gpu = l1b2c1->initialize(config_file, l1b1c2->get_output_gpu());

    //Layer-1, basic-block-2, conv2
    Conv128LayerParam* l1b2c2 = new Conv128LayerParam("L1B2C2", l1b2c1->output_height, 
            l1b2c1->output_width, 3, 3, 64, 64, batch,1,1,true,1,1,false,true,true,128);
    Conv128LayerParam* l1b2c2_gpu = l1b2c2->initialize(config_file, l1b2c1->get_output_gpu(),
            l1b1c2->get_output_residual_gpu());

    //=============
    //Layer-2, basic-block-1, conv1
    Conv128LayerParam* l2b1c1 = new Conv128LayerParam("L2B1C1", l1b2c2->output_height, 
            l1b2c2->output_width, 3, 3, 64, 128, batch, 2, 2);
    Conv128LayerParam* l2b1c1_gpu = l2b1c1->initialize(config_file, l1b2c2->get_output_gpu());

    //Layer-2, basic-block-1, conv2
    Conv128LayerParam* l2b1c2 = new Conv128LayerParam("L2B1C2", l2b1c1->output_height, 
            l2b1c1->output_width, 3, 3, 128, 128, batch,1,1,true,1,1,false,true,true,128,true);
    Conv128LayerParam* l2b1c2_gpu = l2b1c2->initialize(config_file, l2b1c1->get_output_gpu(),
            l1b2c2->get_output_residual_gpu());

    //Layer-2, basic-block-2, conv1
    Conv128LayerParam* l2b2c1 = new Conv128LayerParam("L2B2C1", l2b1c2->output_height, 
            l2b1c2->output_width, 3, 3, 128, 128, batch, 1, 1);
    Conv128LayerParam* l2b2c1_gpu = l2b2c1->initialize(config_file, l2b1c2->get_output_gpu());

    //Layer-2, basic-block-2, conv2
    Conv128LayerParam* l2b2c2 = new Conv128LayerParam("L2B2C2", l2b2c1->output_height, 
            l2b2c1->output_width, 3, 3, 128, 128, batch,1,1,true,1,1,false,true,true,128);
    Conv128LayerParam* l2b2c2_gpu = l2b2c2->initialize(config_file, l2b2c1->get_output_gpu(),
            l2b1c2->get_output_residual_gpu());

    //=============
    //Layer-3, basic-block-1, conv1
    Conv128LayerParam* l3b1c1 = new Conv128LayerParam("L3B1C1", l2b2c2->output_height, 
            l2b2c2->output_width, 3, 3, 128, 256, batch, 2, 2);
    Conv128LayerParam* l3b1c1_gpu = l3b1c1->initialize(config_file, l2b2c2->get_output_gpu());

    //Layer-3, basic-block-1, conv2
    Conv128LayerParam* l3b1c2 = new Conv128LayerParam("L3B1C2", l3b1c1->output_height, 
            l3b1c1->output_width, 3, 3, 256, 256, batch,1,1,true,1,1,false,true,true,128,true);
    Conv128LayerParam* l3b1c2_gpu = l3b1c2->initialize(config_file, l3b1c1->get_output_gpu(),
            l2b2c2->get_output_residual_gpu());

    //Layer-3, basic-block-2, conv1
    Conv128LayerParam* l3b2c1 = new Conv128LayerParam("L3B2C1", l3b1c2->output_height, 
            l3b1c2->output_width, 3, 3, 256, 256, batch, 1, 1);
    Conv128LayerParam* l3b2c1_gpu = l3b2c1->initialize(config_file, l3b1c2->get_output_gpu());

    //Layer-3, basic-block-2, conv2
    Conv128LayerParam* l3b2c2 = new Conv128LayerParam("L3B2C2", l3b2c1->output_height, 
            l3b2c1->output_width, 3, 3, 256, 256, batch,1,1,true,1,1,false,true,true,256);
    Conv128LayerParam* l3b2c2_gpu = l3b2c2->initialize(config_file, l3b2c1->get_output_gpu(),
            l3b1c2->get_output_residual_gpu());

    //=============
    //Layer-4, basic-block-1, conv1
    Conv128LayerParam* l4b1c1 = new Conv128LayerParam("L4B1C1", l3b2c2->output_height, 
            l3b2c2->output_width, 3, 3, 256, 512, batch, 2, 2);
    Conv128LayerParam* l4b1c1_gpu = l4b1c1->initialize(config_file, l3b2c2->get_output_gpu());

    //Layer-4, basic-block-1, conv2
    Conv128LayerParam* l4b1c2 = new Conv128LayerParam("L4B1C2", l4b1c1->output_height, 
            l4b1c1->output_width, 3, 3, 512, 512, batch,1,1,true,1,1,false,true,true,256,true);
    Conv128LayerParam* l4b1c2_gpu = l4b1c2->initialize(config_file, l4b1c1->get_output_gpu(),
            l3b2c2->get_output_residual_gpu());

    //Layer-4, basic-block-2, conv1
    Conv128LayerParam* l4b2c1 = new Conv128LayerParam("L4B2C1", l4b1c2->output_height, 
            l4b1c2->output_width, 3, 3, 512, 512, batch, 1, 1);
    Conv128LayerParam* l4b2c1_gpu = l4b2c1->initialize(config_file, l4b1c2->get_output_gpu());

    //Layer-4, basic-block-2, conv2
    Conv128LayerParam* l4b2c2 = new Conv128LayerParam("L4B2C2", l4b2c1->output_height, 
            l4b2c1->output_width, 3, 3, 512, 512, batch,1,1,true,1,1,true,false,true,512);
    Conv128LayerParam* l4b2c2_gpu = l4b2c2->initialize(config_file, l4b2c1->get_output_gpu(),
            l4b1c2->get_output_residual_gpu());

    //=============
    //Layer-5
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, (l4b2c2->output_height)
            *(l4b2c2->output_width)*512, 512); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, l4b2c2->get_output_gpu());
    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, 512, output_size);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc1->get_output_gpu());  

    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 512*sizeof(int)*32;
    cudaFuncSetAttribute(resnet128, cudaFuncAttributeMaxDynamicSharedMemorySize,shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, resnet128, numThreads, shared_memory);

    //cudaFuncSetAttribute(resnet128, cudaFuncAttributePreferredSharedMemoryCarveout,0);

    void* args[] = {&bconv1_gpu, 
        &l1b1c1_gpu, 
        &l1b1c2_gpu,
        &l1b2c1_gpu,
        &l1b2c2_gpu,
        &l2b1c1_gpu, 
        &l2b1c2_gpu,
        &l2b2c1_gpu,
        &l2b2c2_gpu,
        &l3b1c1_gpu, 
        &l3b1c2_gpu,
        &l3b2c1_gpu,
        &l3b2c2_gpu,
        &l4b1c1_gpu, 
        &l4b1c2_gpu,
        &l4b2c1_gpu,
        &l4b2c2_gpu,
        &bfc1_gpu,
        &bout_gpu};


    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    #pragma omp barrier
    //START_TIMER;
    
    cudaEventRecord(comp_start);

    cudaLaunchCooperativeKernel((void*)resnet128, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);
    
    cudaEventRecord(comp_stop);

    //if (i_gpu == 0) cudaMemset(bout->get_output_gpu(), 0, bout->output_bytes());
    //if (i_gpu == 1) cudaMemset(bout->get_output_gpu(), 0, bout->output_bytes());

    cudaEventRecord(comm_start);

    CHECK_NCCL( ncclReduce(bout->get_output_gpu(), bout->get_output_gpu(), 
                bout->output_size(), ncclFloat, ncclMax, 0, comm[i_gpu], 0) );

    cudaEventRecord(comm_stop);

    //STOP_TIMER;
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    float comp_time, comm_time;
    
    cudaEventElapsedTime(&comp_time,comp_start,comp_stop);
    cudaEventElapsedTime(&comm_time,comm_start,comm_stop);

    comp_times[i_gpu] =  comp_time;
    comm_times[i_gpu] =  comm_time;

    ncclCommDestroy(comm[i_gpu]);
    //CHECK_NCCL(ncclCommDestroy(comm[i_gpu]));

    //================ Output =================
    //if (i_gpu == 0)
    //{
    //float* output = bout->download_output();
    //validate_prediction(output, image_labels, output_size, batch);
    //}


/*
    float* out = l1b2c1->download_full_output();
    //float* out = l1b1c2->download_full_output();
    //for (int i=0; i<512; i++)
    for (int i=4096; i<4096+512; i++)
    {
        printf("%.f ", out[i]);
        if ((i+1)%32==0) printf("\n");
    }
    printf("\n===%f===\n", bout->bn_scale[0]);
*/ 

    delete bconv1;
    delete l1b1c1;
    delete l1b1c2;
    delete l1b2c1;
    delete l1b2c2;

    delete l2b1c1;
    delete l2b1c2;
    delete l2b2c1;
    delete l2b2c2;

    delete l3b1c1;
    delete l3b1c2;
    delete l3b2c1;
    delete l3b2c2;

    delete l4b1c1;
    delete l4b1c2;
    delete l4b2c1;
    delete l4b2c2;

    delete bfc1;
    delete bout;
} //end of omp

    float max_comp_time = 0;
    float max_comm_time = 0;
    for (int i=0; i<n_gpu; i++)
    {
        if (comp_times[i]>max_comp_time)
            max_comp_time = comp_times[i];
        if (comm_times[i]>max_comm_time)
            max_comm_time = comm_times[i];
    }
    *pcomp_time = max_comp_time;
    *pcomm_time = max_comm_time;
}



int main()
{
    const int tests = 2;
    for (int b=256; b<512; b*=2)
    {
        float avg_comp_time = 0;
        float avg_comm_time = 0;
        float comp_time, comm_time;
        for (int i=0; i<tests; i++)
        {
            run(b,&comp_time,&comm_time);
            avg_comp_time += comp_time;
            avg_comm_time += comm_time;
        }
        avg_comp_time /= float(tests);
        avg_comm_time /= float(tests);
        printf("Scale_up_test: Batch=%d, Comp_Time=%.3f, Comm_Time:%.3f \n",
                b,avg_comp_time, avg_comm_time);
    }
    return 0;
}




















