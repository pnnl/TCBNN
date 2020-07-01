// ---------------------------------------------------------------------------
// File: cifar10_vgg.cu
// VGG-Net BNN inference source file for CIFAR10.
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
#include "utility.h"
#include "param.h"
#include "kernel.cuh"
#include "data.h"

using namespace cooperative_groups;
using namespace std;


#ifdef NEWFMT
__global__ void vggnet128(
        InConv128LayerParam* bconv1, 
        Conv128LayerParam* bconv2, 
        Conv128LayerParam* bconv3,
        Conv128LayerParam* bconv4, 
        Conv128LayerParam* bconv5, 
        Conv128LayerParam* bconv6,
        Fc128LayerParam* bfc1, 
        Fc128LayerParam* bfc2, 
        Out128LayerParam* bout)
{
    //SET_KERNEL_TIMER;
    grid_group grid = this_grid();

    //========= Conv1 ============
    InConv128LayerFMT(bconv1);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv1);
    //========= Conv2 ============
    Conv128LayerFMT(bconv2);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv2);
    //========= Conv3 ============
    Conv128LayerFMT(bconv3);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv3);
    //========= Conv4 ============
    Conv128LayerFMT(bconv4);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv4);
    //========= Conv5 ============
    Conv128LayerFMT(bconv5);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv5);
    //========= Conv6 ============
    Conv128LayerFMT(bconv6);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv6);
    //========= Fc1 ============
    Fc128LayerFMT(bfc1);
    grid.sync();
    //TICK_KERNEL_TIMER(bfc1);
    //========= Fc2 ============
    Fc128LayerFMT(bfc2);
    grid.sync();
    //TICK_KERNEL_TIMER(bfc2);
    ////========== Output ===========
    Out128LayerFMT(bout);
    //grid.sync();
    //TICK_KERNEL_TIMER(bout);
}

#else
__global__ void vggnet128(
        InConv128LayerParam* bconv1, 
        Conv128LayerParam* bconv2, 
        Conv128LayerParam* bconv3,
        Conv128LayerParam* bconv4, 
        Conv128LayerParam* bconv5, 
        Conv128LayerParam* bconv6,
        Fc128LayerParam* bfc1, 
        Fc128LayerParam* bfc2, 
        Out128LayerParam* bout)
{
    grid_group grid = this_grid();
    //SET_KERNEL_TIMER;
    //========= Conv1 ============
    InConv128Layer(bconv1);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv1);
    //========= Conv2 ============
    Conv128Layer(bconv2);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv2);
    //========= Conv3 ============
    Conv128Layer(bconv3);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv3);
    //========= Conv4 ============
    Conv128Layer(bconv4);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv4);
    //========= Conv5 ============
    Conv128Layer(bconv5);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv5);
    //========= Conv6 ============
    Conv128Layer(bconv6);
    grid.sync();
    //TICK_KERNEL_TIMER(bconv6);
    //========= Fc1 ============
    Fc128Layer(bfc1);
    grid.sync();
    //TICK_KERNEL_TIMER(bfc1);
    //========= Fc2 ============
    Fc128Layer(bfc2);
    grid.sync();
    //TICK_KERNEL_TIMER(bfc2);
    ////========== Output ===========
    Out128Layer(bout);
    //TICK_KERNEL_TIMER(bout);
}
#endif

int main()
{
    int dev = 0;
    cudaSetDevice(dev);
    const unsigned batch = 512;
    const unsigned output_size = 10;
    const unsigned image_height = 32;
    const unsigned image_width = 32;
    const unsigned image_channel = 3;
    const unsigned filter_height = 3;
    const unsigned filter_width = 3;
    const unsigned n_hidden = 1024;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    string cifar10_dir = "/home/lian599/data/cifar10c/test_batch.bin";
    read_CIFAR10_normalized(cifar10_dir, images, image_labels, batch);

    //================ Get Weight =================
    //FILE* config_file = fopen("./cifar10.config","r");
    FILE* config_file = fopen("./vgg_cifar10.csv","r");

    //================ Set Network =================
    //Bconv1 Layer
    InConv128LayerParam* bconv1 = new InConv128LayerParam("Conv1", image_height, image_width, 
            filter_height, filter_width, 3, 128, batch); 
    InConv128LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);

    //Bconv2 Layer
    Conv128LayerParam* bconv2 = new Conv128LayerParam("Conv2", bconv1->output_height, 
            bconv1->output_width, filter_height, filter_width, 128, 128, batch, 1, 1,
            true, 2, 2, false);
    Conv128LayerParam* bconv2_gpu = bconv2->initialize(config_file, bconv1->get_output_gpu());
    //Bconv3 Layer
    Conv128LayerParam* bconv3 = new Conv128LayerParam("Conv3", bconv2->output_height, 
            bconv2->output_width, filter_height, filter_width, 128, 256, batch);
    Conv128LayerParam* bconv3_gpu = bconv3->initialize(config_file, bconv2->get_output_gpu());
    //Bconv4 Layer
    Conv128LayerParam* bconv4 = new Conv128LayerParam("Conv4", bconv3->output_height, 
            bconv3->output_width, filter_height, filter_width, 256, 256, batch, 1, 1,
            true, 2, 2, false);
    Conv128LayerParam* bconv4_gpu = bconv4->initialize(config_file, bconv3->get_output_gpu());
    //Bconv5 Layer
    Conv128LayerParam* bconv5 = new Conv128LayerParam("Conv5", bconv4->output_height, 
            bconv4->output_width, filter_height, filter_width, 256, 512, batch);
    Conv128LayerParam* bconv5_gpu = bconv5->initialize(config_file, bconv4->get_output_gpu());
    //Bconv6 Layer
    Conv128LayerParam* bconv6 = new Conv128LayerParam("Conv6", bconv5->output_height, 
            bconv5->output_width, filter_height, filter_width, 512, 512, batch, 1, 1,
            true, 2, 2, true);
    Conv128LayerParam* bconv6_gpu = bconv6->initialize(config_file, bconv5->get_output_gpu());
    //Fc1 Layer
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, (bconv6->output_height)
            *(bconv6->output_width)*512, n_hidden); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, bconv6->get_output_gpu());
    //Fc2 Layer
    Fc128LayerParam* bfc2 = new Fc128LayerParam("Fc2", batch, n_hidden, n_hidden); 
    Fc128LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    
    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, n_hidden, output_size);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc2->get_output_gpu());  


    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    //int shared_memory = 512*sizeof(int)*32;
    int shared_memory = 256*sizeof(int)*32;
    cudaFuncSetAttribute(vggnet128, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, vggnet128, numThreads, shared_memory);

    //printf("\n========= blk:%d ==========\n",numBlocksPerSm);


    void* args[] = {&bconv1_gpu, &bconv2_gpu, &bconv3_gpu, &bconv4_gpu, &bconv5_gpu, &bconv6_gpu,
        &bfc1_gpu, &bfc2_gpu, &bout_gpu};

    START_TIMER;

    cudaLaunchCooperativeKernel((void*)vggnet128, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    STOP_TIMER;
    CUDA_CHECK_KERNEL();

/*
    float* out = bfc1->download_full_output();
    for (int i=65536; i<65536+256; i++)
    //for (int i=8192; i<8192+256; i++)
    {
        printf("%.f ", out[i]);
        if ((i+1)%16==0) printf("\n");
    }

    printf("\n===%f===\n", bout->bn_scale[0]);
*/
/*
    float* ss = bconv1->download_full_output();
    int a = 0;
    int b = 100;
    int max_width = 4;
    for (int i=a; i<b; i++)
    {
        printf("%*.0f ",max_width, ss[i]);
        if ( (i-a+1)%18 == 0)
            printf("\n");
    }
    printf("\n");
*/

    //================ Output =================
    float* output = bout->download_output();
    //validate_prediction(output, image_labels, output_size, batch);


    //for (int i=0; i<256; i++)
    //{
    //printf("%f ",output[i]);
    //if ((i+1)%10==0) printf("\n");
    //}

    delete bconv1;
    delete bconv2;
    delete bconv3;
    delete bconv4;
    delete bconv5;
    delete bconv6;
    delete bfc1;
    delete bfc2;
    delete bout;

    return 0;
}

