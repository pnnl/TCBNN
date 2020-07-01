// ---------------------------------------------------------------------------
// File: mnist_mlp.cu
// MLP BNN inference source file for MNIST.
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

__global__ void mnist_mlp(In128LayerParam* bin, Fc128LayerParam* fc1, Fc128LayerParam* fc2, 
        Fc128LayerParam* fc3, Out128LayerParam* bout)
{
    //SET_KERNEL_TIMER;
    grid_group grid = this_grid();
    //========= Input ============
    In128LayerFMT(bin);
    grid.sync();
    //TICK_KERNEL_TIMER(bin);
    //========== FC1 ============
    Fc128LayerFMT(fc1);
    grid.sync();
    //TICK_KERNEL_TIMER(fc1);
    //========== FC2 ============
    Fc128LayerFMT(fc2);
    grid.sync();
    //TICK_KERNEL_TIMER(fc2);
    ////========== FC3 ============
    Fc128LayerFMT(fc3);
    grid.sync();
    //TICK_KERNEL_TIMER(fc3);
    //========== Output ===========
    Out128LayerFMT(bout);
    //grid.sync();
    //TICK_KERNEL_TIMER(bout);
}

#else

__global__ void mnist_mlp(In128LayerParam* bin, Fc128LayerParam* fc1, Fc128LayerParam* fc2, 
        Fc128LayerParam* fc3, Out128LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Input ============
    In128Layer(bin);
    grid.sync();
    //========== FC1 ============
    Fc128Layer(fc1);
    grid.sync();
    //========== FC2 ============
    Fc128Layer(fc2);
    grid.sync();
    ////========== FC3 ============
    Fc128Layer(fc3);
    grid.sync();
    ////========== Output ===========
    Out128Layer(bout);
}
#endif


int main()
{
    //=============== Configuration =================
    int dev = 0;
    cudaSetDevice(dev);
    const unsigned batch = 32768;
    const unsigned output_size = 10;
    const unsigned n_hidden = 1024;
    const unsigned image_height = 28;
    const unsigned image_width = 28;
    const unsigned image_size = image_height*image_width;

    //=============== Get Input and Label =================
    string mnist_dir = "/home/lian599/data/mnist/t10k-images-idx3-ubyte";
    float* images = NULL;
    SAFE_ALOC_HOST(images, image_height*image_width*batch*sizeof(float));
    string mnist_label = "/home/lian599/data/mnist/t10k-labels-idx1-ubyte";
    unsigned* image_labels = NULL;
    SAFE_ALOC_HOST(image_labels, batch*sizeof(unsigned));
    read_MNIST_normalized(mnist_dir, mnist_label, images, image_labels, batch);

    //================ Get Weight =================
    FILE* config_file = fopen("./mlp_mnist.csv","r");

    //================ Set Network =================
    //Input Layer
    In128LayerParam* bin = new In128LayerParam("Fin", batch, image_size);
    In128LayerParam* bin_gpu = bin->initialize(images);
    //Fc1 Layer
    Fc128LayerParam* bfc1 = new Fc128LayerParam("Fc1", batch, image_size, n_hidden); 
    Fc128LayerParam* bfc1_gpu = bfc1->initialize(config_file, bin->get_output_gpu());
    //Fc2 Layer
    Fc128LayerParam* bfc2 = new Fc128LayerParam("Fc2", batch, n_hidden, n_hidden); 
    Fc128LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    //Fc3 Layer
    Fc128LayerParam* bfc3 = new Fc128LayerParam("Fc3", batch, n_hidden, n_hidden); 
    Fc128LayerParam* bfc3_gpu = bfc3->initialize(config_file, bfc2->get_output_gpu());
    //Out Layer
    Out128LayerParam* bout = new Out128LayerParam("Fout", batch, n_hidden, output_size);
    Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc3->get_output_gpu());

    //Out128LayerParam* bout = new Out128LayerParam("Fout", batch, n_hidden, output_size);
    //Out128LayerParam* bout_gpu = bout->initialize(config_file, bfc1->get_output_gpu());

    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 64*sizeof(int)*32;
    cudaFuncSetAttribute(mnist_mlp, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, mnist_mlp, numThreads, shared_memory);


    void* args[] = {&bin_gpu, &bfc1_gpu, &bfc2_gpu, &bfc3_gpu, &bout_gpu};

    START_TIMER;
    cudaLaunchCooperativeKernel((void*)mnist_mlp, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    //mnist_mlp<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(bin_gpu, 
    //bfc1_gpu, bfc2_gpu, bfc3_gpu, bout_gpu);
    STOP_TIMER;

    CUDA_CHECK_KERNEL();



    //================ Output =================

    float* output = bout->download_output();
    //validate_prediction(output, image_labels, output_size, batch);

/*
    for (int i=0; i<100; i++) 
    {
        printf("%.3f ", output[i]);
        if ((i+1)%10==0) printf("\n");
    }
    printf("\n");

   
    float* out = bfc1->download_full_output();
    for (int i=0; i<256; i++) 
    {
        printf("%.0f ", out[i]);
        if ((i+1)%16==0) printf("\n");
    }
    printf("\n");
    */

/*
    printf("\n====\n");

    float* out = bfc1->download_full_output();
    for (int i=0; i<100; i++)
    {
        printf("%.0f ", out[i]);
        if ((i+1)%10==0) printf("\n");
    }
    printf("\n");

    printf("\n=OO===\n");

    unsigned* out1 = bfc1->download_output();
    for (int i=0; i<(4*(bfc1->output_bit_size())); i++)
    {
        printf("%x ", out1[i]);
        if ((i+1)%10==0) printf("\n");
    }
    printf("\n");
*/


/*
    uin32* dump_weight = NULL;
    SAFE_ALOC_HOST(dump_weight, bout->weight_bit_bytes())
        CUDA_SAFE_CALL( cudaMemcpy(dump_weight, bout->weight_gpu, bout->weight_bit_bytes(),
                    cudaMemcpyDeviceToHost));
    for (int i=0; i<32; i++)
    {
        printf("%x ", dump_weight[i]);
        if ((i+1)%16==0) printf("\n");
    }
*/





    //================ Release =================
    delete bin;
    delete bfc1;
    delete bfc2;
    delete bfc3;
    delete bout;

    SAFE_FREE_HOST(image_labels);
    SAFE_FREE_HOST(images);

    return 0;
}
