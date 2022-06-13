#include <cudnn.h>
#include <iostream>

#define TEST_TIMES 10

typedef unsigned long long ullong;

#include "conv2d_kernel.cu"
#include "bmma_kernel.cu"

//std::exit(EXIT_FAILURE);                               \

using namespace std;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
    }                                                        \
  }

bool check_result(float* p1, float* p2, const int N) 
{
    bool flag = true;
    for (int i = 0; i < N; i++) 
    {
        //printf("(%f,%f)",p1[i],p2[i]);
        float diff = p1[i] - p2[i];
        if (fabs(diff) > 1e-6) flag = false;
    }
    return flag;
}
bool check_result(int* p1, float* p2, const int N)
{
    bool flag = true;
    for (int i = 0; i < N; i++) 
    {
        float diff = (float)p1[i] - p2[i];
        if (fabs(diff) > 1e-6)
        {
            //printf("(%d,%.0f)",p1[i],p2[i]);
            //printf("(%.0f)",(float)p1[i]-p2[i]);
            flag = false;
        }
    }
    return flag;
}

int main(int argc, char const *argv[]) 
{
    cudaSetDevice(1);

    if (argc != 7) {
        printf("./bconv input_size filter_size batch in_channel out_channel stride\n");
        exit(1);
    }

    unsigned input_width = atoi(argv[1]);
    unsigned input_height = atoi(argv[1]);
    unsigned filter_width = atoi(argv[2]);
    unsigned filter_height = atoi(argv[2]);
    unsigned batch_size = atoi(argv[3]);
    unsigned in_channels = atoi(argv[4]);
    unsigned out_channels = atoi(argv[5]);
    unsigned stride_h = atoi(argv[6]);
    unsigned stride_w = atoi(argv[6]);
    bool ispadding = true;

    cudaEvent_t start, stop;


    double cudnn_base_time, cudnn_fast_time, bblas_time, b64blas_time;
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    //input_size: argv[1]
    //filter_size: argv[2]
    //batch_size: argv[3]
    //in_channels: argv[4]
    //out_channels: argv[5]
    //stride_size: argv[6]

    unsigned output_height = ispadding?(input_height+stride_h-1)/stride_h
        :((input_height-filter_height)/stride_h+1);
    unsigned output_width = ispadding?(input_width+stride_w-1)/stride_w
        :((input_width-filter_width)/stride_w+1);

    const int pad_h = ispadding?(((output_height-1)*stride_h+filter_height-input_height)>>1):0;
    const int pad_w = ispadding?(((output_width-1)*stride_h+filter_width-input_width)>>1):0; 

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                /*format=*/CUDNN_TENSOR_NHWC,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*batch_size=*/batch_size,
                /*channels=*/in_channels,
                /*image_height=*/input_height,
                /*image_width=*/input_width));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                /*format=*/CUDNN_TENSOR_NHWC,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*batch_size=*/batch_size,
                /*channels=*/out_channels,
                /*image_height=*/output_height,
                /*image_width=*/output_width));

    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*format=*/CUDNN_TENSOR_NHWC,
                /*out_channels=*/out_channels,
                /*in_channels=*/in_channels,
                /*filter_height=*/filter_height,
                /*filter_width=*/filter_width));


    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                /*pad_height=*/pad_h,
                /*pad_width=*/pad_w,
                /*vertical_stride=*/stride_h,
                /*horizontal_stride=*/stride_w,
                /*dilation_height=*/1,
                /*dilation_width=*/1,
                /*mode=*/CUDNN_CROSS_CORRELATION,
                ///*mode=*/CUDNN_CONVOLUTION,
                /*computeType=*/CUDNN_DATA_FLOAT));




    //============================= Input
    int input_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
    int input_size = batch_size * in_channels * input_height * input_width;

    float* d_input = NULL;
    float* h_input=NULL;
	h_input = (float*)malloc(input_bytes);
    cudaMalloc(&d_input, input_bytes);

    srand(time(0));
	for (int i = 0; i < input_size ; i ++) {
        double x = (double)rand() / RAND_MAX;
        h_input[i] = (x > 0.5) ? 1 : -1;
        
        //h_input[i] = -1;
    }
    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    
    //============================= Filter
    int filter_size = in_channels * out_channels * filter_height * filter_width;
    int filter_bytes = in_channels * out_channels * filter_height * filter_width*sizeof(float);

    //printf("==========\n");
    //printf("--%d,%d--",filter_size, filter_bytes);
    //printf("\n==========\n");
        

    float* d_filter = NULL;
    float* h_filter = NULL;
	h_filter = (float*)malloc(filter_bytes);
    cudaMalloc(&d_filter, filter_bytes);

	for (int i = 0; i < filter_size ; i ++) {
        double x = (double)rand() / RAND_MAX;
        h_filter[i] = (x > 0.5) ? 1 : -1;
    }
    cudaMemcpy(d_filter, h_filter, filter_bytes, cudaMemcpyHostToDevice);
    const float alpha = 1, beta = 0;

    //============================= Output
    int output_size =  batch_size * out_channels * output_width * output_height; 
    int output_bytes =  batch_size * out_channels * output_width * output_height * sizeof(float); 
    float* d_output = NULL;
    float* h_output = NULL;
	h_output = (float*)malloc(output_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);


    //============================= CUDNN-BASE
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor,
                filter_descriptor, convolution_descriptor, output_descriptor,
                CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, /*memoryLimitInBytes=*/0,
                &convolution_algorithm));

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor,
                filter_descriptor, convolution_descriptor, output_descriptor,
                convolution_algorithm, &workspace_bytes));

    void* d_workspace = NULL;
    cudaMalloc(&d_workspace, workspace_bytes);

    //----------------------- 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i=0; i<TEST_TIMES; i++)
    {
        checkCUDNN(cudnnConvolutionForward(cudnn,
                    &alpha,
                    input_descriptor,
                    d_input,
                    filter_descriptor,
                    d_filter,
                    convolution_descriptor,
                    convolution_algorithm,
                    d_workspace,
                    workspace_bytes,
                    &beta,
                    output_descriptor,
                    d_output));
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);

    cudnn_base_time = (milliseconds*1e3)/double(TEST_TIMES);
    cudaFree(d_workspace);
    //----------------------- 

    //============================= CUDNN-FAST
    checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor,
                filter_descriptor, convolution_descriptor, output_descriptor,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, /*memoryLimitInBytes=*/0,
                &convolution_algorithm));
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor,
                filter_descriptor, convolution_descriptor, output_descriptor,
                convolution_algorithm, &workspace_bytes));
    cudaMalloc(&d_workspace, workspace_bytes);
    //----------------------- 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i=0; i<TEST_TIMES; i++)
    {
        checkCUDNN(cudnnConvolutionForward(cudnn,
                    &alpha,
                    input_descriptor,
                    d_input,
                    filter_descriptor,
                    d_filter,
                    convolution_descriptor,
                    convolution_algorithm,
                    d_workspace,
                    workspace_bytes,
                    &beta,
                    output_descriptor,
                    d_output));
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);

    cudnn_fast_time = (milliseconds*1e3)/double(TEST_TIMES);
    //----------------------- 

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_workspace);


    //============================= BCONV-32bit
    float* d_bout = NULL;
    float* h_bout = NULL;
	h_bout = (float*)malloc(output_bytes);
    cudaMalloc(&d_bout, output_bytes);
    cudaMemset(d_bout, 0, output_bytes);

    float* h_myfilter = NULL;
    float* d_myfilter = NULL;

	h_myfilter = (float*)malloc(filter_bytes);
    cudaMalloc(&d_myfilter, filter_bytes);

    ////height, width, in, out
    ///*format=*/CUDNN_TENSOR_NHWC, out, height, width, in
    for (int i=0; i<out_channels; i++)
    {
        for (int j=0; j<filter_height*filter_width; j++)
        {
            for (int k=0; k<in_channels; k++)
            {
                h_myfilter[j*in_channels*out_channels + k*out_channels + i] 
                    = h_filter[i*in_channels*filter_width*filter_height
                    + j*in_channels + k];
            }
        }
    }

    cudaMemcpy(d_myfilter, h_myfilter, filter_bytes, cudaMemcpyHostToDevice);

    unsigned* filterBinarized = NULL;
    unsigned binarized_size = out_channels*in_channels*filter_height*filter_width;
    unsigned binarized_bytes = out_channels*in_channels*filter_height*filter_width*sizeof(int);
    cudaMalloc(&filterBinarized, binarized_bytes);
    cudaMemset(filterBinarized, 0, binarized_bytes);
    bool use_64bit = false;

    //----------------------- 
    bblas_time = Conv2dFunctor<float>(
                d_input,
                d_myfilter,
                d_bout,
                filterBinarized,
                batch_size,
                input_width,
                input_height,
                in_channels,
                out_channels,
                filter_width,
                filter_height,
                stride_h, //Vertical
                stride_w, //Horizontal
                output_width,
                output_height,
                pad_h,
                pad_w,
                use_64bit);

    cudaMemcpy(h_bout, d_bout, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_bout);

    //============================= BCONV-64bit
    float* d_bout64 = NULL;
    float* h_bout64 = NULL;
    h_bout64 = (float*)malloc(output_bytes);
    cudaMalloc(&d_bout64, output_bytes);
    cudaMemset(d_bout64, 0, output_bytes);
    use_64bit = true;

    //----------------------- 
    b64blas_time = Conv2dFunctor<float>(
                d_input,
                d_myfilter,
                d_bout64,
                filterBinarized,
                batch_size,
                input_width,
                input_height,
                in_channels,
                out_channels,
                filter_width,
                filter_height,
                stride_h, //Vertical
                stride_w, //Horizontal
                output_width,
                output_height,
                pad_h,
                pad_w,
                use_64bit);
    //----------------------- 
    cudaMemcpy(h_bout64, d_bout64, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_bout64);

    //============================= BCONV-32bit-Bin
    unsigned* d_bout_bin = NULL;
    unsigned* h_bout_bin = NULL;
	h_bout_bin = (unsigned*)malloc(output_bytes);
    cudaMalloc(&d_bout_bin, output_bytes);
    cudaMemset(d_bout_bin, 0, output_bytes);
 
    use_64bit = false;

    //----------------------- 
    double bconv32_bin_time = Conv2dFunctorBinFineGrined<unsigned>(
                (unsigned*)&d_input[0],
                d_myfilter,
                d_bout_bin,
                filterBinarized,
                batch_size,
                input_width,
                input_height,
                in_channels,
                out_channels,
                filter_width,
                filter_height,
                stride_h, //Vertical
                stride_w, //Horizontal
                output_width,
                output_height,
                pad_h,
                pad_w,
                use_64bit);

    cudaMemcpy(h_bout_bin, d_bout_bin, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_bout_bin);

    //============================= BCONV-64bit-Bin
    ullong* d_bout64_bin = NULL;
    ullong* h_bout64_bin = NULL;
	h_bout64_bin = (ullong*)malloc(output_bytes);
    cudaMalloc(&d_bout64_bin, output_bytes);
    cudaMemset(d_bout64_bin, 0, output_bytes);
    use_64bit = true;
    //----------------------- 
    double bconv64_bin_time = Conv2dFunctorBinFineGrined<ullong>(
                (ullong*)d_input,
                d_myfilter,
                d_bout64_bin,
                filterBinarized,
                batch_size,
                input_width,
                input_height,
                in_channels,
                out_channels,
                filter_width,
                filter_height,
                stride_h, //Vertical
                stride_w, //Horizontal
                output_width,
                output_height,
                pad_h,
                pad_w,
                use_64bit);

    cudaMemcpy(h_bout64_bin, d_bout64_bin, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_bout64_bin);

    //============================= BMMA
    int* d_bmma = NULL;
    int* h_bmma = NULL;
    h_bmma = (int*)malloc(output_bytes);
    cudaMalloc(&d_bmma, output_bytes);
    cudaMemset(d_bmma, 0, output_bytes);

    int* d_bmma_buf = NULL;
    cudaMalloc(&d_bmma_buf, output_bytes);
    cudaMemset(d_bmma_buf, 0, output_bytes);

    //----------------------- 

    unsigned* input_binarized = NULL;
    unsigned input_binarized_size = in_channels*input_height*input_width*batch_size;
    unsigned input_binarized_bytes = input_binarized_size * sizeof(unsigned);
    cudaMalloc(&input_binarized, input_binarized_bytes);
    cudaMemset(input_binarized, 0, input_binarized_bytes);

    dim3 blockDim(32);
    dim3 filterDim(filter_height*filter_width, out_channels);
    dim3 inputDim(input_width, input_height, batch_size);
    dim3 gridDim(output_width, output_height, batch_size/8*out_channels/8);
    dim3 outputDim(output_width, output_height, batch_size);

        BTC_bin_filter<float><<<filterDim,blockDim>>>(d_myfilter, filterBinarized, 
                in_channels, out_channels, filter_width, filter_height);
        BTC_bin_input<float><<<inputDim,blockDim>>>(d_input, input_binarized,
                in_channels, batch_size, input_width, input_height);

    cudaEventRecord(start);


    for (int i=0; i<TEST_TIMES; i++)
    {

        BTC_Conv2d_balance<<<gridDim, blockDim>>>(input_binarized, filterBinarized, d_bmma_buf, 
                in_channels, out_channels, input_width, input_height, filter_width, filter_height,
                batch_size, stride_h, stride_w, 
                output_width, output_height, pad_h, pad_w);
    }

    cudaEventRecord(stop);

    BTC_trans_output<<<outputDim, blockDim>>>(d_bmma_buf, d_bmma, out_channels, batch_size,
        output_width, output_height);

    //cudaThreadSynchronize();
    //cudaError err = cudaGetLastError();

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    float bmma_time = (milliseconds*1e3)/TEST_TIMES;

    //----------------------- 
    cudaMemcpy(h_bmma, d_bmma, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_bmma);
    cudaFree(d_bmma_buf);


    //============================= BMMA_FMT
    int* d_bmma_fmt = NULL;
    int* h_bmma_fmt = NULL;
    h_bmma_fmt = (int*)malloc(output_bytes);
    cudaMalloc(&d_bmma_fmt, output_bytes);
    cudaMemset(d_bmma_fmt, 0, output_bytes);

    int* d_bmma_fmt_buf = NULL;
    cudaMalloc(&d_bmma_fmt_buf, output_bytes);
    cudaMemset(d_bmma_fmt_buf, 0, output_bytes);

    //----------------------- 

    unsigned* input_binarized_fmt = NULL;
    unsigned input_binarized_fmt_size = in_channels*input_height*input_width*batch_size;
    unsigned input_binarized_fmt_bytes = input_binarized_fmt_size * sizeof(unsigned);
    cudaMalloc(&input_binarized_fmt, input_binarized_fmt_bytes);
    cudaMemset(input_binarized_fmt, 0, input_binarized_fmt_bytes);

    //dim3 blockDim(32);
    //dim3 filterDim(filter_height*filter_width, out_channels);
    //dim3 inputDim(input_width, input_height, batch_size);
    //dim3 gridDim(output_width, output_height, batch_size/8*out_channels/8);
    //dim3 outputDim(output_width, output_height, batch_size);

        BTC_bin_filter_128<float><<<filterDim,blockDim>>>(d_myfilter, filterBinarized, 
                in_channels, out_channels, filter_width, filter_height);
        BTC_bin_input_128<float><<<inputDim,blockDim>>>(d_input, input_binarized_fmt,
                in_channels, batch_size, input_width, input_height);

    cudaEventRecord(start);


    for (int i=0; i<TEST_TIMES; i++)
    {

        BTC_Conv2d_balance_128<<<gridDim, blockDim>>>(input_binarized_fmt, filterBinarized, 
                d_bmma_fmt_buf, 
                in_channels, out_channels, input_width, input_height, filter_width, filter_height,
                batch_size, stride_h, stride_w, 
                output_width, output_height, pad_h, pad_w);
    }

    cudaEventRecord(stop);

    BTC_trans_output<<<outputDim, blockDim>>>(d_bmma_fmt_buf, d_bmma_fmt, out_channels, batch_size,
        output_width, output_height);

    //cudaThreadSynchronize();
    //cudaError err = cudaGetLastError();

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    float bmma_fmt_time = (milliseconds*1e3)/TEST_TIMES;

    //----------------------- 
    cudaMemcpy(h_bmma_fmt, d_bmma_fmt, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_bmma_fmt);


    //============================= BMMA_bin
    int* d_bmma_bin = NULL;
    int* h_bmma_bin = NULL;
    h_bmma_bin = (int*)malloc(output_bytes);
    cudaMalloc(&d_bmma_bin, output_bytes);
    cudaMemset(d_bmma_bin, 0, output_bytes);

    int* d_bmma_bin_buf = NULL;
    cudaMalloc(&d_bmma_bin_buf, output_bytes);
    cudaMemset(d_bmma_bin_buf, 0, output_bytes);

    //----------------------- 

    unsigned* input_bin_binarized = NULL;
    unsigned input_bin_binarized_size = in_channels*input_height*input_width*batch_size;
    unsigned input_bin_binarized_bytes = input_bin_binarized_size * sizeof(unsigned);
    cudaMalloc(&input_bin_binarized, input_bin_binarized_bytes);
    cudaMemset(input_bin_binarized, 0, input_bin_binarized_bytes);

        BTC_bin_filter<float><<<filterDim,blockDim>>>(d_myfilter, filterBinarized, 
                in_channels, out_channels, filter_width, filter_height);
        BTC_bin_input<float><<<inputDim,blockDim>>>(d_input, input_bin_binarized,
                in_channels, batch_size, input_width, input_height);

    cudaEventRecord(start);


    for (int i=0; i<TEST_TIMES; i++)
    {

        BTC_Conv2d_balance_bin<<<gridDim, blockDim>>>(input_bin_binarized, 
                filterBinarized, d_bmma_bin_buf, 
                in_channels, out_channels, input_width, input_height, filter_width, filter_height,
                batch_size, stride_h, stride_w, 
                output_width, output_height, pad_h, pad_w);
    }

    cudaEventRecord(stop);

    BTC_trans_output<<<outputDim, blockDim>>>(d_bmma_bin_buf, d_bmma_bin, 
            out_channels, batch_size,
            output_width, output_height);

    //cudaThreadSynchronize();
    //cudaError err = cudaGetLastError();

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    float bmma_bin_time = (milliseconds*1e3)/TEST_TIMES;

    //----------------------- 
    cudaMemcpy(h_bmma_bin, d_bmma_bin, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_bmma_bin);
    cudaFree(d_bmma_bin_buf);






    //============================= BMMA_FMT
    int* d_bmma_fmt_bin = NULL;
    int* h_bmma_fmt_bin = NULL;
    h_bmma_fmt_bin = (int*)malloc(output_bytes);
    cudaMalloc(&d_bmma_fmt_bin, output_bytes);
    cudaMemset(d_bmma_fmt_bin, 0, output_bytes);

    int* d_bmma_fmt_bin_buf = NULL;
    cudaMalloc(&d_bmma_fmt_bin_buf, output_bytes);
    cudaMemset(d_bmma_fmt_bin_buf, 0, output_bytes);

    //----------------------- 

    unsigned* input_binarized_fmt_bin = NULL;
    unsigned input_binarized_fmt_bin_size = in_channels*input_height*input_width*batch_size;
    unsigned input_binarized_fmt_bin_bytes = input_binarized_fmt_bin_size * sizeof(unsigned);
    cudaMalloc(&input_binarized_fmt_bin, input_binarized_fmt_bin_bytes);
    cudaMemset(input_binarized_fmt_bin, 0, input_binarized_fmt_bin_bytes);

    //dim3 blockDim(32);
    //dim3 filterDim(filter_height*filter_width, out_channels);
    //dim3 inputDim(input_width, input_height, batch_size);
    //dim3 gridDim(output_width, output_height, batch_size/8*out_channels/8);
    //dim3 outputDim(output_width, output_height, batch_size);

        BTC_bin_filter_128<float><<<filterDim,blockDim>>>(d_myfilter, filterBinarized, 
                in_channels, out_channels, filter_width, filter_height);
        BTC_bin_input_128<float><<<inputDim,blockDim>>>(d_input, input_binarized_fmt_bin,
                in_channels, batch_size, input_width, input_height);

    cudaEventRecord(start);


    for (int i=0; i<TEST_TIMES; i++)
    {

        BTC_Conv2d_balance_128_bin<<<gridDim, blockDim>>>(input_binarized_fmt_bin, 
                filterBinarized, 
                d_bmma_fmt_bin_buf, 
                in_channels, out_channels, input_width, input_height, filter_width, filter_height,
                batch_size, stride_h, stride_w, 
                output_width, output_height, pad_h, pad_w);
    }

    cudaEventRecord(stop);

    BTC_trans_output<<<outputDim, blockDim>>>(d_bmma_fmt_bin_buf, d_bmma_fmt_bin, 
            out_channels, batch_size,
            output_width, output_height);

    //cudaThreadSynchronize();
    //cudaError err = cudaGetLastError();

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    float bmma_fmt_bin_time = (milliseconds*1e3)/TEST_TIMES;

    //----------------------- 
    cudaMemcpy(h_bmma_fmt_bin, d_bmma_fmt_bin, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_bmma_fmt_bin);











    ////==================================================== 
    printf("BBLAS-32 success: %d\n", check_result(h_output, h_bout, output_size));
    printf("BBLAS-64 success: %d\n", check_result(h_output, h_bout64, output_size));
    printf("BMMA success: %d\n", check_result(h_bmma, h_output, output_size));
    printf("BMMA-FMT128 success: %d\n", check_result(h_bmma_fmt, h_output, output_size));

    printf("CUDNN-BASE:%.3lf, CUDNN-FAST:%.3lf, BCONV-32:%.3lf, BCONV-64:%.3lf, BCONV-32-Bin:%.3lf, BCONV-64-Bin:%.3lf, BMMA:%.3lf, BMMA-FMT:%.3lf, BMMA-Bin:%.3lf, BMMA-FMT-Bin:%.3lf \n", 
            cudnn_base_time, cudnn_fast_time, bblas_time, b64blas_time,
            bconv32_bin_time, bconv64_bin_time, bmma_time, bmma_fmt_time,
            bmma_bin_time, bmma_fmt_bin_time);
    //printf("CUDNN:%.3lf BSTC-32:%.3lf \n", cudnn_time, bblas_time);


    // Do something with h_output ...

    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(filterBinarized);

    cudaFree(d_myfilter);

    free(h_filter);
    free(h_input);
    free(h_output);
    free(h_bout);
    free(h_bout64);
    free(h_myfilter);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);


}
