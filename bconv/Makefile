############################################################################### 
#    Description:  Accelerate BNN via TensorCores in Turing/Ampere GPU
#                  Please see our TPDS paper "Accelerating Binarized Neural 
#                  Networks via Bit-Tensor-Cores in Turing GPUs" for detail.
#                  https://arxiv.org/abs/2006.16578
#       
#      PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850.
#
#         Author:  Ang Li
#        Website:  https://www.angliphd.com
############################################################################### 

CC = nvcc

FLAGS = -arch=sm_75 -O3 -std=c++11 -Xptxas --maxrregcount=64 -w 
#LINK = -lcublas -lcudnn -I/home/lian599/cudnn_10.0/include/ -L/home/lian599/cudnn_10.0/lib64/
LINK = -lcublas -lcudnn -I/share/apps/cuda/10.1.105/include/ -L/share/apps/cuda/10.1.105/lib64/
#LINK = -lcublas -lcudnn 

bconv: benchmark-bconv.cu conv2d_kernel.cu bmma_kernel.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bconv.cu -o $@

clean:
	rm -f bconv *.o
