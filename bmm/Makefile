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

FLAGS = -arch=sm_80 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96 
LINK = -lcublas

all: bmmu bmm

bmm: benchmark-bmm.cu matmul_kernel.cu binary_kernels.cu tensorcore_kernel.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bmm.cu -o $@

bmmu: bmmu.cu matmul_kernel.cu binary_kernels.cu tensorcore_kernel.cu
	$(CC) $(FLAGS) $(LINK) bmmu.cu -o $@

clean:
	rm -f bmm *.o
