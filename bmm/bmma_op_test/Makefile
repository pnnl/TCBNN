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
 
CC=nvcc
FLAGS=-arch=sm_75 -O3 -maxrregcount=64 -Xptxas="-v"
#FLAGS= -gencode arch=compute_75,code=sm_75 -O3

#bmm: bmm.cu
	#$(CC) $(FLAGS) $^ -o $@

main: main_test.cu
	$(CC) $(FLAGS) $^ -o $@

#main: main.cu
	#$(CC) $(FLAGS) $^ -o $@

clean:
	rm bmm
	#rm main
