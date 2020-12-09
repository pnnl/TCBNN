# ---------------------------------------------------------------------------
# File: Makefile
# ---------------------------------------------------------------------------
# See our arXiv paper for detail: https://arxiv.org/abs/2006.16578
# Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# GitHub repo: http://www.github.com/pnnl/TCBNN
# PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850
# BSD Lincese.
# Richland, 99352, WA, USA. June-30-2020.
# ---------------------------------------------------------------------------


NVCC = nvcc
NVCC_FLAG = -std=c++11 -O3 -w  -arch=sm_75 -maxrregcount=64 -rdc=true 
LIBS = -ljpeg

# For debug
#NVCC_FLAG = -std=c++11 -w -O0 -g -G -arch=sm_75 -maxrregcount=64 -rdc=true -Xptxas -v

#all: cifar10_resnet mnist_mlp cifar10_vgg
#all: imagenet_vgg imagenet_alexnet imagenet_resnet
all: imagenet_resnet cifar10_resnet mnist_mlp cifar10_vgg imagenet_alexnet imagenet_vgg

#all: benn_scaleup 



mnist_mlp: mnist_mlp.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ mnist_mlp.cu data.cpp $(LIBS)

cifar10_vgg: cifar10_vgg.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ cifar10_vgg.cu data.cpp $(LIBS)

cifar10_resnet: cifar10_resnet.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ cifar10_resnet.cu data.cpp $(LIBS)

imagenet_alexnet: alexnet.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ alexnet.cu data.cpp $(LIBS)

imagenet_vgg: imagenet_vgg.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ imagenet_vgg.cu data.cpp $(LIBS)

imagenet_resnet: imagenet_resnet.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ imagenet_resnet.cu data.cpp $(LIBS)

benn_scaleup: benn_scaleup.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -Xcompiler -fopenmp -lnccl -I/home/lian599/opt/nccl_v1/include -L/home/lian599/opt/nccl_v1/lib -o $@ benn_scaleup.cu data.cpp $(LIBS)

benn_scaleout: benn_scaleout.cu param.h kernel.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -ccbin mpicxx -o $@ benn_scaleup.cu data.cpp $(LIBS)


clean:
	rm mnist_mlp cifar10_vgg cifar10_resnet

