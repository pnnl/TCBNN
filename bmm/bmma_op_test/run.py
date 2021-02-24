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
 
import string
import os
import subprocess

Test_time = 5

def run_once(k):
    cmd = "./main 1024 1024 1024 " + str(k) + "\n"
    print cmd
    res = subprocess.check_output(cmd, shell=True)
    for line in res.split("\n"):
        if line.find("cycles") != -1:
            words = line.strip().split(" ")
            cycles = int(words[3])
            print cycles
    return cycles


fout = open("read_cycle.txt","w",buffering=0)
latency = []


#for k in range(32, 1025, 32): #global read
#for k in range(8, 1025, 8): #global write
for k in range(128, 1025, 128): #shared read
#for k in range(8, 1025, 8): #shared write
    cycles = 0
    for t in range(0,Test_time): 
        cycles += run_once(k)
    cycles /= Test_time
    latency.append(cycles)
    lin = str(k) + " " + str(cycles) + "\n"
    fout.write(lin)

print latency
fout.close()







