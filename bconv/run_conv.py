import string
import commands
import os
import math

platform="RTX2080TI"
arch="TU104"
outfile = open(str('bconv_') + platform + '.txt',"w",0)

def dumpstr(schm,i,f,b,c,o,s,t,test):
    out = str("INSERT INTO bconv VALUES ('") + \
            schm + "','" + platform + "','" + arch + "'," + \
            str(i) + ',' + str(f) + ',' + str(b) + ',' + str(c) + ',' + str(o) + ',' +\
            str(s) + ',' + str(t) + ",'" + str(test) + "');\n"
    outfile.write(out)
    #print out
    return out

def test(inputs, filters, batches, in_channels, out_channels, strids, test):
    for i in inputs:
        for f in filters:
            for b in batches:
                for c,o in zip(in_channels, out_channels):
                    for s in strids:
                        cmd = str("./bconv ") + str(i) + ' ' + str(f) + ' ' + str(b) + ' ' + str(c) + ' ' + str(o) + ' ' + str(s)
                        print '$:' + cmd
                        feedback = commands.getoutput(cmd).strip().split(',')
                        cudnn_base_t = float(feedback[0][feedback[0].find(':')+1:])
                        cudnn_fast_t = float(feedback[1][feedback[1].find(':')+1:])
                        bconv32 = float(feedback[2][feedback[2].find(':')+1:])
                        bconv64 = float(feedback[3][feedback[3].find(':')+1:])
                        bconv32_bin = float(feedback[4][feedback[4].find(':')+1:])
                        bconv64_bin = float(feedback[5][feedback[5].find(':')+1:])
                        bmma = float(feedback[6][feedback[6].find(':')+1:])
                        bmma_fmt = float(feedback[7][feedback[7].find(':')+1:])
                        bmma_bin = float(feedback[8][feedback[8].find(':')+1:])
                        bmma_fmt_bin = float(feedback[9][feedback[9].find(':')+1:])

                        dumpstr('cudnn_base',i,f,b,c,o,s,cudnn_base_t,test)
                        dumpstr('cudnn_fast',i,f,b,c,o,s,cudnn_fast_t,test)
                        dumpstr('bconv32',i,f,b,c,o,s,bconv32,test)
                        dumpstr('bconv64',i,f,b,c,o,s,bconv64,test)
                        dumpstr('bconv32_bin',i,f,b,c,o,s,bconv32_bin,test)
                        dumpstr('bconv64_bin',i,f,b,c,o,s,bconv64_bin,test)
                        dumpstr('bmma',i,f,b,c,o,s,bmma,test)
                        dumpstr('bmma_bin',i,f,b,c,o,s,bmma_bin,test)
                        dumpstr('bmma_fmt',i,f,b,c,o,s,bmma_fmt,test)
                        dumpstr('bmma_fmt_bin',i,f,b,c,o,s,bmma_fmt_bin,test)

def test_input():
    input_size = [i for i in range(32,257,32)]
    filter_size = [3]
    batch = [16]
    in_channel = [128]
    out_channel = [128]
    stride = [1]
    test(input_size, filter_size, batch, in_channel, out_channel, stride, "input")


def test_filter():
    input_size = [64]
    filter_size = [11,9,7,5,3]
    batch = [16]
    in_channel = [128]
    out_channel = [128]
    stride = [1]
    test(input_size, filter_size, batch, in_channel, out_channel, stride, "filter")

def test_batch():
    input_size = [64]
    filter_size = [3]
    batch = [8,16,32,64,128,256,512]
    in_channel = [128]
    out_channel = [128]
    stride = [1]
    test(input_size, filter_size, batch, in_channel, out_channel, stride, "batch")

def test_channel():
    input_size = [64]
    filter_size = [3]
    batch = [16]
    in_channel = [i for i in range(128,2049,128)]
    out_channel = [i for i in range(128,2049,128)]
    stride = [1]
    test(input_size, filter_size, batch, in_channel, out_channel, stride, "channel")


test_input()
test_filter()
test_batch()
test_channel()

outfile.close()





