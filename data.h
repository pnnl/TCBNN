// ---------------------------------------------------------------------------
// File: data.h
// Load images and labels. Functions to load batched input images and labels 
// for MNIST, Cifar-10 and ImageNet in raw or batched format.
// ---------------------------------------------------------------------------
// See our arXiv paper for detail: https://arxiv.org/abs/2006.16578
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/TCBNN
// PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850
// BSD Lincese.
// Richland, 99352, WA, USA. June-30-2020.
// ---------------------------------------------------------------------------



#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cstring>
#include <jpeglib.h>
#include <jerror.h>
#include <math.h>

using namespace std;
typedef unsigned char uchar;


/** @brief Read raw batched images and labels for MNIST.
 *
 *  MNIST can be downloaded from http://yann.lecun.com/exdb/mnist/
 *  Return file is in shape [batch, height=28, width=28, channel=1].
 *
 *  @param filename The path of the image file, e.g., "./t10k-images-idx3-ubyte".
 *  @param labelname The path of the label file. e.g., "./t10k-labels-idx1-ubyte".
 *  @param images Returned batched iamges data.
 *  @param labels Returned batched labels.
 *  @param batch Batch size or the fetched number of images.
 *  @return Void
 */
void read_MNIST_raw(string filename, string labelname, uchar* images, unsigned* labels, const unsigned batch);

/** @brief Read normalized batched images and labels for MNIST.
 *
 *  MNIST can be downloaded from http://yann.lecun.com/exdb/mnist/
 *  Return file has already been normalized using [mean: 0.1307, std: 0.3081]
 *  and is in shape [batch, channel=1, height=28, width=28].
 *
 *  @param filename The path of the image file, e.g., "./t10k-images-idx3-ubyte".
 *  @param labelname The path of the label file. e.g., "./t10k-labels-idx1-ubyte".
 *  @param images Returned batched iamges data.
 *  @param labels Returned batched labels.
 *  @param batch Batch size or the fetched number of images.
 *  @return Void.
 */
void read_MNIST_normalized(string filename, string labelname, float* images, unsigned* labels, const unsigned batch);

/** @brief Read raw batched images and labels for Cifar-10.
 *
 *  Cifar-10 can be downloaded from http://www.cs.toronto.edu/~kriz/cifar.html
 *  Returned file is in shape [batch, height=32, width=32, channel=3].
 *  3 channel bytes (0-255) are packed into an unsigned int with 4th byte blank.
 *
 *  @param filename The path of the image file, e.g., "./test_batch.bin".
 *  @param images Returned batched iamges data.
 *  @param labels Returned batched labels.
 *  @param batch Batch size or the fetched number of images.
 *  @return Void.
 */
void read_CIFAR10_raw(string filename, unsigned* images, unsigned* labels, const unsigned batch);

/** @brief Read normalized batched images and labels for Cifar-10.
 *
 *  Cifar-10 can be downloaded from http://www.cs.toronto.edu/~kriz/cifar.html
 *  Returned file has already been normalized [mean: R:0.4914,G:0.4822,B:0.4465, 
 *  std: R:0.2470, G:0.2435, B:0.2616] and in shape [batch, channel=3, height=32, width=32].
 *
 *  @param filename The path of the image file, e.g., "./test_batch.bin".
 *  @param images Returned batched iamges data.
 *  @param labels Returned batched labels.
 *  @param batch Batch size or the fetched number of images.
 *  @return Void.
 */
void read_CIFAR10_normalized(string filename, float* images, unsigned* labels, const unsigned batch);

/** @brief Read raw batched images and labels for ImageNet.
 *
 *  ImageNet can be downloaded from http://image-net.org/download
 *  Returned file is in shape [batch, height=224, width=224, channel=3] and crop the original images in the center.
 *  3 channel bytes (0-255) are packed into an unsigned int with 4th byte blank.
 *
 *  @param descfilename The path of the description file, e.g., "./imagenet_fiels.csv".
 *  @param images Returned batched iamges data.
 *  @param labels Returned batched labels.
 *  @param batch Batch size or the fetched number of images.
 *  @return Void.
 */
void read_ImageNet_raw(const char* descfilename, unsigned* images, unsigned* labels, int batch);

/** @brief Read batched images and labels for ImageNet.
 *
 *  ImageNet can be downloaded from http://image-net.org/download
 *  Returned file has already been normalized [mean: R:0.485, G:0.456, B:0.406,
 *  std: R:0.229, G:0.224, B:0.225] and is in shape [batch, channel=3, height=224, width=224].
 *
 *  @param descfilename The path of the description file, e.g., "./imagenet_fiels.csv".
 *  @param images Returned batched iamges data.
 *  @param labels Returned batched labels.
 *  @param batch Batch size or the fetched number of images.
 *  @return Void.
 */
void read_ImageNet_normalized(const char* descfilename, float* images, unsigned* labels, int batch);

#endif


