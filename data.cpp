// ---------------------------------------------------------------------------
// File: data.cpp
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

#include "data.h"

//========================== MNIST ================================
inline int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_MNIST_raw(string filename, string labelname, uchar* images, unsigned* labels, const unsigned batch)
{
    const int image_height = 28;
    const int image_width = 28;
    const int image_channel = 1;
    
    if (images == NULL)
    {
        fprintf(stderr, "NULL images pointer in reading MNIST in data.cpp.\n");
        exit(1);
    }
    if (labels == NULL)
    {
        fprintf(stderr, "NULL labels pointer in reading MNIST in data.cpp.\n");
        exit(1);
    }
    memset(images,0,batch*image_height*image_width*image_channel*sizeof(uchar));
    memset(labels,0,batch*sizeof(unsigned));

    //Read images
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

#ifdef VERBOSE
        printf("Reading raw MNIST images in a batch of %u in data.cpp.\n", batch);
#endif
        for (int i = 0; i < batch; ++i)
            for (int h = 0; h < image_height; ++h)
                for (int w = 0; w < image_width; ++w)
                {
                    uchar temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    images[i*image_height*image_width+h*image_width+w] = temp;
                }
    }
    else
    {
        fprintf(stderr, "Error in reading MNIST image file '%s' in data.cpp.\n", filename.c_str());
        exit(1);
    }

    //Read labels
    ifstream labelfile(labelname, ios::binary);
    if (labelfile.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        labelfile.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        labelfile.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

#ifdef VERBOSE
        printf("Reading raw MNIST labels in a batch of %u in data.cpp.\n", batch);
#endif

        for(int i = 0; i < batch; ++i)
        {
            uchar temp = 0;
            labelfile.read((char*) &temp, sizeof(temp));
            labels[i]= (int)temp;
        }
    }
    else
    {
        fprintf(stderr, "Error in reading MNIST label file '%s' in data.cpp.\n", labelname.c_str());
        exit(1);
    }

    file.close();
    labelfile.close();
}

void read_MNIST_normalized(string filename, string labelname, float* images, unsigned* labels, const unsigned batch)
{
    const int image_height = 28;
    const int image_width = 28;
    const int image_channel = 1;

    if (images == NULL)
    {
        fprintf(stderr, "NULL images pointer in reading MNIST in data.cpp.\n");
        exit(1);
    }
    if (labels == NULL)
    {
        fprintf(stderr, "NULL labels pointer in reading MNIST in data.cpp.\n");
        exit(1);
    }
    memset(images,0,batch*image_height*image_width*image_channel*sizeof(float));
    memset(labels,0,batch*sizeof(unsigned));

    //Read images
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

#ifdef VERBOSE
        printf("Reading normalized MNIST images in a batch of %u in data.cpp.\n", batch);
#endif
        for (int i = 0; i < batch; ++i)
            for (int h = 0; h < image_height; ++h)
                for (int w = 0; w < image_width; ++w)
                {
                    uchar temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    images[i*image_height*image_width+h*image_width+w] = (((float)temp/255.f)-0.1307f)/0.3081f; //Normalize
                }
    }
    else
    {
        fprintf(stderr, "Error in reading MNIST image file '%s' in data.cpp.\n", filename.c_str());
        exit(1);
    }

    //Read labels
    ifstream labelfile (labelname, ios::binary);
    if (labelfile.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        labelfile.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        labelfile.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

#ifdef VERBOSE
        printf("Reading normalized MNIST labels in a batch of %u in data.cpp.\n", batch);
#endif

        for(int i = 0; i < batch; ++i)
        {
            uchar temp = 0;
            labelfile.read((char*) &temp, sizeof(temp));
            labels[i]= (int)temp;
        }
    }
    else
    {
        fprintf(stderr, "Error in reading MNIST label file '%s' in data.cpp.\n", labelname.c_str());
        exit(1);
    }
    file.close();
    labelfile.close();
}


//========================== CIFAR-10 ================================
void read_CIFAR10_raw(string filename, unsigned* images, unsigned* labels, const unsigned batch)
{
    const int image_height = 32;
    const int image_width = 32;
    const int image_channel = 3;

    if (images == NULL)
    {
        fprintf(stderr, "NULL images pointer in reading Cifar-10 in data.cpp.\n");
        exit(1);
    }
    if (labels == NULL)
    {
        fprintf(stderr, "NULL labels pointer in reading Cifar-10 in data.cpp.\n");
        exit(1);
    }

    memset(images,0,batch*image_height*image_width*sizeof(unsigned));
    memset(labels,0,batch*sizeof(unsigned));

    uchar* ui = (uchar*)&images[0];
    ifstream file(filename, ios::binary);
    if (file.is_open())
    {

#ifdef VERBOSE
        printf("Reading raw CIFAR-10 in a batch of %u in data.cpp.\n", batch);
#endif
        for (int l = 0; l < batch; ++l)
        {
            //Cifar10 data stored in <1xlabel><r:1024><g:1024><b:1024>
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels[l] = (unsigned)temp;
            for (int c = 0; c<image_channel; c++) //channel
                for (int h=0; h<image_height; h++)
                    for (int w=0; w<image_width; w++)
                    {
                        file.read((char*) &temp, sizeof(temp));
                        ui[(l*image_height*image_width+h*image_width+w)*4+c] = temp;
                    }
        }
    }
    else
    {
        fprintf(stderr, "Error in reading Cifar-10 bin file '%s' in data.cpp.\n", filename.c_str());
        exit(1);
    }
    file.close();
}


void read_CIFAR10_normalized(string filename, float* images, unsigned* labels, const unsigned batch)
{
    const int image_height = 32;
    const int image_width = 32;
    const int image_channel = 3;

    if (images == NULL)
    {
        fprintf(stderr, "NULL images pointer in reading Cifar-10 in data.cpp.\n");
        exit(1);
    }
    if (labels == NULL)
    {
        fprintf(stderr, "NULL labels pointer in reading Cifar-10 in data.cpp.\n");
        exit(1);
    }

    memset(images,0,batch*image_height*image_width*image_channel*sizeof(float));
    memset(labels,0,batch*sizeof(unsigned));

    ifstream file(filename, ios::binary);
    if (file.is_open())
    {
#ifdef VERBOSE
        printf("Reading normalized CIFAR-10 in a batch of %u in data.cpp.\n", batch);
#endif
        for (int l = 0; l < batch; ++l)
        {
            //Cifar10 data stored in <1xlabel><r:1024><g:1024><b:1024>
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels[l] = (unsigned)temp;

            //Processing Red channel
            for (int h=0; h<image_height; h++)
                for (int w=0; w<image_width; w++)
                {
                    file.read((char*) &temp, sizeof(temp));
                    images[l*image_height*image_width*image_channel 
                        + 0*image_height*image_width + h*image_width + w] = 
                        ((float)temp/255.0f-0.4914f)/0.2470f;
                }
            //Processing Green channel
            for (int h=0; h<image_height; h++)
                for (int w=0; w<image_width; w++)
                {
                    file.read((char*) &temp, sizeof(temp));
                    images[l*image_height*image_width*image_channel 
                        + 1*image_height*image_width + h*image_width + w] = 
                        ((float)temp/255.0f-0.4822f)/0.2435f;
                }
            //Processing Blue channel
            for (int h=0; h<image_height; h++)
                for (int w=0; w<image_width; w++)
                {
                    file.read((char*) &temp, sizeof(temp));
                    images[l*image_height*image_width*image_channel 
                        + 2*image_height*image_width + h*image_width + w] = 
                        ((float)temp/255.0f-0.4465f)/0.2616f;
                }
        } 
    }
    else
    {
        fprintf(stderr, "Error in reading Cifar-10 bin file '%s' in data.cpp.\n", filename.c_str());
        exit(1);
    }
    file.close();
}

//========================== ImageNet =================================
void read_raw_jpeg(const char* filename, uchar* fdata, int image_height, int image_width)
{
    unsigned long x, y;
    unsigned long data_size;     // length of the file
    int channels;               //  3 =>RGB   4 =>RGBA 
    unsigned int type;  
    unsigned char * rowptr[1];    // pointer to an array
    unsigned char * jdata;        // data for the image
    struct jpeg_decompress_struct info; //for our jpeg info
    struct jpeg_error_mgr err;          //the error handler
    FILE* file = fopen(filename, "rb");  //open the file
    info.err = jpeg_std_error(& err);     
    jpeg_create_decompress(& info);   //fills info structure

    if(!file) 
    {
        fprintf(stderr, "Error reading JPEG file %s in data.cpp.\n", filename);
        exit(1);
    }
    jpeg_stdio_src(&info, file);    
    jpeg_read_header(&info, TRUE);   // read jpeg file header
    jpeg_start_decompress(&info);    // decompress the file

    //set width and height
    x = info.output_width;
    y = info.output_height;
    channels = info.num_components;
    data_size = x * y * 4;
    jdata = (uchar*)malloc(data_size);
    memset(jdata,0,data_size);
    while (info.output_scanline < info.output_height)
    {
        // Enable jpeg_read_scanlines() to fill our jdata array
        rowptr[0] = (uchar*)jdata +  3* info.output_width * info.output_scanline; 
        jpeg_read_scanlines(&info, rowptr, 1);
    }
    jpeg_finish_decompress(&info);
    double scale_w = (double)x/(double)image_width;
    double scale_h = (double)y/(double)image_height;
    for (int c=0; c<3; c++) // channel
    {
        for (int h=0; h<image_height; h++)
        {
            for (int w=0; w<image_width; w++)
            {
                int map_h = floor(scale_h*(double)h);
                int map_w = floor(scale_w*(double)w);
                fdata[(h*image_width+w)*4+c] = jdata[c*x*y+ map_h*x+map_w];
            }
        }
    }
    free(jdata);
}



void read_ImageNet_raw(const char* descfilename, unsigned* images, unsigned* labels, int batch)
{
    const int image_height = 224;
    const int image_width = 224;
    const int image_channel = 3;

    if (images == NULL)
    {
        fprintf(stderr, "NULL images pointer in reading ImageNet in data.cpp.\n");
        exit(1);
    }
    if (labels == NULL)
    {
        fprintf(stderr, "NULL labels pointer in reading ImageNet in data.cpp.\n");
        exit(1);
    }
    memset(images,0,batch*image_height*image_width*sizeof(unsigned));
    memset(labels,0,batch*sizeof(unsigned));

    FILE* descfile = fopen("imagenet_files.txt", "r");
    if(descfile == NULL) 
    {
        fprintf(stderr, "Error reading Imagenet Description File '%s'.\n", descfilename);
        exit(1);
    }
#ifdef VERBOSE
        printf("Reading raw ImageNet in a batch of %u in data.cpp.\n", batch);
#endif
    char filename[128];
    for (int i=0; i<batch; i++)
    {
        unsigned tag;
        fscanf(descfile, " %[^,\n],", filename);
        fscanf(descfile, "%u", &tag);
        labels[i] = tag;
        read_raw_jpeg(filename, (uchar*)&images[i*image_height*image_width], image_height, image_width);
    }
    fclose(descfile);
}


inline float lerp(float s, float e, float t)
{
    return s+(e-s)*t;
}
inline float blerp(float c00, float c10, float c01, float c11, float tx, float ty)
{
        return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}
void scale_transform(const uchar *src, uchar *dst, int dst_h, int dst_w, int src_h, int src_w)
{
    int x, y;
    for(x=0, y=0; y<dst_h; x++)
    {
        if(x > dst_w)
        {
            x=0; y++;
        }
        float gx = x / (float)(dst_w) * (src_w-1);
        float gy = y / (float)(dst_h) * (src_h-1);
        int gxi = (int)gx;
        int gyi = (int)gy;
            
        for (int c=0; c<3; c++)
        {
            float c00 = (float)src[c*src_w*src_h + (gyi*src_w)+gxi];
            float c10 = (float)src[c*src_w*src_h + (gyi*src_w)+gxi+1];
            float c01 = (float)src[c*src_w*src_h + ((gyi+1)*src_w)+gxi];
            float c11 = (float)src[c*src_w*src_h + ((gyi+1)*src_w)+gxi+1];
            float result = blerp(c00, c10, c01, c11, gx-gxi, gy-gyi);
            dst[c*dst_h*dst_w + y*dst_w + x] = (unsigned char)result;
        }
    }
}

void read_normalized_jpeg(const char* filename, float* fdata, int image_height, int image_width)
{
    unsigned long org_width, org_height;
    unsigned long data_size;     // length of the file
    int channels;               //  3 =>RGB   4 =>RGBA 
    unsigned int type;  
    unsigned char * rowptr[1];    // pointer to an array
    unsigned char * hwc_data;        // HWC data for the image
    unsigned char * chw_data;        // CHW data for the image
    struct jpeg_decompress_struct info; //for our jpeg info
    struct jpeg_error_mgr err;          //the error handler
    FILE* file = fopen(filename, "rb");  //open the file
    info.err = jpeg_std_error(& err);     
    jpeg_create_decompress(& info);   //fills info structure

    if(!file) 
    {
        fprintf(stderr, "Error reading JPEG file %s in data.cpp.\n", filename);
        exit(1);
    }
    jpeg_stdio_src(&info, file);    
    jpeg_read_header(&info, TRUE);   // read jpeg file header
    jpeg_start_decompress(&info);    // decompress the file

    //set width and height
    org_width = info.output_width;
    org_height = info.output_height;
    channels = info.num_components;
    data_size = org_width * org_height * 4;
    hwc_data = (uchar*)malloc(data_size);
    memset(hwc_data,0,data_size);
    while (info.output_scanline < info.output_height)
    {
        // Enable jpeg_read_scanlines() to fill our hwc_data array
        rowptr[0] = (uchar*)hwc_data +  3* info.output_width * info.output_scanline; 
        jpeg_read_scanlines(&info, rowptr, 1);
    }
    jpeg_finish_decompress(&info);

    chw_data = (uchar*)malloc(data_size);
    memset(chw_data,0,data_size);

    //Convert from HWC to CHW
    for (int h=0; h<org_height; h++)
        for (int w=0; w<org_width; w++)
            for (int c=0; c<3; c++)
                chw_data[c*org_height*org_width+h*org_width+w] 
                    = hwc_data[h*org_width*3 + w*3 + c];

    //Scale and Bilinear Interpolate
    float scale = 256.0f/min(org_width, org_height);
    int scaled_height = 0;
    int scaled_width = 0; 
    if (org_width < org_height)
    {
        scaled_width = 256; scaled_height = int(scale*org_height);
    }
    else
    {
        scaled_height = 256; scaled_width = int(scale*org_width);
    }
    
    //printf("=========%d,%d=========\n",scaled_height, scaled_width);
    uchar* scale_data = (uchar*)malloc(3*scaled_height*scaled_width*sizeof(uchar));
    scale_transform(chw_data, scale_data, scaled_height, scaled_width, org_height, org_width); 

    //CenterCrop and Normalize
    for (int h=0; h<image_height; h++)
    {
        for (int w=0; w<image_width; w++)
        {
            int map_h = h + (scaled_height - image_height)/2;
            int map_w = w + (scaled_width - image_width)/2; 
            //printf("h:%d, w:%d, map_h:%d, map_w:%d, scale:%f, scaled_height:%d, scaled_width:%d\n", h,w,map_h,map_w,scale, scaled_height, scaled_width);
            
            fdata[0*image_height*image_width + h*image_width + w] 
                = ((float)scale_data[0*scaled_width*scaled_height
                        + map_h*scaled_width+map_w]/255.0f)*4.3668f-2.1179f;
            fdata[1*image_height*image_width + h*image_width + w] 
                = ((float)scale_data[1*scaled_width*scaled_height
                        + map_h*scaled_width+map_w]/255.0f)*4.4643f-2.0357f;
            fdata[2*image_height*image_width + h*image_width + w] 
                = ((float)scale_data[2*scaled_width*scaled_height
                        + map_h*scaled_width+map_w]/255.0f)*4.4444f-1.8044f;
        }
    }

    free(chw_data);
    free(hwc_data);
    free(scale_data);
}

void read_ImageNet_normalized(const char* descfilename, float* images, unsigned* labels, int batch)
{
    const int image_height = 224;
    const int image_width = 224;
    const int image_channel = 3;

    if (images == NULL)
    {
        fprintf(stderr, "NULL images pointer in reading ImageNet in data.cpp.\n");
        exit(1);
    }
    if (labels == NULL)
    {
        fprintf(stderr, "NULL labels pointer in reading ImageNet in data.cpp.\n");
        exit(1);
    }
    memset(images,0,batch*image_height*image_width*image_channel*sizeof(float));
    memset(labels,0,batch*sizeof(unsigned));

    FILE* descfile = fopen(descfilename, "r");
    if(descfile == NULL) 
    {
        fprintf(stderr, "Error reading Imagenet Description File '%s'.\n", descfilename);
        exit(1);
    }
#ifdef VERBOSE
        printf("Reading normalized ImageNet in a batch of %u in data.cpp.\n", batch);
#endif
    char filename[128];
    for (int i=0; i<batch; i++)
    {
        unsigned tag;
        fscanf(descfile, " %[^,\n],", filename);
        fscanf(descfile, "%u", &tag);
        labels[i] = tag;
        //printf("Load file: %s with tag: %d\n",filename, tag);
        read_normalized_jpeg(filename, &images[i*image_height*image_width*image_channel], image_height, image_width);
    }

    fclose(descfile);
}


