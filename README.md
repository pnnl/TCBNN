# Tensor-Core Accelerated Binarized Neural Network
An efficient Binarized-Neural-Network (BNN) design accelerated by 
NVIDIA Turing Bit-Tensor-Cores. 
Please see our paper on [arXiv](https://arxiv.org/abs/2006.16578)) for details.

For our referencing BSTC SBNN design, please see our [SuperComputing-20 paper](https://dl.acm.org/doi/10.1145/3295500.3356169) for detail and our [SBNN repository](https://github.com/uuudown/SBNN).

![alt text](example.png)

## Current version

Latest version: **0.1**

## About TC-BNN:

Despite foreseeing tremendous speedups over conventional deep neural networks, the performance advantage of binarized neural networks (BNNs) has merely been showcased on general-purpose processors such as CPUs and GPUs. In fact, due to being unable to leverage bit-level-parallelism with a word-based architecture, GPUs have been criticized for extremely low utilization (1%) when executing BNNs. Consequently, the latest tensorcores in NVIDIA Turing GPUs start to experimentally support bit computation. In this work, we look into this brand new bit computation capability and characterize its unique features. We show that the stride of memory access can significantly affect performance delivery and a data-format co-design is highly desired to support the tensorcores for achieving superior performance than existing software solutions without tensorcores. We realize the tensorcore-accelerated BNN design, particularly the major functions for fully-connect and convolution layers — bit matrix multiplication and bit convolution. Evaluations on two NVIDIA Turing GPUs show that, with ResNet-18, our BTC-BNN design can process ImageNet at a rate of 5.6K images per second, 77% faster than state-of-the-art.

## Make and Run
Update **Makefile** accordingly and make. You will need a NVIDIA Turing GPU (Compute Capability-7.5) to be able to run.
```text
make
```

## Authors 

#### [Ang Li](http://www.angliphd.com/), Pacific Northwest National Laboratory (PNNL)

## Citation format

For research articles, please cite our paper:

- Ang Li, Simon Su, "Accelerating Binarized Neural Networks via Bit-Tensor-Cores in Turing GPUs" [[arXiv:2006.16578]](https://arxiv.org/abs/2006.16578).

Bibtex:
```text
@article{li2020accelerating,
    title={Accelerating Binarized Neural Networks via Bit-Tensor-Cores in Turing GPUs},
    author={Li, Ang and Su, Simon},
    journal={arXiv preprint arXiv:2006.16578},
    year={2020}
}

```

## License

This project is licensed under the BSD License, see [LICENSE](LICENSE) file for details.

## Acknowledgments

PNNL-IPID: 31925-E, ECCN: EAR99, IR: PNNL-SA-152850

This research was supported by PNNL's DeepScience-HPC and DMC-CFA LDRD projects. This research was supported by the U.S. DOE Office of Science, Office of Advanced Scientific Computing Research, under award 66150: "CENATE - Center for Advanced Architecture Evaluation". The Pacific Northwest National Laboratory (PNNL) is operated by Battelle for the U.S. Department of Energy (DOE) under contract DE-AC05-76RL01830. 

## Contributing

Please contact us If you'd like to contribute to TC-BNN. Thank you!
