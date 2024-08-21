# QMix: Quality-aware Learning with Mixed Noise for Robust Retinal Disease Diagnosis



**[NOTE!!] The code will be gradually opened, and be completely opened after this paper is published.**

Due to the complex nature of medical image acquisition and annotation, medical datasets inevitably contain noise. This adversely affects the robustness and generalization of deep neural networks. Previous noise learning methods mainly considered noise arising from images being mislabeled, i.e. label noise, assuming all mislabeled images were of high quality. However, medical images can also suffer from severe data quality issues, i.e. data noise, where discriminative visual features for disease diagnosis are missing. In this paper, we propose QMix, a noise learning framework that learns a robust disease diagnosis model under mixed noise scenarios.
QMix alternates between sample separation and quality-aware semi-supervised training in each epoch. The sample separation phase uses a joint uncertainty-loss criterion to effectively separate (1) correctly labeled images, (2) mislabeled high-quality images, and (3) mislabeled low-quality images. The semi-supervised training phase then learns a robust disease diagnosis model from the separated samples. Specifically, we propose a sample-reweighing loss to mitigate the effect of mislabeled low-quality images during training, and a contrastive enhancement loss to further distinguish them from correctly labeled images. QMix achieved state-of-the-art performance on five public retinal image datasets and exhibited significant improvements in robustness against mixed noise. 

![](https://github.com/FDU-VTS/DRTiD/blob/main/src/intro2.png)


## Paper
This repository provides the official PyTorch implementation of QMix in the following papers:

**QMix: Quality-aware Learning with Mixed Noise for Robust Retinal Disease Diagnosis** [[arxiv](https://arxiv.org/pdf/2404.05169)]

Junlin Hou, Jilan Xu, Rui Feng, Hao Chen

The Hong Kong University of Science and Technology, Fudan University

## Citation
```
@article{hou2024qmix,
  title={QMix: Quality-aware Learning with Mixed Noise for Robust Retinal Disease Diagnosis},
  author={Hou, Junlin and Xu, Jilan and Feng, Rui and Chen, Hao},
  journal={arXiv preprint arXiv:2404.05169},
  year={2024}
}
```

## Acknowledgments

```

```

## Contact

If you have any questions, please feel free to contact Dr. Junlin Hou (csejlhou@ust.hk).
