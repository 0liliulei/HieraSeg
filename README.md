# HieraSeg (CVPR'22)
[Liulei Li](https://scholar.google.com/citations?user=eCrBWngAAAAJ&hl=en), [Tianfei Zhou](https://www.tfzhou.com/), [Wenguan Wang](https://sites.google.com/view/wenguanwang/)&dagger;, [Jianwu Li](https://scholar.google.com/citations?hl=en&user=rWIZNIwAAAAJ), [Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en)

[[`arXiv`](https://arxiv.org/abs/2203.14335)] [[`BibTeX`](#CitingHieraSeg)]

## Updates
* The Pytorch implementation has already been released!
* New work [GMMSeg](https://arxiv.org/abs/2210.02025), the first generative neural framework for semantic segmentation has been accepted by NeurIPS2022! [[code for GMMSeg](https://github.com/leonnnop/GMMSeg)]
* This repo will release an official **PaddlePaddle** implementation for paper: Deep Hierarchical Semantic Segmentation.

## Abstract
Humans are able to recognize structured relations in observation, allowing us to decompose complex scenes into simpler parts and abstract the visual world in multiple levels. However, such hierarchical reasoning ability of human perception remains largely unexplored in current literature of semantic segmentation. Existing work is often aware of flatten labels and predicts target classes exclusively for each pixel. In this paper, we instead address hierarchical semantic segmentation (HSS), which aims at structured, pixel-wise description of visual observation in terms of a class hierarchy. We devise HSSN, a general HSS framework that tackles two critical issues in this task: i) how to efficiently adapt existing hierarchy-agnostic segmentation networks to the HSS setting, and ii) how to leverage the hierarchy information to regularize HSS network learning. To address i), HSSN directly casts HSS as a pixel-wise multi-label classification task, only bringing minimal architecture change to current segmentation models. To solve ii), HSSN first explores inherent properties of the hierarchy as a training objective, which enforces segmentation predictions to obey the hierarchy structure. Further, with hierarchy-induced margin constraints, HSSN reshapes the pixel embedding space, so as to generate well-structured pixel representations and improve segmentation eventually. We conduct experiments on four semantic segmentation datasets (i.e., Mapillary Vistas 2.0, Cityscapes, LIP, and PASCAL-Person-Part), with different class hierarchies, segmentation network architectures and backbones, showing the generalization and superiority of HSSN.
<p align="center">
<img src="https://github.com/0liliulei/HieraSeg/blob/main/fig.png" width="650">
</p>

## Other Related Work
[GMMSeg: Gaussian Mixture based Generative Semantic Segmentation Models](https://github.com/leonnnop/GMMSeg)

[Exploring Cross-Image Pixel Contrast for Semantic Segmentation (ICCV21-Oral)](https://github.com/tfzhou/ContrastiveSeg)

[Rethinking Semantic Segmentation: A Prototype View (CVPR22-Oral)](https://github.com/tfzhou/ProtoSeg)

## <a name="CitingHieraSeg"></a>Citing HieraSeg
```BibTeX
@article{li2022deep,
  title={Deep Hierarchical Semantic Segmentation},
  author={Li, Liulei and Zhou, Tianfei and Wang, Wenguan and Li, Jianwu and Yang, Yi},
  journal={arXiv preprint arXiv:2203.14335},
  year={2022}
}
```
