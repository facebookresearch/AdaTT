# AdaTT

Welcome to the AdaTT repository! This repository provides a PyTorch library for multitask learning, specifically focused on the models evaluated in the paper ["AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations" (KDD'23)"](https://doi.org/10.1145/3580305.3599769).

[**[arXiv]**](https://arxiv.org/abs/2304.04959)    [**[slides]**](https://drive.google.com/file/d/1I8XpxPxwhP9KXuztEguYkuMM10kiJDS7/view?usp=sharing)

## Models

This repository implements the following models:

+ AdaTT [[Paper]](https://doi.org/10.1145/3580305.3599769)
+ MMoE [[Paper]](https://dl.acm.org/doi/10.1145/3219819.3220007)
+ Multi-level MMoE (an extension of MMoE)
+ PLE [[Paper]](https://doi.org/10.1145/3383313.3412236)
+ Cross-stitch [[Paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf)
+ Shared-bottom [[Paper]](https://link.springer.com/article/10.1023/a:1007379606734)

To facilitate the integration and selection of these models, we have implemented a class called `CentralTaskArch`.

## License

AdaTT is MIT-licensed.

## Citation

If you find AdaTT's paper or code helpful, please consider citing:
```
@article{li2023adatt,
  title={AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations},
  author={Li, Danwei and Zhang, Zhengyu and Yuan, Siyang and Gao, Mingze and Zhang, Weilin and Yang, Chaofei and Liu, Xi and Yang, Jiyan},
  journal={arXiv preprint arXiv:2304.04959},
  year={2023}
}
```
