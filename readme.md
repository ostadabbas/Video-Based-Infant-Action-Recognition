# Infant Action Recognition

This is the official code repository for our paper:

 Hatamimajoumerd E, Daneshvar KP, Huang X, Luan L, Amraee Somaieh, Ostadabbas S. Challenges in Video-Based Infant Action Recognition: A Critical Examination of the State of the Art, *WACV Workshop on omputer Vision with Small Data:
A Focus on Infants and Endangered Animals, 2024*.


Contact: [Elaheh Hatamimajoumerd](e.hatamimajoumerd@neu.edu), [Sarah Ostadabbas](ostadabbas@ece.neu.edu)

### Table of contents
1. [Introduction](#introduction)
2. [InfActPrimitive Dataset](#infact)
3. [Environment](#environment)
4. [Instructions](#instructions)
5. [Citation](#citation)
6. [Acknowledgment](#acknowledgement)
7. [License](#license)

## Introduction
<a name="introduction"></a>
In this work, we present a specific pipeline for end-to-end skeleton-based infant action recognition (illustrated below),conducting experiments  on the InfActPrimitive dataset using state-of-the-art skeleton-based action recognition models. These experiments provide a benchmark for evaluating the performance of infant action recognition algorithms.



## InfActPrimitive Dataset
<a name="InfActPrimitive"></a>
Preprocessed  infant 2D and 3D skeleton data can be downloaded 
[here](https://drive.google.com/file/d/1TiuTul5b5XtJgKZeOCnrAH8WKmxb6Rld/view?usp=sharing)


## Environment
<a name="environment"></a>
The code was developed using Python 3.7 on Ubuntu 18.04.

Please install dependencies:
   ```
   pip install -r requirements.txt
   ```


## Instructions
<a name="instructions"></a>
### Data Preparation
#### NTU
- To process NTU RGB-D skeleton, download NTU60 from "[their github repository](https://github.com/shahroudy/NTURGB-D),
- extract the files into datasets folder, 
- And run datasets/ntu_55_5_split.py to preprocess the dataset"
   ```  
### Training
- use `main.py` script to train the model An examples of how to use this script is show bellow
- to use a simpler model with less encoding layers, use `--load_simple=True`
- You can adjust the number of the weights in spatial convolutions with `--num_enc_channels=True`, the default value is 64


To train the model on 55 classes of NTU, run the following code

```
python main.py --half=True --batch_size=128 --test_batch_size=128 \
    --step 90 100 --num_epoch=110 --n_heads=3 --num_worker=4 --k=1 \
    --dataset=ntu_55_split --num_class=55 --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 \
    --use_vel=False --datacase=NTU120_CSub_jlim --weight_decay=0.0005 \
    --num_person=2 --num_point=12 --graph=graph.ntu_rgb_d_12joints.Graph --feeder=feeders.feeder_ntu.Feeder
```

### Fine-tuning

- to finetune the model, you can traing the model again, but with pretrained weights.
- to load pretrained weights from a previous experiment, use the `--weights=<weights_dir>`
- to finetune a model with lower learning rate, use the `--base-lr=<value>` argument. The recommended value is `1e-3`
- to finetune a model with frozen weights, use the `--freeze_encoder=True` argument

```
python main.py --half=True --batch_size=128 --test_batch_size=128 \
    --step 90 100 --num_epoch=110 --n_heads=3 --num_worker=4 --k=1 \
    --dataset=ntu_5_split --num_class=5 --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 \
    --use_vel=False --datacase=NTU120_CSub_jlim --weight_decay=0.0005 \
    --num_person=2 --num_point=12 --graph=graph.ntu_rgb_d_12joints.Graph --feeder=feeders.feeder_ntu.Feeder\
    --weights=<weights_dir> --freeze_encoder=True
```

### Inference

```
python main.py --half=True --test_batch_size=128 \
    --n_heads=3 --num_worker=4     --k=1 --dataset=ntu_5_split \
    --num_class=5 --use_vel=False --datacase=NTU120_CSub_jlim  \
    --num_person=2 --num_point=12 --graph=graph.ntu_rgb_d_12joints.Graph --feeder=feeders.feeder_ntu.Feeder\
    --phase=test --save_score=True --weights=<weights_dir>
```


## Citation
<a name="citation"></a>

If you use our code or models in your research, please cite with:
```
@inproceedings{huang2023infAct,
  title={{Challenges in Video-Based Infant Action Recognition: A Critical Examination of the State of the Art}},
  author={ Hatamimajoumerd, Elaheh,DaneshvarKakhaki, Pooria and Huang, Xiaofei and  Luan, Lingfei and Somaieh Amraee  and Ostadabbas, Sarah},
  booktitle={WACV Workshop on omputer Vision with Small Data:
A Focus on Infants and Endangered Animals(WACVW)},
  month={1},
  year={2024}
}
```

## Acknowledgement
<a name="acknowledgement"></a>
This reporitory is using the following repos:

[InfoGCN: Representation Learning for Human Skeleton-based Action Recognition](https://openaccess.thecvf.com/content/CVPR2022/html/Chi_InfoGCN_Representation_Learning_for_Human_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html)

[MMAction open source toolbox](https://github.com/open-mmlab/mmaction)

## License 
<a name="license"></a>
This code is for non-commercial purpose only. 

By downloading or using any of the datasets provided by the ACLab, you are agreeing to the “Non-commercial Purposes” condition. “Non-commercial Purposes” means research, teaching, scientific publication and personal experimentation. Non-commercial Purposes include use of the Dataset to perform benchmarking for purposes of academic or applied research publication. Non-commercial Purposes does not include purposes primarily intended for or directed towards commercial advantage or monetary compensation, or purposes intended for or directed towards litigation, licensing, or enforcement, even in part. These datasets are provided as-is, are experimental in nature, and not intended for use by, with, or for the diagnosis of human subjects for incorporation into a product.

For further inquiry please contact: Augmented Cognition Lab at Northeastern University: [http://www.northeastern.edu/ostadabbas/ 
](http://www.northeastern.edu/ostadabbas/).






