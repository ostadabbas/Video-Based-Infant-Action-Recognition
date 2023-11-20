This reporitory is using "[InfoGCN: Representation Learning for Human Skeleton-based Action Recognition](https://openaccess.thecvf.com/content/CVPR2022/html/Chi_InfoGCN_Representation_Learning_for_Human_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html)", CVPR22. 

## Managing conda environment

The required package for running this code are listed in requirements.txt

## Data preprocessing 

### NTU
- To process NTU RGB-D skeleton, download NTU60 from "[their github repository](https://github.com/shahroudy/NTURGB-D),
- extract the files into datasets folder, 
- And run datasets/ntu_55_5_split.py to preprocess the dataset"

## Training
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

## Fine-tuning

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

## Inference

```
python main.py --half=True --test_batch_size=128 \
    --n_heads=3 --num_worker=4     --k=1 --dataset=ntu_5_split \
    --num_class=5 --use_vel=False --datacase=NTU120_CSub_jlim  \
    --num_person=2 --num_point=12 --graph=graph.ntu_rgb_d_12joints.Graph --feeder=feeders.feeder_ntu.Feeder\
    --phase=test --save_score=True --weights=<weights_dir>
```
