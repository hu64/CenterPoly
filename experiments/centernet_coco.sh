#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python test.py ctdet --test --exp_id stdCTNET --dataset coco --arch resdcn_101 --keep_res --batch_size 1 --load_model ../models/ctdet_coco_resdcn101.pth
python test.py ctdet --test --exp_id stdCTNET-18 --dataset coco --arch resdcn_18 --keep_res --batch_size 1 --load_model ../models/ctdet_coco_resdcn18.pth