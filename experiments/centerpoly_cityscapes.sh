#!/usr/bin/env bash

cd src

python main.py polydet --val_intervals 1 --exp_id hg_32pts --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth