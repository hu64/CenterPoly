#!/usr/bin/env bash

cd src

python main.py polydet --val_intervals 5 --exp_id hourglass_32pts --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python test.py polydet --test --exp_id hg_32pts --dataset cityscapes --nbr_points 32 --arch hourglass --keep_res --load_model /store/dev/CenterPoly/exp/cityscapes/polydet/hg_32pts/model_best.pth
