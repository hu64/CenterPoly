#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python main.py polydet --val_intervals 1 --exp_id from_cityscapes --poly_weight 1 --depth_weight 0.1 --elliptical_gt --nbr_points 16 --dataset IDD --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
python test.py polydet --nms --exp_id from_cityscapes --nbr_points 16 --dataset IDD --arch smallhourglass  --load_model ../exp/IDD/polydet/from_cityscapes/model_best.pth
