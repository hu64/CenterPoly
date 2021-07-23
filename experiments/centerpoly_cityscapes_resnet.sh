#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python main.py polydet --val_intervals 12 --exp_id from_coco_16_resnetdcn101_l1_no_fg --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch resdcn_101  --batch_size 7 --lr 2e-4 --load_model ../models/ctdet_coco_resdcn101.pth

# python main.py polydet --val_intervals 12 --exp_id from101_res50_nofg --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch resdcn_50  --batch_size 10 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_coco_16_resnetdcn101_l1_no_fg/model_best.pth
# python main.py polydet --val_intervals 12 --exp_id from101_res50_nofg --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch resdcn_50  --batch_size 10 --lr 2e-4 --resume
# python test.py --nms polydet --exp_id from101_res50_nofg --nbr_points 16 --dataset cityscapes --arch resdcn_50 --load_model ../exp/cityscapes/polydet/from101_res50_nofg/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id fromct_res18_nofg --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch resdcn_18  --batch_size 28 --lr 2e-4 --load_model ../models/ctdet_coco_resdcn18.pth
# python test.py polydet --exp_id fromct_res18_nofg --nbr_points 16 --dataset cityscapes --arch resdcn_18 --load_model ../exp/cityscapes/polydet/fromct_res18_nofg/model_best.pth
