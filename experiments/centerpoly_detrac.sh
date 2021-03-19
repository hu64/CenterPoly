#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python test.py --nms polydet --exp_id polydet --dataset uadetrac1on10_b --arch smallhourglass --load_model  ../exp/uadetrac1on10_b/ctdet/ctdet/model_best.pth  # ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth #
# python test.py --nms polydet --exp_id from_cityscapes --dataset uadetrac1on10_b --arch smallhourglass --load_model  ../exp/uadetrac1on10_b/polydet/from_cityscapes/model_last.pth  # ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth #

# python main.py polydet --val_intervals 24 --exp_id from_cityscapes --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset uadetrac1on10_b --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python main.py polydet --val_intervals 1 --exp_id from_cityscapes --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset uadetrac1on10_b --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
python test.py --nms polydet --exp_id from_cityscapes --dataset uadetrac1on10_b --arch smallhourglass --load_model  ../exp/uadetrac1on10_b/polydet/from_cityscapes/model_best.pth  # ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth #