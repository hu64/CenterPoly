#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python main.py ctdet --val_intervals 2 --exp_id ctdet --elliptical_gt --dataset uadetrac1on10_b --arch smallhourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python test.py --nms ctdet --exp_id ctdet --dataset uadetrac1on10_b --arch smallhourglass --load_model ../exp/uadetrac1on10_b/ctdet/ctdet/model_best.pth

python test.py --nms ctdet --exp_id ctdet --dataset uadetrac1on10_b --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth# ../exp/uadetrac1on10_b/ctdet/ctdet/model_best.pth
