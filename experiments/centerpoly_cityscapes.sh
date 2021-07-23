#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# Main Results Stuff
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_B --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_B --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python main.py polydet --test --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_B --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth
# python test.py polydet --exp_id from_ctdet_smhg_1cnv_16_pw1_B_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth
# --eval_oracle_border_hm --eval_oracle_poly

# Ablation Study stuff
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_32_pw1 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_32_pw1 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python test.py --nms polydet --exp_id from_ctdet_smhg_1cnv_32_pw1 --nbr_points 32 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_32_pw1/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_64_pw1 --poly_weight 1 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python test.py --nms polydet --exp_id from_ctdet_smhg_1cnv_64_pw1 --nbr_points 64 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_64_pw1/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_8_pw1 --poly_weight 1 --elliptical_gt --nbr_points 8 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python test.py --nms polydet --exp_id from_ctdet_smhg_1cnv_8_pw1 --nbr_points 8 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_8_pw1/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_no_ell --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_no_ell --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_no_cg --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_no_cg --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python test.py polydet --exp_id from_ctdet_smhg_1cnv_16_pw1_no_cg --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_no_cg/model_best.pth
