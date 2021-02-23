#!/usr/bin/env bash

cd src

# python main.py polydet --val_intervals 1 --exp_id newgt_pw10_lr2e4 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/cat_spec_poly_newgt_pw5_lr2e4/model_best.pth
# python main.py polydet --val_intervals 1 --exp_id newgt_pw10_lr2e4 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python main.py --eval_oracle_hm --eval_oracle_offset --eval_oracle_poly --test polydet --val_intervals 1 --exp_id newgt_pw10_lr2e4_test --poly_weight 50 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
# python main.py --eval_oracle_hm --eval_oracle_offset --eval_oracle_poly --test polydet --val_intervals 1 --exp_id oracle_test_new_gt --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
# python main.py polydet --val_intervals 1 --exp_id newgt_pw50_lr2e4 --poly_weight 50 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth

# python main.py polydet --val_intervals 1 --exp_id pw10_lr2e4_square --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 12 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
# python main.py polydet --val_intervals 1 --exp_id pw10_lr2e4_square --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 12 --master_batch 4 --lr 2e-4 --resume


# python main.py polydet --val_intervals 1 --exp_id resnet50 --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch res_18  --batch_size 16 --master_batch 4 --lr 2e-4

# python main.py polydet --val_intervals 1 --exp_id newgt_pw20_lr2e4 --poly_weight 20 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth

# python main.py polydet --val_intervals 1 --exp_id pw10_lr2e5_freeze --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-5 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
# python main.py polydet --val_intervals 1 --exp_id pw10_lr2e5_freeze_64pts --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-5 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
# python main.py polydet --val_intervals 1 --exp_id pw10_lr2e5_freeze_64pts --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-5 --resume

# python main.py polydet --val_intervals 1 --exp_id pw10_lr2e5_freeze_reg --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-5 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth

# python main.py polydet --test --val_intervals 1 --eval_oracle_hm --eval_oracle_offset --eval_oracle_poly --eval_oracle_pseudo_depth --exp_id depth_test --poly_weight 10 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 3 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
python main.py polydet --val_intervals 1 --exp_id shg_32_pw10 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth  # ../models/ctdet_coco_hg.pth