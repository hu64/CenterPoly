#!/usr/bin/env bash

cd src

# python main.py polydet --val_intervals 4 --exp_id hourglass_32pts --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 4 --test --exp_id hourglass_32pts --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --load_model /store/dev/CenterPoly/exp/cityscapes/polydet/hourglass_32pts/model_best.pth
# python test.py polydet --test --exp_id hg_32pts --dataset cityscapes --nbr_points 32 --arch hourglass --keep_res --load_model /store/dev/CenterPoly/exp/cityscapes/polydet/hg_32pts/model_best.pth


# python main.py polydet --val_intervals 4 --exp_id res_50_64pts --poly_weight 1 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch res_50  --batch_size 8 --master_batch 4 --lr 2e-4

# python main.py polydet --val_intervals 4 --exp_id hourglass_64pts_nowh --poly_weight 1 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth

# python main.py polydet --val_intervals 4 --exp_id hourglass_64pts_nowh_WeSu --poly_weight 1 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 4 --exp_id hourglass_64pts_nowh_WeSu --poly_weight 1 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --resume
# python main.py polydet --val_intervals 4 --exp_id hourglass_64pts_pw10_WeSu --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth

# python main.py polydet --val_intervals 4 --exp_id hourglass_64pts_pw5_lr1e4_WeSu --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 4 --exp_id hourglass_64pts_pw5_lr1e4_WeSu --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --resume

python main.py polydet --val_intervals 4 --exp_id test2 --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 4 --exp_id test2 --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --resume