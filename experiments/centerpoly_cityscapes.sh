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
# python main.py polydet --val_intervals 4 --test --exp_id hourglass_64pts_pw5_lr1e4_WeSu --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --resume
# python test.py polydet --test --exp_id hourglass_64pts_pw5_lr1e4_WeSu --dataset cityscapes --nbr_points 64 --arch hourglass --keep_res --load_model /store/dev/CenterPoly/exp/cityscapes/polydet/hourglass_64pts_pw5_lr1e4_WeSu/model_best.pth

# python main.py polydet --val_intervals 4 --exp_id test2 --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 4 --exp_id test2 --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --resume

# python main.py polydet --val_intervals 8 --exp_id small_hg_draw --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 8 --exp_id small_hg --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 1e-4 --resume
# python test.py polydet --test --exp_id small_hg --dataset cityscapes --nbr_points 64 --arch smallhourglass --keep_res --load_model /store/dev/CenterPoly/exp/cityscapes/polydet/small_hg/model_best.pth

# python main.py polydet --val_intervals 12 --exp_id dla_34 --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch dla_34  --batch_size 10 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_dla_2x.pth
# python main.py polydet --val_intervals 12 --exp_id dla_34 --poly_weight 5 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch dla_34  --batch_size 8 --master_batch 4 --lr 1e-4 --resume

# python main.py polydet --val_intervals 12 --exp_id hg_64pts_lossNorm_WeSu --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 12 --exp_id hg_64pts_lossNorm_WeSu --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --resume

# python main.py polydet --val_intervals 12 --exp_id hg_64pts_lossNormOn1K_WeSu --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 12 --exp_id hg_64pts_lossNormOn1K_WeSu --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --resume

# python main.py polydet --val_intervals 12 --exp_id hg_64pts_lossNormWithin_WeSu --poly_weight 100 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth

# python main.py polydet --val_intervals 12 --exp_id hg_64pts_lossNormMulOn1K_WeSu --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 12 --exp_id hg_64pts_lossNormMulOn1K_WeSu --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch hourglass  --batch_size 2 --master_batch 4 --lr 1e-4 --resume

python main.py polydet --val_intervals 12 --exp_id hg_64pts_lossKL_WeSu --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 1e-4 --load_model ../models/ctdet_coco_hg.pth