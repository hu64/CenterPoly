#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python main.py polydet --val_intervals 24 --exp_id from_cityscapes --poly_weight 1 --depth_weight 0 --elliptical_gt --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python test.py polydet --nms --keep_res --exp_id from_cityscapes --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/kitti_poly/polydet/from_cityscapes/model_last.pth
python test.py polydet --nms --keep_res --exp_id cityscapes_model --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth

# python main.py polydet --val_intervals 4 --exp_id from_cityscapes_pw10 --poly_weight 10 --depth_weight 0 --elliptical_gt --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --batch_size 8 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python test.py polydet --nms --keep_res --exp_id from_cityscapes_pw10 --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/kitti_poly/polydet/from_cityscapes_pw10/model_best.pth

# python main.py polydet --val_intervals 4 --exp_id from_cityscapes_pw10_32 --poly_weight 10 --depth_weight 0 --elliptical_gt --nbr_points 32 --dataset kitti_poly --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python test.py polydet --nms --keep_res --exp_id from_cityscapes_pw10_32 --nbr_points 32 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/kitti_poly/polydet/from_cityscapes_pw10_32/model_best.pth

# python main.py polydet --val_intervals 4 --exp_id from_cityscapes_pw01 --poly_weight 0.1 --depth_weight 0 --elliptical_gt --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python test.py polydet --nms --keep_res --exp_id from_cityscapes_pw01 --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/kitti_poly/polydet/from_cityscapes_pw01/model_last.pth

# python main.py polydet --val_intervals 24 --exp_id from_cityscapes_lr2e5 --poly_weight 1 --depth_weight 0 --elliptical_gt --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-5 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python test.py polydet --nms --keep_res --exp_id from_cityscapes_lr2e5 --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/kitti_poly/polydet/from_cityscapes_lr2e5/model_best.pth

# python main.py polydet --val_intervals 4 --exp_id from_cityscapes_pw100 --poly_weight 100 --depth_weight 0 --elliptical_gt --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --batch_size 8 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python test.py polydet --nms --keep_res --exp_id from_cityscapes_pw100 --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/kitti_poly/polydet/from_cityscapes_pw100/model_best.pth