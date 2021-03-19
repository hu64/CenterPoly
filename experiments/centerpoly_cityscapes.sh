#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src


# python main.py polydet --val_intervals 24 --exp_id resnet18_32pts --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch res_18  --batch_size 16 --master_batch 4 --lr 2e-4
# python main.py polydet --val_intervals 24 --exp_id resnet18_32pts_2 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch res_18  --batch_size 16 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/resnet18_32pts/model_best.pth
# python test.py polydet --exp_id resnet18_32pts_2 --nbr_points 32 --dataset cityscapes --arch res_18  --batch_size 16 --load_model ../exp/cityscapes/polydet/resnet18_32pts_2/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16 --poly_weight 10 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet/model_best.pth

# python test.py polydet --exp_id from_ctdet_smhg_1cnv_16 --nms --nbr_points 16 --dataset cityscapes --arch smallhourglass  --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth  # ../exp/cityscapes/ctdet/ctdet/model_best.pth  #
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1 --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16 --poly_weight 10 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch resnet_50  --batch_size 12 --master_batch 4 --lr 2e-4

# python main.py polydet --val_intervals 24 --exp_id from_coco_ext_poly_ext_depth --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth  # ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python test.py polydet --exp_id from_coco_ext_poly_ext_depth --nms --nbr_points 16 --dataset cityscapes --arch smallhourglass  --load_model ../exp/cityscapes/polydet/from_coco_ext_poly_ext_depth/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_coco_dla --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch dlav0_34  --batch_size 8 --master_batch 4 --lr 2e-4
# python main.py polydet --val_intervals 24 --exp_id from_coco_dla --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch dlav0_34  --batch_size 8 --master_batch 4 --lr 2e-4 --resume
# python test.py --nms polydet --exp_id from_coco_dla --nbr_points 16 --dataset cityscapes --arch dlav0_34 --load_model ../exp/cityscapes/polydet/from_coco_dla/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1 --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth

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