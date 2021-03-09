#!/usr/bin/env bash

cd src

# python main.py polydet --val_intervals 1 --exp_id newgt_pw10_lr2e4 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/cat_spec_poly_newgt_pw5_lr2e4/model_best.pth
# python main.py polydet --val_intervals 1 --exp_id newgt_pw10_lr2e4 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python main.py --test polydet --val_intervals 1 --exp_id newgt_pw10_lr2e4_test --poly_weight 50 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
# python test.py polydet --exp_id newgt_pw10_lr2e4_test --nbr_points 32 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
# python main.py --test polydet --val_intervals 1 --exp_id oracle_test_new_gt --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
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
# python main.py polydet --val_intervals 12 --exp_id shg_32_pw10 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 12 --exp_id shg_32_pw10 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume

# python main.py polydet --test --eval_oracle_hm --eval_oracle_offset --eval_oracle_poly --eval_oracle_pseudo_depth --val_intervals 1 --exp_id shg_32_pw10_ri_test --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth
# python main.py polydet --val_intervals 1 --exp_id shg_32_pw10_ri --poly_weight 10 --elliptical_gt --nbr_points 64 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python test.py polydet --exp_id shg_32_pw10_ri --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4  --load_model ../exp/cityscapes/polydet/shg_32_pw10_ri/model_last.pth

# python main.py polydet --val_intervals 6 --exp_id shg_32_pw10_ri_512_512 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 8 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 6 --exp_id shg_32_pw10_ri_512_512 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 8 --master_batch 4 --lr 2e-4 --resume
# python main.py polydet --test --eval_oracle_poly --eval_oracle_offset --eval_oracle_pseudo_depth --val_intervals 6 --exp_id shg_32_pw10_ri_512_512_test --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 8 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/shg_32_pw10_ri_512_512/model_best.pth
# python test.py polydet --keep_res --exp_id shg_32_pw10_ri_512_512_test --nbr_points 32 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/shg_32_pw10_ri_512_512/model_best.pth

# python main.py polydet --val_intervals 6 --exp_id shg_32_pw10_ri_512_512_resume --poly_weight 0.1 --hm_weight 5 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 8 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/shg_32_pw10_ri_512_512/model_best.pth

# python main.py polydet --val_intervals 6 --exp_id hg_32_pw10_rp_512_512 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch hourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 6 --exp_id hg_32_pw10_ri_512_512_p6 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 12 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/shg_32_pw10_ri_512_512/model_best.pth
# python main.py polydet --val_intervals 6 --exp_id hg_32_pw10_ri_512_512_p6 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 12 --master_batch 4 --lr 2e-4 --resume
# python main.py polydet --test --val_intervals 6 --exp_id hg_32_pw10_ri_512_512_p6 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 12 --master_batch 4 --lr 2e-4 --resume
# python test.py polydet --exp_id hg_32_pw10_ri_512_512_p6 --nbr_points 32 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/hg_32_pw10_ri_512_512_p6/model_best.pth

# python main.py polydet --val_intervals 1 --exp_id hg_32_pw10_ri_512_512_p7 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 12 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/shg_32_pw10_ri_512_512/model_best.pth
# python test.py --keep_res polydet --exp_id hg_32_pw10_ri_512_512_p7 --nbr_points 32 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/hg_32_pw10_ri_512_512_p7/model_best.pth

# python main.py polydet --val_intervals 1 --exp_id from_best_resume --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-5 --load_model ../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id resnet18_32pts --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch res_18  --batch_size 16 --master_batch 4 --lr 2e-4
# python main.py polydet --val_intervals 24 --exp_id resnet18_32pts_2 --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch res_18  --batch_size 16 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/resnet18_32pts/model_best.pth
# python test.py polydet --exp_id resnet18_32pts_2 --nbr_points 32 --dataset cityscapes --arch res_18  --batch_size 16 --load_model ../exp/cityscapes/polydet/resnet18_32pts_2/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id shg_pw1_ri_cat_spec --cat_spec_poly --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py polydet --val_intervals 24 --exp_id shg_pw1_ri_cat_spec --cat_spec_poly --poly_weight 1 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python test.py polydet --exp_id shg_pw1_ri_cat_spec --cat_spec_poly --nbr_points 32 --dataset cityscapes --arch smallhourglass  --load_model ../exp/cityscapes/polydet/shg_pw1_ri_cat_spec/model_last.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_3cnv --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet/model_best.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_3cnv_2 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_3cnv/model_best.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_3cnv_2 --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python test.py polydet --exp_id from_ctdet_smhg_3cnv_2_test --nbr_points 32 --dataset cityscapes --arch smallhourglass  --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_3cnv_2/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_3cnv_print_depth --poly_weight 10 --elliptical_gt --nbr_points 32 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_3cnv/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16 --poly_weight 10 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet/model_best.pth

# python test.py polydet --exp_id from_ctdet_smhg_1cnv_16 --nms --nbr_points 16 --dataset cityscapes --arch smallhourglass  --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1 --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16 --poly_weight 10 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch resnet_50  --batch_size 12 --master_batch 4 --lr 2e-4

# python main.py polydet --val_intervals 24 --exp_id from_coco_ext_poly_ext_depth --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth  # ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16/model_best.pth
# python test.py polydet --exp_id from_coco_ext_poly_ext_depth --nms --nbr_points 16 --dataset cityscapes --arch smallhourglass  --load_model ../exp/cityscapes/polydet/from_coco_ext_poly_ext_depth/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_coco_dla --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch dlav0_34  --batch_size 8 --master_batch 4 --lr 2e-4
# python main.py polydet --val_intervals 24 --exp_id from_coco_dla --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch dlav0_34  --batch_size 8 --master_batch 4 --lr 2e-4 --resume
# python test.py --nms polydet --exp_id from_coco_dla --nbr_points 16 --dataset cityscapes --arch dlav0_34 --load_model ../exp/cityscapes/polydet/from_coco_dla/model_best.pth

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_frz --poly_weight 10 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet/model_best.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_frz_unet --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet/model_best.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_unet --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet/model_best.pth
# python main.py polydet --val_intervals 24 --exp_id from_ctdet_unet --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume
# python test.py --nms polydet --exp_id from_ctdet_unet --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_unet/model_best.pth
python main.py polydet --val_intervals 24 --exp_id frz_coco_except_heads --poly_weight 1 --elliptical_gt --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth