#!/usr/bin/env bash

source /store/dev/anaconda3/etc/profile.d/conda.sh
conda activate centernet
cd src

# python main.py ctdet --val_intervals 2 --exp_id ctdet --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py ctdet --test --eval_oracle_hm --eval_oracle_offset --eval_oracle_wh --val_intervals 2 --exp_id ctdet --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet/model_last.pth
# python main.py ctdet --test --val_intervals 2 --exp_id ctdet --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet/model_last.pth
# python test.py --nms ctdet --exp_id ctdet --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/ctdet/ctdet/model_best.pth
# python main.py ctdet --val_intervals 2 --exp_id ctdet --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --resume

# python main.py ctdet --val_intervals 6 --exp_id ctdet_hg --dataset cityscapes --arch hourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python main.py ctdet --test --val_intervals 6 --exp_id ctdet_hg --dataset cityscapes --arch smallhourglass  --batch_size 1 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/ctdet/ctdet_hg/model_best.pth
# python test.py ctdet --exp_id ctdet_hg --dataset cityscapes --arch hourglass --load_model ../exp/cityscapes/ctdet/ctdet_hg/model_best.pth

# python main.py ctdet --val_intervals 2 --exp_id ctdet_cg --elliptical_gt --dataset cityscapes --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python test.py --nms ctdet --exp_id ctdet_cg --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/ctdet/ctdet_cg/model_best.pth
