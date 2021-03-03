#!/usr/bin/env bash

cd src

python main.py ctdet --test --eval_oracle_hm --eval_oracle_offset --eval_oracle_wh --val_intervals 2 --exp_id ctdet --dataset coco --arch smallhourglass  --batch_size 6 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth