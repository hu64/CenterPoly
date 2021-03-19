from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class UADETRAC1ON10_b(data.Dataset):
    num_classes = 1
    # default_resolution = [512, 512]
    default_resolution = [512, 1024]

    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(UADETRAC1ON10_b, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))

        if split == 'test':
            self.annot_path = os.path.join(base_dir, 'COCO-format/test-1-on-200_b.json')
        elif split == 'val':
            self.annot_path = os.path.join(base_dir, 'COCO-format/val-1-on-10_b.json')
        else:
            self.annot_path = os.path.join(base_dir, 'COCO-format/train-1-on-10_b.json')

        self.max_objs = 128
        # self.max_objs = 1
        self.class_name = [
            '__background__', 'object']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing UA-Detrac {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def convert_polygon_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    score = bbox[4]
                    depth = bbox[-1]
                    label = self.class_name[cls_ind]
                    polygon = list(map(self._to_float, bbox[5:-1]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "polygon": polygon,
                        "score": float("{:.2f}".format(score)),
                        "depth": float(depth),
                    }
                    detections.append(detection)
        return detections

    def save_results(self, results, save_dir):
        if self.opt.task == 'polydet':
            json.dump(self.convert_polygon_eval_format(results),
                      open('{}/results.json'.format(save_dir), 'w'))
        else:
            json.dump(self.convert_eval_format(results),
                      open('{}/results.json'.format(save_dir), 'w'))

    def bb_intersection_fraction(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # return the intersection over union value
        return interArea / ((boxB[2]-boxB[0]) * (boxB[3]-boxB[1]))

    def is_inside(self, boxA, boxB):
        return boxB[0] > boxA[0] and boxB[2] < boxA[2] and boxB[1] > boxA[1] and boxB[3] < boxA[3]

    def clear_ignrs_from_results(self, all_bboxes):
        # if all_bboxes is None:
        #     return all_bboxes
        ignrs_dir = '/store/datasets/UA-Detrac/DETRAC-Test-Det/evaluation/igrs'
        anno = json.load(open(self.annot_path))
        id_to_seq = {}
        id_to_file = {}
        seq_to_ignrs = {}
        for image in anno['images']:
            seq = os.path.dirname(image['file_name']).split('/')[-1]
            id_to_seq[image['id']] = seq
            id_to_file[image['id']] = image['file_name']
            if seq in seq_to_ignrs:
                continue
            ignrs_lines = open(os.path.join(ignrs_dir, seq + '_IgR.txt'))
            ignMask = np.zeros((540, 960))
            for line in ignrs_lines:
                box = [int(float(item)) for item in line.split(',')]
                ignMask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1.0
                # ignMask[box[0]:box[0] + box[2], box[1]:box[1] + box[3]] = 1
                # ignBoxes = ignMask.copy()

            for i in range(1, ignMask.shape[0]):
                ignMask[i, 0] += ignMask[i-1, 0]
            for j in range(1, ignMask.shape[1]):
                ignMask[0, j] += ignMask[0, j-1]
            for i in range(1, ignMask.shape[0]):
                for j in range(1, ignMask.shape[1]):
                    ignMask[i, j] = ignMask[i, j] + ignMask[i-1, j] + ignMask[i, j-1] - ignMask[i-1, j-1]

            # import cv2
            # ignImg = (((ignMask - np.min(ignMask)) / np.max(ignMask))*255).astype(np.uint8)
            # cv2.imshow(seq, ignBoxes)
            # cv2.imshow(seq, ignImg)
            # cv2.waitKey()
            seq_to_ignrs[seq] = ignMask

        for image_id in all_bboxes:
            seq_name = id_to_seq[image_id]
            ignoreMask = seq_to_ignrs[seq_name]
            for cls_ind in all_bboxes[image_id]:
                keep = []
                # bbox_mask = np.zeros((540, 960))
                # for index in range(all_bboxes[image_id][cls_ind].shape[0]):
                #     bbox = all_bboxes[image_id][cls_ind][index]
                #     if bbox[4] > 0.7:
                #          bbox_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
                # import cv2
                # cv2.imshow(id_to_file[image_id], bbox_mask)
                # cv2.waitKey()
                for index in range(all_bboxes[image_id][cls_ind].shape[0]):
                    x0, y0, x1, y1 = [int(item) for item in all_bboxes[image_id][cls_ind][index][0:4]]
                    # x0 *= int(960 / 1024)
                    # x1 *= int(960 / 1024)
                    # y0 *= int(540 / 640)
                    # y1 *= int(540 / 640)
                    tl = ignoreMask[min(540-1, y0), min(960-1, x0)]
                    tr = ignoreMask[min(540-1, y0), min(960-1, x1)]
                    bl = ignoreMask[min(540-1, y1), min(960-1, x0)]
                    br = ignoreMask[min(540-1, y1), min(960-1, x1)]
                    ignoreValue = tl + br - tr - bl
                    if ignoreValue / ((y1-y0) * (x1-x0)) < 0.5:
                        keep.append(index)
                    else:
                        all_bboxes[image_id][cls_ind][index][4] = 0
                # all_bboxes[image_id][cls_ind] = all_bboxes[image_id][cls_ind][keep]

        return all_bboxes

    def fix_res(self, all_bboxes):
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                for index in range(all_bboxes[image_id][cls_ind].shape[0]):
                    all_bboxes[image_id][cls_ind][index][0] = np.clip(all_bboxes[image_id][cls_ind][index][0], 0, 960)
                    all_bboxes[image_id][cls_ind][index][2] = np.clip(all_bboxes[image_id][cls_ind][index][2], 0, 960)
                    all_bboxes[image_id][cls_ind][index][1] = np.clip(all_bboxes[image_id][cls_ind][index][1], 0, 540)
                    all_bboxes[image_id][cls_ind][index][3] = np.clip(all_bboxes[image_id][cls_ind][index][3], 0, 540)
        return all_bboxes

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        # results = self.clear_ignrs_from_results(results)
        # results = self.fix_res(results)
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
