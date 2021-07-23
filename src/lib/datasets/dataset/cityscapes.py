from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import json
import os
from PIL import Image, ImageDraw, ImageChops
import torch.utils.data as data
import glob
from multiprocessing import Pool
from pycocotools.cocoeval import COCOeval
import cv2
import bresenham
from shapely.geometry import Polygon

FG = False


def write_mask_image(args):
    polygon, mask_path = args
    poly_points = []
    for i in range(0, len(polygon) - 1, 2):
        poly_points.append((int(polygon[i]), int(polygon[i + 1])))
    polygon_mask = Image.new('L', (2048, 1024), 0)
    ImageDraw.Draw(polygon_mask).polygon(poly_points, outline=0, fill=255)
    polygon_mask.save(mask_path)


class CITYSCAPES(data.Dataset):
    if FG:
        num_classes = 8
    else:
        num_classes = 8
    # default_resolution = [1024, 2048]
    default_resolution = [512, 1024]
    # default_resolution = [512, 512]

    mean = np.array([0.28404999637454165, 0.32266921542410754, 0.2816898182839038], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.04230349568017417, 0.04088212241688149, 0.04269893084955519],dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(CITYSCAPES, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        self.split = split
        self.opt = opt

        base_dir = '../cityscapesStuff/BBoxes'

        if split == 'test':
            self.annot_path = os.path.join(base_dir, 'test.json')
        elif split == 'val':
            # self.annot_path = os.path.join(base_dir, 'val' + str(self.opt.nbr_points) + '_grid_based.json')
            if FG:
                self.annot_path = os.path.join(base_dir, 'val' + str(self.opt.nbr_points) + '_regular_interval_fg3.json')
            else:
                self.annot_path = os.path.join(base_dir, 'val' + str(self.opt.nbr_points) + '_regular_interval.json')
            # self.annot_path = os.path.join(base_dir, 'val' + str(self.opt.nbr_points) + '_real_points_fg3.json')
        else:
            # self.annot_path = os.path.join(base_dir, 'train' + str(self.opt.nbr_points) + '_grid_based.json')
            if FG:
                self.annot_path = os.path.join(base_dir, 'train' + str(self.opt.nbr_points) + '_regular_interval_fg3.json')
            else:
                self.annot_path = os.path.join(base_dir, 'train' + str(self.opt.nbr_points) + '_regular_interval.json')
            # self.annot_path = os.path.join(base_dir, 'train' + str(self.opt.nbr_points) + '_real_points_fg3.json')

        self.max_objs = 128
        self.class_name = [
            '__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'pole', 'traffic sign', 'traffic light']
        self.label_to_id = {'person':24, 'rider':25, 'car':26, 'truck':27, 'bus':28, 'train':31, 'motorcycle':32, 'bicycle':33, 'pole':-1, 'traffic sign':-1, 'traffic light':-1}
        # self.class_frequencies = {'person': 0.15, 'rider': 0.03, 'car': 0.20, 'truck': 0.03, 'bus': 0.03, 'train': 0.03, 'motorcycle': 0.03, 'bicycle': 0.03, 'pole': 0.3, 'traffic sign': 0.15, 'traffic light': 0.03}
        self.class_frequencies = {'person': 0.14062428170827013, 'rider': 0.015518384984665498, 'car': 0.20898266905714155, 'truck': 0.003822132907776267, 'bus': 0.0031719762791339126, 'train': 0.0012740443025920892, 'motorcycle': 0.005831707941761728, 'bicycle': 0.0322057384531526, 'pole': 0.34640870553158515, 'traffic sign': 0.16402335310072175, 'traffic light': 0.07813700573319936}
        if FG:
            self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        else:
            self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8]
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



        print('==> initializing cityscapes {} data.'.format(split))
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

    def convert_polygon_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                if cls_ind == 'fg':
                    continue
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

    def format_and_write_to_cityscapes(self, all_bboxes, save_dir):
        id_to_file = {}
        anno = json.load(open(self.annot_path))
        for image in anno['images']:
            id_to_file[image['id']] = image['file_name']

        masks_dir = os.path.join(save_dir, 'masks')
        if not os.path.exists(masks_dir):
            os.mkdir(masks_dir)
        # fg_dir = os.path.join(save_dir, 'fg')
        # if not os.path.exists(fg_dir):
        #     os.mkdir(fg_dir)

        for image_id in all_bboxes:
            image_name = id_to_file[int(image_id)]
            text_file = open(os.path.join(save_dir, os.path.basename(image_name).replace('.png', '.txt')), 'w')
            count = 0
            ones = np.ones((1024, 2048))
            to_remove_mask = np.zeros((1024, 2048))
            param_list = []
            # fg_path = os.path.join(fg_dir, os.path.basename(image_name))
            # fg = all_bboxes[image_id]['fg']
            # fg = np.array(cv2.resize(fg, (2048, 1024)))
            # thresh = 0.5
            # fg[fg >= thresh] = 1
            # fg[fg < thresh] = 0
            # fg = fg.astype(np.uint8)
            # cv2.imwrite(fg_path, fg*255)
            for cls_ind in all_bboxes[image_id]:
                if cls_ind == 'fg':
                    continue
                for bbox in all_bboxes[image_id][cls_ind]:
                    if bbox[4] > 0.05:
                        depth = bbox[-1]
                        label = self.class_name[cls_ind]
                        polygon = list(map(self._to_float, bbox[5:-1]))
                        polygon = [(int(x), int(y)) for x, y in zip(polygon[0::2], polygon[1::2])]
                        # if label != 'pole' and label != 'traffic sign' and label != 'traffic light':
                        #     mask_path = os.path.join(masks_dir, os.path.basename(image_name).replace('.png', '_' + str(count) + '.png'))
                        #     text_file.write('masks/' + os.path.basename(mask_path) + ' ' + str(self.label_to_id[label]) + ' ' + str(bbox[4]) + '\n')
                        #     count += 1
                        param_list.append((polygon, bbox[4], label, depth))

            for args in sorted(param_list, key=lambda x: x[-1]):
                points, score, label, depth = args
                polygon_mask = Image.new('L', (2048, 1024), 0)
                if label != 'pole' and label != 'traffic sign' and label != 'traffic light':

                    # round edges
                    # try:
                    #     polygon = Polygon((points))
                    #     polygon = polygon.buffer(10, join_style=1).buffer(-10.0, join_style=1)
                    #     x, y = polygon.exterior.coords.xy
                    #     points = [(int(item[0]), int(item[1])) for item in zip(x, y)]
                    # except:
                    #     do_nothing = True

                    ImageDraw.Draw(polygon_mask).polygon(points, outline=255, fill=255)

                    # Draw contour
                    contour = list(bresenham.bresenham(points[-1][0], points[-1][1], points[0][0], points[0][1]))
                    for i in range(len(points) - 1):
                        line = bresenham.bresenham(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                        contour += line
                    radius = 2
                    for point in set(contour):
                        ImageDraw.Draw(polygon_mask).ellipse([(point[0] - radius, point[1] - radius),
                                                              (point[0] + radius, point[1] + radius)],
                                                               outline=255, fill=255)

                    # polygon_mask = Image.fromarray(np.array(polygon_mask) * np.array(fg))
                    polygon_mask = Image.fromarray(np.array(polygon_mask) * (ones - to_remove_mask).astype(np.uint8))

                if score >= 0.5:
                    to_remove_mask += np.array(polygon_mask)
                    to_remove_mask[to_remove_mask > 0] = 1
                    # ImageDraw.Draw(to_remove_mask).polygon(points, outline=0, fill=0)
                if label != 'pole' and label != 'traffic sign' and label != 'traffic light' and \
                        np.count_nonzero(polygon_mask) > 100:

                    mask_path = os.path.join(masks_dir, os.path.basename(image_name).replace('.png', '_' + str(count)
                                                                                             + '.png'))
                    text_file.write('masks/' + os.path.basename(mask_path) + ' ' + str(self.label_to_id[label])
                                    + ' ' + str(min(1, score*1.2)) + '\n')
                    count += 1
                    polygon_mask.save(mask_path)
        # with Pool(processes=4) as pool:
        #     pool.map(write_mask_image, param_list)

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        if self.opt.task == 'polydet':
            json.dump(self.convert_polygon_eval_format(results),
                      open('{}/results.json'.format(save_dir), 'w'))
        else:
            json.dump(self.convert_eval_format(results),
                      open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        if self.opt.task == 'ctdet':
            self.save_results(results, save_dir)
            coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
            coco_eval = COCOeval(self.coco, coco_dets, "bbox")
            # coco_eval.params.catIds = [2, 3, 4, 6, 7, 8, 10, 11, 12, 13]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        else:
            self.save_results(results, save_dir)
            res_dir = os.path.join(save_dir, 'results')
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            to_delete = os.path.join(save_dir, 'results/*.txt')
            files = glob.glob(to_delete)
            for f in files:
                os.remove(f)
            to_delete = os.path.join(save_dir, 'results/*/*.png')
            files = glob.glob(to_delete)
            for f in files:
                os.remove(f)
            self.format_and_write_to_cityscapes(results, res_dir)
            os.environ['CITYSCAPES_DATASET'] = '/store/datasets/cityscapes'
            os.environ['CITYSCAPES_RESULTS'] = res_dir
            from datasets.evaluation.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
            AP = evalInstanceLevelSemanticLabeling.getAP()
            return AP
            # return 0
