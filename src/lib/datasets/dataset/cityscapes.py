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


def write_mask_image(args):
    polygon, mask_path = args
    poly_points = []
    for i in range(0, len(polygon) - 1, 2):
        poly_points.append((int(polygon[i]), int(polygon[i + 1])))
    polygon_mask = Image.new('L', (2048, 1024), 0)
    ImageDraw.Draw(polygon_mask).polygon(poly_points, outline=0, fill=255)
    polygon_mask.save(mask_path)


class CITYSCAPES(data.Dataset):
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
            self.annot_path = os.path.join(base_dir, 'val' + str(self.opt.nbr_points) + '_regular_interval.json')
            # self.annot_path = os.path.join(base_dir, 'val' + str(self.opt.nbr_points) + '_real_points.json')
        else:
            self.annot_path = os.path.join(base_dir, 'train' + str(self.opt.nbr_points) + '_regular_interval.json')
            # self.annot_path = os.path.join(base_dir, 'train' + str(self.opt.nbr_points) + '_real_points.json')

        self.max_objs = 128
        self.class_name = [
            '__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        self.label_to_id = {'person':24, 'rider':25, 'car':26, 'truck':27, 'bus':28, 'train':31, 'motorcycle':32, 'bicycle':33}
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
                        polys = list(map(self._to_float, bbox[5:]))
                        detection["polygons"] = polys
                    detections.append(detection)
        return detections

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
                    polygon = list(map(self._to_float, bbox[5:]))

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

        for image_id in all_bboxes:
            image_name = id_to_file[int(image_id)]
            text_file = open(os.path.join(save_dir, os.path.basename(image_name).replace('.png', '.txt')), 'w')
            count = 0
            for cls_ind in all_bboxes[image_id]:
                param_list = []
                to_remove_mask = Image.new('L', (2048, 1024), 1)
                for bbox in all_bboxes[image_id][cls_ind]:
                    if bbox[4] > 0.05:
                        score = str(bbox[4])
                        depth = bbox[-1]
                        label = self.class_name[cls_ind]
                        polygon = list(map(self._to_float, bbox[5:]))
                        # poly_points = []
                        # for i in range(0, len(polygon)-1, 2):
                        #     poly_points.append((int(polygon[i]), int(polygon[i+1])))
                        # polygon_mask = Image.new('L', (2048, 1024), 0)
                        # ImageDraw.Draw(polygon_mask).polygon(poly_points, outline=0, fill=255)
                        mask_path = os.path.join(masks_dir, os.path.basename(image_name).replace('.png', '_' + str(count) + '.png'))
                        # polygon_mask.save(mask_path)
                        text_file.write('masks/' + os.path.basename(mask_path) + ' ' + str(self.label_to_id[label]) + ' ' + score + '\n')
                        count += 1
                        param_list.append((polygon, mask_path, float(bbox[0]), depth))

                for args in sorted(param_list, key=lambda x: x[-1]):
                    polygon, mask_path, score, depth = args
                    poly_points = []
                    for i in range(0, len(polygon) - 1, 2):
                        poly_points.append((int(polygon[i]), int(polygon[i + 1])))
                    polygon_mask = Image.new('L', (2048, 1024), 0)
                    ImageDraw.Draw(polygon_mask).polygon(poly_points, outline=0, fill=255)
                    polygon_mask = Image.fromarray(np.array(polygon_mask) * np.array(to_remove_mask))
                    if score >= 0.5:
                        ImageDraw.Draw(to_remove_mask).polygon(poly_points, outline=0, fill=0)
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

# os.environ['CITYSCAPES_DATASET'] = '/store/datasets/cityscapes'
# os.environ['CITYSCAPES_RESULTS'] = '/usagers2/huper/dev/CenterPoly/exp/cityscapes/polydet/oracle_test_new_gt/results'
# from datasets.evaluation.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
# AP = evalInstanceLevelSemanticLabeling.getAP()