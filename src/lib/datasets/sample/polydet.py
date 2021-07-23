from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_ellipse_gaussian
from utils.image import draw_dense_reg
import math
import bresenham
from PIL import Image, ImageDraw


def find_first_non_zero_pixel(points, instance_image):
  points = list(points)
  coord = points[0]

  for pixel in points:
    pixel = list(pixel)
    pixel[0] = np.clip(pixel[0], 0, instance_image.shape[1]-1)
    pixel[1] = np.clip(pixel[1], 0, instance_image.shape[0]-1)
    coord = pixel

    if instance_image[pixel[1], pixel[0]] > 0:
      break
  return coord


def find_points_from_box(box, n_points):
  assert n_points % 4 == 0, "n_points should be a multiple of four"  # simpler this way
  x0, y0, x1, y1 = box
  nbr_points = int(n_points/4)
  x_interval = (x1 - x0) / nbr_points
  y_interval = (y1 - y0) / nbr_points
  points = []
  for i in range(nbr_points):
    points.append((round(x0 + i * x_interval), y0))
  for i in range(nbr_points):
    points.append((x1, round(y0 + i * y_interval)))
  for i in range(nbr_points):
    points.append((round(x1 - i * x_interval), y1))
  for i in range(nbr_points):
    points.append((x0, round(y1 - i * y_interval)))
  return points

class PolydetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    if 'Detrac' in img_path:
      instance_path = img_path.replace('images', 'mask')
    elif 'cityscapes' in img_path:
      instance_path = img_path.replace('leftImg8bit', 'fg').replace('_fg', '_polygons')
    DRAW = False
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)

    num_objs = min(len(anns), self.max_objs)
    num_points = self.opt.nbr_points
    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    instance_img = cv2.resize(cv2.imread(instance_path, 0), (width, height))

    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        instance_img = instance_img[:, ::-1]
        c[0] =  width - c[0] - 1

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    instance_img = cv2.warpAffine(instance_img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    if DRAW:
      old_inp = inp.copy()
    inp = (inp.astype(np.float32) / 255.)

    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    instance_img = cv2.resize(instance_img, (output_w, output_h))
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    border_hm = np.zeros((1, output_h, output_w), dtype=np.float32)
    pseudo_depth = np.zeros((self.max_objs, 1), dtype=np.float32)
    poly = np.zeros((self.max_objs, num_points*2), dtype=np.float32)
    dense_poly = np.zeros((num_points*2, output_h, output_w), dtype=np.float32)
    cat_spec_poly = np.zeros((self.max_objs, num_classes * num_points*2), dtype=np.float32)
    cat_spec_mask_poly = np.zeros((self.max_objs, num_classes * num_points*2), dtype=np.uint8)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    freq_mask = np.zeros((self.max_objs), dtype=np.float32)

    fg = np.zeros((1, output_h, output_w), dtype=np.float32)
    fg[0, :, :] = instance_img
    fg[fg != 0] = 1

    if self.opt.elliptical_gt:
      draw_gaussian = draw_ellipse_gaussian
    else:
      draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])

      pseudo_depth[k] = ann['pseudo_depth']
      cls_id = int(self.cat_ids[ann['category_id']])
      cls_name = self.class_name[ann['category_id']]

      points_on_border = ann['poly']
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        for i in range(0, len(points_on_border), 2):
          points_on_border[i] = width - points_on_border[i] - 1
      for i in range(0, len(points_on_border), 2):
        points_on_border[i], points_on_border[i+1] = affine_transform([points_on_border[i], points_on_border[i+1]], trans_output)
        points_on_border[i] = np.clip(points_on_border[i], 0, output_w - 1)
        points_on_border[i+1] = np.clip(points_on_border[i+1], 0, output_h - 1)

      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius

        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        # exp
        mass_cx, mass_cy = 0, 0
        for i in range(0, len(points_on_border), 2):
          mass_cx += points_on_border[i]
          mass_cy += points_on_border[i+1]
        ct[0] = mass_cx / (len(points_on_border)/2)
        ct[1] = mass_cy / (len(points_on_border)/2)
        ct_int = ct.astype(np.int32)

        #exp
        if DRAW:
          pts = np.array(points_on_border, np.int32)
          pts = pts.reshape((-1, 1, 2))
          old_inp = cv2.resize(old_inp, (output_w, output_h))
          old_inp = cv2.polylines(old_inp, [pts], True, (0, 255, 255))
          old_inp = cv2.rectangle(old_inp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
          old_inp = cv2.circle(old_inp, tuple(ct_int), 1, (0, 255, 255), 1)

        if self.opt.elliptical_gt:
          radius_x = radius if h > w else int(radius * (w / h))
          radius_y = radius if w >= h else int(radius * (h / w))
          draw_gaussian(hm[cls_id], ct_int, radius_x, radius_y)
        else:
          draw_gaussian(hm[cls_id], ct_int, radius)

        wh[k] = 1. * w, 1. * h

        # points_on_border = np.array(points_on_border).astype(np.float32)
        # print(points_on_border)
        #exp
        points_on_box = find_points_from_box(bbox, self.opt.nbr_points)
        for i in range(0, len(points_on_border), 2):
          draw_umich_gaussian(border_hm[0], (int(points_on_border[i]), int(points_on_border[i+1])), radius)
          poly[k][i] = points_on_border[i] - ct[0]
          poly[k][i+1] = points_on_border[i+1] - ct[1]
          # poly[k][i] = points_on_border[i] - points_on_box[int(i/2)][0]
          # poly[k][i+1] = points_on_border[i+1] - points_on_box[int(i/2)][1]

          if self.opt.cat_spec_poly:
            cat_spec_poly[k][(cls_id * (num_points*2)) + i] = points_on_border[1] - ct[0]
            cat_spec_poly[k][(cls_id * (num_points*2)) + (i+1)] = points_on_border[i+1] - ct[1]
            cat_spec_mask_poly[k][(cls_id * (num_points*2)) + i: (cls_id * (num_points*2)) + (i + 2)] = 1

        # print('h: ', output_h, ' w: ', output_w, ' 0: ', np.max(poly[0::2]), ' 1: ', np.max(poly[1::2]), ' ct: ', ct)
        # poly[k][0::2] /= output_w
        # poly[k][1::2] /= output_h
        # poly[k] *= 1000
        # print('poly: ', poly[k])

        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        freq_mask[k] = self.class_frequencies[cls_name]

        if self.opt.dense_poly:
          # print('radius: ', radius)
          draw_dense_reg(dense_poly, hm.max(axis=0), ct_int, poly[k], radius)
          # print('points_on_border: ', points_on_border)
          # print('poly[k]: ', poly[k])
          # print('dense_poly: ', dense_poly[:, ct_int[1], ct_int[0]])


        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    if DRAW:
      # cv2.imwrite(os.path.join('/store/datasets/cityscapes/test_images/polygons/', img_path.replace('/', '_').replace('.jpg', '_instance.jpg')), cv2.resize(instance_img, (input_w, input_h)))
      cv2.imwrite(os.path.join('/store/datasets/cityscapes/test_images/polygons/', img_path.replace('/', '_')), cv2.resize(old_inp,  (input_w, input_h)))

    if np.count_nonzero(freq_mask) == 0:
      freq_mean = 1.0  # don't boost loss if no objects
    else:
      freq_mean = np.sum(freq_mask) / (np.count_nonzero(freq_mask))
      # freq_mean = np.clip(freq_mean, 0.1, 1)

    # pseudo_depth /= self.opt.K
    # print('x: ', np.mean(np.abs(poly[0::2])), 'y: ', np.mean(np.abs(poly[1::2])))

    if self.opt.cat_spec_poly:
      ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'poly': poly, 'cat_spec_poly': cat_spec_poly, 'cat_spec_mask': cat_spec_mask_poly, 'pseudo_depth':pseudo_depth}
    else:
      ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'poly': poly, 'pseudo_depth':pseudo_depth, 'freq_mask':freq_mean, 'border_hm': border_hm, 'wh':wh, 'fg':fg}

    if self.opt.dense_poly:
      # hm_a = hm.max(axis=0, keepdims=True)
      # hm_a = hm.max(axis=0, keepdims=True)
      # hm_a[hm_a != 0] = 1
      dense_poly_mask = dense_poly.copy()
      dense_poly_mask[dense_poly_mask != 0] = 1
      # print(hm_a.sum())
      # print('hm_a: ', hm_a)
      # print('hm_a.shape: ', hm_a.shape)
      # print('hm_a.sum: ', hm_a.sum())
      # dense_poly_mask = np.concatenate([hm_a]*2*self.opt.nbr_points, axis=0)
      # print('dense_poly.shape: ', dense_poly.shape)
      ret.update({'dense_poly': dense_poly, 'dense_poly_mask': dense_poly_mask})
      del ret['poly']
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id,'out_width':input_w, 'out_height': input_h}
      ret['meta'] = meta
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    return ret
