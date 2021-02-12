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


def find_first_different_pixel(points, edge_image):
  points = list(points)
  coord = points[0]
  center_value = edge_image[coord[1], coord[0]]

  for i, pixel in enumerate(points):
    coord = pixel
    if edge_image[pixel[1], pixel[0]] != center_value:
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
      edge_path = img_path.replace('images', 'mask')
    elif 'cityscapes' in img_path:
      edge_path = img_path.replace('leftImg8bit', 'polygons_maskrcnn')

    DRAW = False
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)

    num_objs = min(len(anns), self.max_objs)
    num_points = self.opt.nbr_points
    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    edge_img = cv2.resize(cv2.imread(edge_path, 0), (width, height))

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
        edge_img = edge_img[:, ::-1]
        c[0] =  width - c[0] - 1

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    edge_img = cv2.warpAffine(edge_img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)

    if DRAW:
      old_inp = inp.copy()
    inp = (inp.astype(np.float32) / 255.)

    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)


    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    edge_img = cv2.resize(edge_img, (output_w, output_h))
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    poly = np.zeros((self.max_objs, num_points*2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    size_norm = np.zeros((self.max_objs, 1), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

    if self.opt.elliptical_gt:
      draw_gaussian = draw_ellipse_gaussian
    else:
      draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
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

        wh[k] = 1. * w, 1. * h

        points_on_box = find_points_from_box(box=bbox, n_points=num_points)
        points_on_border = []
        for point_on_box in points_on_box:
          line = bresenham.bresenham(int(ct[0]), int(ct[1]), int(point_on_box[0]), int(point_on_box[1]))
          points_on_border.append(find_first_different_pixel(line, edge_img))
        mass_cx, mass_cy = 0, 0
        for point_on_border in points_on_border:
          mass_cx += point_on_border[0]
          mass_cy += point_on_border[1]
        ct[0] = mass_cx/len(points_on_border)
        ct[1] = mass_cy / len(points_on_border)
        ct_int = ct.astype(int)

        if DRAW:
          pts = np.array(points_on_border, np.int32)
          pts = pts.reshape((-1, 1, 2))
          old_inp = cv2.resize(old_inp, (output_w, output_h))
          old_inp = cv2.polylines(old_inp, [pts], True, (0, 255, 255))
          old_inp = cv2.rectangle(old_inp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))

        if self.opt.elliptical_gt:
          radius_x = radius if h > w else int(radius * (w / h))
          radius_y = radius if w >= h else int(radius * (h / w))
          draw_gaussian(hm[cls_id], ct_int, radius_x, radius_y)
        else:
          draw_gaussian(hm[cls_id], ct_int, radius)

        def PolyArea(x, y):
          return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        xs = []
        ys = []
        for i, point_on_border in enumerate(points_on_border):
          poly[k][i*2] = point_on_border[0] - ct[0]
          xs.append(point_on_border[0])
          poly[k][i*2 + 1] = point_on_border[1] - ct[1]
          ys.append(point_on_border[1])
        size_norm[k] = PolyArea(xs, ys)


        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1



    if DRAW:
      cv2.imwrite(os.path.join('/store/datasets/cityscapes/test_images/polygons/', img_path.replace('/', '_').replace('.jpg', '_edge.jpg')), cv2.resize(edge_img, (input_w, input_h)))
      cv2.imwrite(os.path.join('/store/datasets/cityscapes/test_images/polygons/', img_path.replace('/', '_')), cv2.resize(old_inp,  (input_w, input_h)))

    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'poly': poly, 'size_norm': size_norm}

    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      meta = {'c': c, 's': s, 'img_id': img_id}
      ret['meta'] = meta
    return ret
