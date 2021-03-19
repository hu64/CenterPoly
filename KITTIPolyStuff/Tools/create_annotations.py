import os
import glob
import json
import csv
import numpy as np
import math
import bresenham
from PIL import Image, ImageDraw
import cv2

NBR_POINTSS = 16, 32, 64
# from cityscapes scripts, thee labels have instances:
have_instances = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
have_instances_labels = [24, 25, 26, 27, 28, 31, 32, 33]
label_to_id = {'person':24, 'rider':25, 'car':26, 'truck':27, 'bus':28, 'train':31, 'motorcycle':32, 'bicycle':33}
id_to_label = {v: k for k, v in label_to_id.items()}

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


def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = ang + 360 if ang < 0 else ang
    return ang if ang < 180 else 360 - ang


def get_distance(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


def get_mid_point(a, b):
    return [int((a[0]+b[0])/2), int((a[1]+b[1])/2)]


def polygon_to_box(polygon):
    x0 = x1 = polygon[0][0]
    y0 = y1 = polygon[0][1]

    for point in polygon:
        x, y = point
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
    return x0, y0, x1, y1


def rotate_points(points, bbox):
    tl_corner = bbox[0], bbox[2]
    distances = []
    for point in points:
        distances.append(get_distance(point, tl_corner))
    min_index = np.argsort(distances)[0]
    return points[min_index:] + points[:min_index]


max_objects_per_img = 0
max_point_per_polygon = 0

sets = 'train', 'test'  # , 'test'
possible_labels = set()
for NBR_POINTS in NBR_POINTSS:
    for data_set in sets:
        if 'test' in data_set:
            spamwriter = csv.writer(open('../BBoxes/' + data_set + '.csv', 'w'), delimiter=',',quotechar='', quoting=csv.QUOTE_NONE)
        else:
            spamwriter = csv.writer(open('../BBoxes/' + data_set + str(NBR_POINTS) + '.csv', 'w'), delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            trainvalwriter = csv.writer(open('../BBoxes/' + 'trainval' + str(NBR_POINTS) + '.csv', 'w'), delimiter=',',
                                    quotechar='', quoting=csv.QUOTE_NONE)
            valwriter = csv.writer(open('../BBoxes/' + 'val' + str(NBR_POINTS) + '.csv', 'w'), delimiter=',',
                                    quotechar='', quoting=csv.QUOTE_NONE)
        image_count = 0
        for filename in sorted(glob.glob('/store/datasets/KITTIPoly/' + ('training' if data_set == 'train' else 'testing') + '/image_2/*.png', recursive=True)):
            image_count += 1
            if data_set == 'test':
                spamwriter.writerow((os.path.abspath(filename), 0, 0, 1, 1, 'car', 0))
            elif data_set == 'train':
                gt_path = filename.replace('image_2', 'instance')
                # print(gt_path)
                gt_labels = cv2.imread(gt_path, 0)
                instances_mask = gt_labels.copy()
                instances_mask[instances_mask == 255] = 0
                instances_mask[instances_mask == 24] = 255
                instances_mask[instances_mask == 25] = 255
                instances_mask[instances_mask == 26] = 255
                instances_mask[instances_mask == 27] = 255
                instances_mask[instances_mask == 28] = 255
                instances_mask[instances_mask == 31] = 255
                instances_mask[instances_mask == 32] = 255
                instances_mask[instances_mask == 33] = 255
                instances_mask[instances_mask != 255] = 0
                instances_mask[instances_mask == 255] = 1

                gt_ids = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED) * instances_mask
                ids = np.unique(gt_ids)
                count = 0

                for id in ids:
                    if id == 0:
                        continue
                    id_mask = gt_ids.copy()
                    id_mask[id_mask==255] = 0
                    id_mask[id_mask==id] = 255
                    id_mask[id_mask!=255] = 0
                    # cv2.imshow('', id_mask)
                    # cv2.waitKey()
                    ys, xs = np.where(gt_ids == id)
                    bbox = x0, y0, x1, y1 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                    if False and np.sum(id_mask)/255 >= 10000:
                        cv2.imshow('', cv2.imread(filename)[y0:y1, x0:x1])
                        cv2.waitKey()
                    if gt_labels[ys[0], xs[0]] not in id_to_label:
                        continue
                    label = id_to_label[gt_labels[ys[0], xs[0]]]
                    possible_labels.add(label)
                    items = [os.path.abspath(filename), x0, y0, x1, y1, label, count]

                    points_on_box = find_points_from_box(box=bbox, n_points=NBR_POINTS)
                    points_on_border = []
                    ct = int(x0 + ((x1 - x0) / 2)), int(y0 + ((y1 - y0) / 2))
                    for point_on_box in points_on_box:
                        line = bresenham.bresenham(int(point_on_box[0]), int(point_on_box[1]), int(ct[0]), int(ct[1]))
                        points_on_border.append(find_first_non_zero_pixel(line, id_mask))
                    polygon = points_on_border
                    for point in np.array(polygon).flatten():
                        items.append(point)
                    trainvalwriter.writerow(tuple(items))
                    if image_count % 20 == 0:
                        valwriter.writerow(tuple(items))
                    else:
                        spamwriter.writerow(tuple(items))
                    count += 1
print(possible_labels)