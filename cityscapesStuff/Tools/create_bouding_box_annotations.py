import os
import glob
import json
import csv
import numpy as np
import math
import bresenham
from PIL import Image, ImageDraw

METHODS = 'real_points', 'regular_interval'
COARSE = False
NBR_POINTSS = 16, 32, 64
# from cityscapes scripts, thee labels have instances:
have_instances = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


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
if COARSE:
    sets = 'train', 'val'
else:
    sets = 'train', 'val', 'test'

for method in METHODS:
    for NBR_POINTS in NBR_POINTSS:
        for data_set in sets:
            if COARSE:
                spamwriter = csv.writer(open('../BBoxes/' + data_set + '_coarse.csv', 'w'), delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            elif 'test' in data_set:
                spamwriter = csv.writer(open('../BBoxes/' + data_set + '.csv', 'w'), delimiter=',',quotechar='', quoting=csv.QUOTE_NONE)
            else:
                spamwriter = csv.writer(open('../BBoxes/' + data_set + str(NBR_POINTS) + '_' + method +'.csv', 'w'), delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            for filename in sorted(glob.glob('/store/datasets/cityscapes/leftImg8bit/' + data_set + '/*/*.png', recursive=True)):
                if COARSE:
                    gt_path = filename.replace('leftImg8bit', 'gtCoarse').replace('.png', '_polygons.json')
                else:
                    gt_path = filename.replace('leftImg8bit', 'gtFine').replace('.png', '_polygons.json')
                # img_path = filename.replace('gtFine', 'leftImg8bit').replace('json', 'png').replace('_polygons', '')
                data = json.load(open(gt_path))
                objects = data['objects']
                objects.reverse()
                count = 0
                for object in objects:
                    label = object['label']
                    if label in have_instances:
                        bbox = x0, y0, x1, y1 = polygon_to_box(object['polygon'])
                        items = [os.path.abspath(filename), x0, y0, x1, y1, label, count]

                        if method == 'real_points':
                            while len(object['polygon']) > NBR_POINTS:
                                distances = []
                                for i in range(1, len(object['polygon'])):
                                    distances.append(get_distance(object['polygon'][i - 1], object['polygon'][i]))
                                min_index = np.argsort(distances)[0]
                                del object['polygon'][min_index]

                            while len(object['polygon']) < NBR_POINTS:
                                distances = []
                                for i in range(1, len(object['polygon'])):
                                    distances.append(get_distance(object['polygon'][i-1], object['polygon'][i]))
                                max_index = np.argsort(distances)[-1]
                                new_point = get_mid_point(object['polygon'][max_index], object['polygon'][max_index+1])
                                object['polygon'].insert(max_index+1, new_point)

                            object['polygon'] = rotate_points(object['polygon'], bbox)

                        elif method == 'regular_interval':
                            poly_img = Image.new('L', (2048, 1024), 0)
                            ImageDraw.Draw(poly_img).polygon([tuple(item) for item in object['polygon']], outline=0, fill=255)
                            poly_img = np.array(poly_img)
                            points_on_box = find_points_from_box(box=bbox, n_points=NBR_POINTS)
                            points_on_border = []
                            ct = int((x1-x0)/2), int((y1-y0)/2)
                            for point_on_box in points_on_box:
                              line = bresenham.bresenham(int(point_on_box[0]), int(point_on_box[1]), int(ct[0]), int(ct[1]))
                              points_on_border.append(find_first_non_zero_pixel(line, poly_img))
                            object['polygon'] = points_on_border


                        if len(object['polygon']) > max_point_per_polygon:
                            max_point_per_polygon = len(object['polygon'])
                        for point in np.array(object['polygon']).flatten():
                            items.append(point)
                        spamwriter.writerow(tuple(items))
                        count += 1
                if count > max_objects_per_img:
                    max_objects_per_img = count
                if count == 0:
                    spamwriter.writerow((os.path.abspath(filename), -1, -1, -1, -1, 'no_object', 0))
                if data_set == 'test':
                    spamwriter.writerow((os.path.abspath(filename), 0, 0, 1, 1, 'car', 0))

        print('max objects: ', max_objects_per_img)
        print('max nbr points: ', max_point_per_polygon)