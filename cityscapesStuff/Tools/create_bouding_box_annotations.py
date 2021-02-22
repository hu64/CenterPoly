import os
import glob
import json
import csv
import numpy as np
import math

COARSE = False
NBR_POINTS = 64
# from cityscapes scripts, thee labels have instances:
have_instances = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


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

max_objects_per_img = 0
if COARSE:
    sets = 'train', 'val'
else:
    sets = 'train', 'val', 'test'

max_point_per_polygon = 0
for data_set in sets:
    if COARSE:
        spamwriter = csv.writer(open('../BBoxes/' + data_set + '_coarse.csv', 'w'), delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    elif 'test' in data_set:
        spamwriter = csv.writer(open('../BBoxes/' + data_set + '.csv', 'w'), delimiter=',',quotechar='', quoting=csv.QUOTE_NONE)
    else:
        spamwriter = csv.writer(open('../BBoxes/' + data_set + str(NBR_POINTS) + 'pts.csv', 'w'), delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for filename in sorted(glob.glob('/store/datasets/cityscapes/leftImg8bit/' + data_set + '/*/*.png', recursive=True)):
        if COARSE:
            gt_path = filename.replace('leftImg8bit', 'gtCoarse').replace('.png', '_polygons.json')
        else:
            gt_path = filename.replace('leftImg8bit', 'gtFine').replace('.png', '_polygons.json')
        # img_path = filename.replace('gtFine', 'leftImg8bit').replace('json', 'png').replace('_polygons', '')
        data = json.load(open(gt_path))
        objects = data['objects']
        count = 0
        for object in objects:
            label = object['label']
            if label in have_instances:
                x0, y0, x1, y1 = polygon_to_box(object['polygon'])
                items = [os.path.abspath(filename), x0, y0, x1, y1, label]

                # if len(object['polygon']) > NBR_POINTS:
                #     angles = []
                #     for i, point in enumerate(object['polygon']):
                #         last = i+1 if i+1 < len(object['polygon']) else 0
                #         angles.append(get_angle(object['polygon'][i-1], object['polygon'][i], object['polygon'][last]))
                #     to_remove = np.argsort(angles)[NBR_POINTS:]
                #     for index in sorted(to_remove, reverse=True):
                #         del object['polygon'][index]

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

                if len(object['polygon']) > max_point_per_polygon:
                    max_point_per_polygon = len(object['polygon'])
                for point in np.array(object['polygon']).flatten():
                    items.append(point)
                spamwriter.writerow(tuple(items))
                count += 1
        if count > max_objects_per_img:
            max_objects_per_img = count
        if count == 0:
            spamwriter.writerow((os.path.abspath(filename), -1, -1, -1, -1, 'no_object'))
        if data_set == 'test':
            spamwriter.writerow((os.path.abspath(filename), 0, 0, 1, 1, 'car'))

print('max objects: ', max_objects_per_img)
print('max nbr points: ', max_point_per_polygon)