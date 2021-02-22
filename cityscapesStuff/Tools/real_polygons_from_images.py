import os
import glob
import json
import csv
import numpy as np
import math
from PIL import Image
import cv2
from PIL import Image, ImageDraw


def mask_to_polygon(mask):
    contour = cv2.findContours(mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for c in contour[1:-1]:
        for point in c[0]:
            points.append(list(point[0]))
    return points

NBR_POINTS = 32

sets = 'train', 'val'
have_instances = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
id_to_label = {24:'person', 25:'rider', 26:'car', 27:'truck', 28:'bus',29:'noinstance', 30:'noinstance', 31:'train', 32:'motorcycle', 33:'bicycle'}

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

for data_set in sets:
    spamwriter = csv.writer(open('../BBoxes/' + data_set + str(NBR_POINTS) + 'pts_NoOverlap.csv', 'w'), delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for filename in sorted(glob.glob('/store/datasets/cityscapes/leftImg8bit/' + data_set + '/*/*.png', recursive=True)):
        gt_path = filename.replace('leftImg8bit', 'gtFine').replace('.png', '_polygons.json')
        # img_path = filename.replace('gtFine', 'leftImg8bit').replace('json', 'png').replace('_polygons', '')
        data = json.load(open(gt_path))
        objects = data['objects']
        objects.reverse()
        remove_mask = np.ones((1024, 2048))
        for object in objects:
            label = object['label']
            if label in have_instances:
                instance_image = Image.new('L', (2048, 1024), 0)
                points = []
                for point in object['polygon']:
                    points.append(tuple(point))
                ImageDraw.Draw(instance_image).polygon(points, outline=0, fill=255)
                instance_image *= remove_mask
                remove_mask[np.array(instance_image) == 255] = 0
                if np.sum(instance_image) <= 255:
                    continue
                polygon = mask_to_polygon(instance_image)

                if len(polygon) <= 1:
                    print(polygon, label, np.sum(instance_image)/255)
                    continue

                x0, y0, x1, y1 = polygon_to_box(polygon)
                items = [os.path.abspath(filename), x0, y0, x1, y1, label]

                while len(polygon) > NBR_POINTS:
                    distances = []
                    for i in range(1, len(polygon)):
                        distances.append(get_distance(polygon[i - 1], polygon[i]))
                    min_index = np.argsort(distances)[0]
                    del polygon[min_index]

                while len(polygon) < NBR_POINTS:
                    distances = []
                    for i in range(1, len(polygon)):
                        distances.append(get_distance(polygon[i-1], polygon[i]))
                    max_index = np.argsort(distances)[-1]
                    new_point = get_mid_point(polygon[max_index], polygon[max_index+1])
                    polygon.insert(max_index+1, new_point)

                for point in np.array(polygon).flatten():
                    items.append(point)
                spamwriter.writerow(tuple(items))

