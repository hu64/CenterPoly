import os
import glob
import json
import csv
from PIL import Image, ImageDraw
import cv2
import numpy as np

# from cityscapes scripts, thee labels have instances:
have_instances = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
sets = 'train', 'val'
for data_set in sets:
    for filename in sorted(glob.glob('/store/datasets/cityscapes/gtFine/' + data_set + '/*/*.json', recursive=True)):
        img_path = filename.replace('gtFine', 'leftImg8bit').replace('json', 'png').replace('_polygons', '')
        # print(img_path)
        w, h, c = cv2.imread(img_path).shape
        # """
        polygon_mask = Image.new('L', (h, w), 0)
        data = json.load(open(filename))
        objects = data['objects']
        for object in objects:
            label = object['label']
            if label in have_instances:
                polygon = [tuple(item) for item in object['polygon']]
                ImageDraw.Draw(polygon_mask).polygon(polygon, outline=255, fill=255)
        polygon_path = filename.replace('gtFine', 'fg').replace('json', 'png').replace('_fg', '')
        if not os.path.exists((os.path.dirname(polygon_path))):
            os.mkdir(os.path.dirname(polygon_path))
        cv2.imwrite(polygon_path, np.array(polygon_mask))
        # """


