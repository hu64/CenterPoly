import os
import glob
import json
import csv


# from cityscapes scripts, thee labels have instances:
have_instances = ['person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
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

sets = 'train', 'val', 'test'
for data_set in sets:
    spamwriter = csv.writer(open('../BBoxes/' + data_set + '.csv', 'w'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for filename in sorted(glob.glob('/store/datasets/cityscapes/leftImg8bit/' + data_set + '/*/*.png', recursive=True)):
        gt_path = filename.replace('leftImg8bit', 'gtFine').replace('.png', '_polygons.json')
        # img_path = filename.replace('gtFine', 'leftImg8bit').replace('json', 'png').replace('_polygons', '')
        data = json.load(open(gt_path))
        objects = data['objects']
        count = 0
        for object in objects:
            label = object['label']
            if label in have_instances:
                x0, y0, x1, y1 = polygon_to_box(object['polygon'])
                spamwriter.writerow((os.path.abspath(filename), x0, y0, x1, y1, label))
                count += 1
        if count == 0:
            spamwriter.writerow((os.path.abspath(filename), -1, -1, -1, -1, 'no_object'))
        if data_set == 'test':
            spamwriter.writerow((os.path.abspath(filename), 0, 0, 1, 1, 'car'))
