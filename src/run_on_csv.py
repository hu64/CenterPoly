import sys
import os
CENTERNET_PATH = '/store/dev/CenterPoly/src/lib/' if os.path.exists('/store/dev/CenterPoly/src/lib/') \
    else '/home/travail/huper/dev/CenterPoly/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts
import os
import cv2
import numpy as np
from PIL import Image

# class_names = ['__background__', 'bus', 'car', 'others', 'van']
# class_names = ['__background__', 'object']
class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
TASK = 'polydet'
# TASK = 'ctdet'

base_dir = os.path.join('/store/dev/CenterPoly/exp/cityscapes/', TASK)
exp_id = 'from_ctdet_smhg_1cnv_16'
model_name = 'model_best.pth'
MODEL_PATH = os.path.join(base_dir, exp_id, model_name)
opt = opts().init('{} --load_model {} --arch smallhourglass --nbr_points 16 --dataset cityscapes'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
DATASET_DIR = '/store/datasets/'

SPLIT = 'test'

if SPLIT == 'test':
    source_lines = open(os.path.join(DATASET_DIR, 'UA-Detrac/test-tf-all.csv'), 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'ua-test.csv'), 'w')
elif SPLIT == 'train1on10':
    source_lines = open(os.path.join(DATASET_DIR, 'train-tf.csv'), 'r').readlines() # + open('/store/datasets/UA-Detrac/val-tf-all.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'ua-train1on10.csv'), 'w')
elif SPLIT == 'trainval':
    source_lines = open(os.path.join(DATASET_DIR, 'train-tf-all.csv'), 'r').readlines() + open('/store/datasets/UA-Detrac/val-tf-all.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'ua-trainval.csv'), 'w')
elif SPLIT == 'uav-test':
    source_lines = open('/store/datasets/UAV/val.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'uav-test.csv'), 'w')
elif SPLIT == 'changedetection':
    source_lines = open('/store/datasets/changedetection/changedetection.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'changedetection.csv'), 'w')
elif SPLIT == 'ped1':
    source_lines = open('/store/datasets/ped1/csv.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'results.csv'), 'w')
elif SPLIT == 'cityscapes_val':
    source_lines = open('../cityscapesStuff/BBoxes/val.csv.csv', 'r').readlines()
    target_file = open(os.path.join(base_dir, exp_id, 'val.csv'), 'w')

images = [item.split(',')[0] for item in source_lines]
images = set(images)

n_images = len(images)
for count, img in enumerate(sorted(list(images))):
    if count % 100 == 0:
        print("Progress: %f%%   \r" % (100*(count/n_images)))
        sys.stdout.write("Progress: %f%%   \r" % (100*(count/n_images)))
        sys.stdout.flush()

    img_path = img.strip()
    im = np.array(Image.open(img_path))
    height, width, channel = im.shape

    # runRet = detector.run(img.strip())
    runRet = detector.run(im)
    ret = runRet['results']
    boxes = []
    for label in range(1, len(class_names)+1):
        for i, det in enumerate(ret[label]):
            box = [int(item) for item in det[:4]]
            score = det[4]
            poly = [int(item) for item in det[5:-1]]
            det = [img.strip()] + box + [class_names[label-1]] + [score] + poly
            print(str(det)[1:-1].translate(str.maketrans('', '', '\' ')), file=target_file)






