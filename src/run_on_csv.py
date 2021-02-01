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
class_names = ['person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
TASK = 'polydet'
# TASK = 'ctdet'

base_dir = os.path.join('/store/dev/CenterPoly/exp/cityscapes/', TASK)
exp_id = 'mask-resdcn_50'
model_name = 'model_best.pth'
MODEL_PATH = os.path.join(base_dir, exp_id, model_name)
opt = opts().init('{} --load_model {} --arch hourglass --nbr_points 32 --dataset cityscapes --keep_res'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
DATASET_DIR = '/store/datasets/'

SPLIT = 'test'

if SPLIT == 'test':
    source_lines = open(os.path.join(DATASET_DIR, 'test-tf-all.csv'), 'r').readlines()
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

if not os.path.exists(base_seg_dir):
    os.mkdir(base_seg_dir)

images = [item.split(',')[0] for item in source_lines]
images = set(images)

n_images = len(images)
for count, img in enumerate(sorted(list(images))):
    if count % 100 == 0:
        print("Progress: %f%%   \r" % (100*(count/n_images)))
        sys.stdout.write("Progress: %f%%   \r" % (100*(count/n_images)))
        sys.stdout.flush()

    img_path = img.strip()
    # img_path = img_path if os.path.exists(img_path) else img_path.replace('/store/', '/home/travail/huper/')
    if opt.task == 'ctdetVid' or opt.task == 'ctdetSpotNetVid':
        N_FRAMES = 11
        middle = int(N_FRAMES / 2)
        index = os.path.basename(img_path).replace('.jpg', '').replace('img', '').replace('.JPEG', '')
        rest = img_path.replace(index + '.jpg', '').replace(os.path.dirname(img_path), '')
        length = len(index)
        modulo = '1'
        for i in range(length):
            modulo += '0'
        img_paths = []
        for i in range(N_FRAMES):
            new_img_path = os.path.dirname(img_path) \
                           + rest \
                           + str((int(index) - (i - middle)) % int(modulo)).zfill(length) + '.jpg'
            if not os.path.exists(new_img_path):
                new_img_path = img_path
            img_paths.append(new_img_path)
        imgs = []
        for path in img_paths:
            loaded_img = cv2.imread(path)
            # print(loaded_img.shape)
            imgs.append(loaded_img)
        im = np.concatenate(imgs, -1)
        # print(im.shape)
        height, width, channels = im.shape
    else:
        im = np.array(Image.open(img_path))
        height, width, channel = im.shape

    # runRet = detector.run(img.strip())
    runRet = detector.run(im)
    ret = runRet['results']
    boxes = []
    for label in [1]:  # , 2, 3, 4]:
        for i, det in enumerate(ret[label]):
            box = [int(item) for item in det[:4]]
            # if float(det[4]) > 0.15 and (box[2] - box[0]) * (box[2] - box[0]) < (width * height) / 2:
            #     boxes.append(box)
            if 'quads' in runRet:
                quads = [int(item) for item in runRet['quads'][0][i]]
                det = [img.strip()] + box + [class_name[label]] + [det[4]] + quads
            else:
                det = [img.strip()] + box + [class_name[label]] + [det[4]]
            print(str(det)[1:-1].translate(str.maketrans('', '', '\' ')), file=target_file)

    """
    map = np.zeros((height, width))
    for box in boxes:
        map[box[1]:box[3], box[0]:box[2]] = 1
        # map[box[2]:box[0], box[3]:box[1]] = 1

    seg = runRet['seg']
    seg_path = os.path.join(base_seg_dir, os.path.dirname(img).split('/')[-2], os.path.basename(img))
    if not os.path.exists(os.path.dirname(seg_path)):
        os.mkdir(os.path.dirname(seg_path))

    seg = np.squeeze(seg, [0, 1])
    sheight, swidth = seg.shape
    hoffset = int((sheight-height)/2)
    woffset = int((swidth-width)/2)
    seg = seg[hoffset:-hoffset, woffset:-woffset]

    seg -= np.min(seg)
    seg /= np.max(seg)
    # seg = cv2.GaussianBlur(seg, (5, 5), 0)
    # seg = seg**8
    # seg = cv2.GaussianBlur(seg, (5, 5), 0)
    # seg -= np.min(seg)
    # seg /= np.max(seg)
    # seg[seg <= 0.67] = 0
    # seg[seg > 0.67] = 1
    # seg *= map
    seg = (seg*255).astype(np.uint8)
    cv2.imwrite(os.path.dirname(seg_path) + '/map_' + os.path.basename(seg_path).replace('.jpg', '.png').strip(), (map*255).astype(np.uint8))
    cv2.imwrite(seg_path.replace('.jpg', '.png').strip(), seg)
    # seg[seg < 240] = 0
    """






