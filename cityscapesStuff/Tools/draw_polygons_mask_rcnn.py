import os
import json
import torchvision
from PIL import Image
from torchvision import transforms as T
import cv2
import numpy as np

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
cityscapes_labels = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()


def get_prediction(img, threshold):
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t_temp = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_t_temp) == 0:
        return [], [], []
    pred_t = pred_t_temp[-1]
    masks = (pred[0]['masks'] > 0.5).squeeze(axis=1).detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class


base_dir = '/usagers2/huper/dev/CenterPoly/cityscapesStuff'
images_dir = 'leftImg8bit'
edges_dir = 'polygons_maskrcnn'
csv_files = ['BBoxes/train.json', 'BBoxes/val.json']

for csv_file in csv_files:
    data = json.load(open(os.path.join(base_dir, csv_file)))
    for image in data['images']:
        image_path = image['file_name']
        mask_path = image_path.replace(images_dir, edges_dir)
        if os.path.exists(mask_path):
            continue
        if not os.path.exists(os.path.dirname(mask_path)):
            os.mkdir(os.path.dirname(mask_path))
        img = Image.open(image_path)
        mask_img = np.zeros((img.size[1], img.size[0]))
        masks, pred_boxes, pred_class = get_prediction(img, threshold=0.25)
        start_pixel_value = 255
        for i, mask in enumerate(masks):
            if pred_class[i] in cityscapes_labels:
                mask_img[mask == True] = start_pixel_value
                start_pixel_value -= 1
        cv2.imwrite(mask_path, mask_img)
        cv2.waitKey()




