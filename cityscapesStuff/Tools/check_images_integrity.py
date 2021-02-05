import cv2
import glob

sets = 'train', 'val', 'test'

for data_set in sets:
    for filename in sorted(glob.glob('/store/datasets/cityscapes/leftImg8bit/' + data_set + '/*/*.png', recursive=True)):
        image = cv2.imread(filename)
        if image is None:
            print(filename)




