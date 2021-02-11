import cv2
import glob
import numpy as np

sets = 'train', 'val', 'test'
b_mean, g_mean, r_mean,  = [], [], []
b_std, g_std, r_std,  = [], [], []
count = 0
for data_set in sets:
    for filename in sorted(glob.glob('/store/datasets/cityscapes/leftImg8bit/' + data_set + '/*/*.png', recursive=True)):
        image = cv2.imread(filename)
        b_mean.append(np.mean(image[:, :, 0]/255))
        g_mean.append(np.mean(image[:, :, 1]/255))
        r_mean.append(np.mean(image[:, :, 2]/255))
        b_std.append(np.std(image[:, :, 0] / 255))
        g_std.append(np.std(image[:, :, 1] / 255))
        r_std.append(np.std(image[:, :, 2] / 255))

print('b mean, std: ', np.mean(b_mean), ', ', np.std(b_std))
print('g mean, std: ', np.mean(g_mean), ', ', np.std(g_std))
print('r mean, std: ', np.mean(r_mean), ', ', np.std(r_std))

# b mean, std:  0.2816898182839038 ,  0.04269893084955519
# g mean, std:  0.32266921542410754 ,  0.04088212241688149
# r mean, std:  0.28404999637454165 ,  0.04230349568017417