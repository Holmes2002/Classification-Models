import torchvision.transforms as transforms
# from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageFilter
import random
import torch
import numpy as np
import logging
from enum import Enum
from augmentation_custom.warp import Distort, Stretch
from augmentation_custom.geometry import Rotate, Perspective, Shrink
from augmentation_custom.noise import GaussianNoise
from augmentation_custom.blur import GaussianBlur, DefocusBlur, GlassBlur, ZoomBlur
from augmentation_custom.camera import Contrast, Brightness


# 0: InterpolationMode.NEAREST,
# 2: InterpolationMode.BILINEAR,
# 3: InterpolationMode.BICUBIC,
# 4: InterpolationMode.BOX,
# 5: InterpolationMode.HAMMING,
# 1: InterpolationMode.LANCZOS,

class InterpolationMode():
    NEAREST = 0
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5
    LANCZOS = 1

class RandomAugData(object):
    def __init__(self, lst_aug_types: list, prob_list: list = None):
        """
        Arg:
            : prob_list: probability choice aug tyoe
            : lst_aug_type: all augmentation method
        """

        self.lst_aug = list()
        if "warp" in lst_aug_types:
            self.lst_aug.append([Distort(), Stretch()])
        if "geometry" in lst_aug_types:
            # self.lst_aug.append([Rotate(), Perspective(), Shrink()])
            self.lst_aug.append([Perspective(), Shrink()])
        if "blur" in lst_aug_types:
            # self.lst_aug.append([GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()])
            self.lst_aug.append([GaussianBlur(), DefocusBlur(), GlassBlur(), ZoomBlur()])
        if "noise" in lst_aug_types:
            self.lst_aug.append([GaussianNoise()])
        if "camera" in lst_aug_types:
            self.lst_aug.append([Contrast(), Brightness()])
        
        if prob_list is None :
            self.prob_list = [0.5] * len(self.lst_aug)
        else:
            assert len(self.lst_aug) == len(prob_list), "The length of 'prob_list' must be the same as the number of augmentations used."
            self.prob_list = prob_list

    def __call__(self, img):
        for i in range(int(len(self.lst_aug))):
            self.mag_range = np.random.randint(0, 3)
            # i = random.randint(0, len(self.lst_aug))
            # if random.random()<0.5:
            img = self.lst_aug[i][np.random.randint(0, len(self.lst_aug[i]))](img, mag = self.mag_range, prob = self.prob_list[i])

        return img     

# class RndAugData(object):
#     def __init__(self, )

if __name__ == "__main__":
    import cv2
    random_StrAug_1 = RandomAugData(lst_aug_types = ['warp', 'geometry', 'blur', 'noise', 'camera'],
                                  prob_list = [0.2, 0.2, 0.2, 0.2, 0.2])

    from PIL import Image
    img = Image.open("./img3.jpg").convert("RGB")
    for i in range(0,1000):
        
        augmented_img_1 = random_StrAug_1(img)
        # augmented_img_2 = random_StrAug_2(img)

        # Save images to compare before and after augmentation.
        # result = cv2.cvtColor(np.hstack((np.array(img), np.array(augmented_img_1), np.array(augmented_img_2))), cv2.COLOR_RGB2BGR)
        augmented_img_1.save("./result/aug_img{}.png".format(i))
        # result1 = cv2.cvtColor(np.ndarray(augmented_img_1), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join("./result",  'img1_random_strAug_1.jpg'), result1)
        # result2 = cv2.cvtColor(np.ndarray(augmented_img_2), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join("./result",  'img1_random_strAug_2.jpg'), result2)
