from io import BytesIO

import numpy as np
import skimage as sk
from PIL import Image, ImageOps
from skimage import color

'''
    PIL resize (W,H)
    cv2 image is BGR
    PIL image is RGB
'''


class Contrast:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = [0.4, .3, .2, .1, .05]
        c = [.6, .5, .4]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = np.asarray(img) / 255.
        means = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - means) * c + means, 0, 1) * 255

        return Image.fromarray(img.astype(np.uint8))


class Brightness:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # W, H = img.size
        # c = [.1, .2, .3, .4, .5]
        c = [.1, .15, .05]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = sk.color.rgb2hsv(img)
        img[:, :, 2] = np.clip(img[:, :, 2] + c, 0, 1)
        img = sk.color.hsv2rgb(img)

        # if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = np.clip(img, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img
        # if isgray:
        # if isgray:
        #    img = color.rgb2gray(img)

        # return Image.fromarray(img.astype(np.uint8))


class JpegCompression:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = [25, 18, 15, 10, 7]
        c = [25, 18, 15]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        output = BytesIO()
        img.save(output, 'JPEG', quality=c)
        return Image.open(output)


class Pixelate:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        # c = [0.6, 0.5, 0.4, 0.3, 0.25]
        c = [0.6, 0.5, 0.4]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = img.resize((int(w * c), int(h * c)), Image.BOX)
        return img.resize((w, h), Image.BOX)

if __name__ == "__main__":
    print("-----")
    from PIL import Image 
    import random

    img = Image.open("img5.jpg").convert("RGB")

    cont = Contrast()
    bri = Brightness()
    jpeg = JpegCompression()
    pix = Pixelate()
    
    #param 
    margin = 1
    prob = 1

    for i in range(0,100):
        margin = random.randint(0,2)
        augmented_img_1 = bri(img, margin, prob)
        augmented_img_1.save("./result/aug_img{}.png".format(i))
    # using Contrast, Brightness