import torch
from torchvision import transforms
from PIL import Image
import random
import os
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import random
from torchvision import transforms
from torchvision.transforms import functional as F

def transform_JPEGcompression(image, compress_range = (10, 80)):
    '''
        Perform random JPEG Compression
    '''
    assert compress_range[0] < compress_range[1], "Lower and higher value not accepted: {} vs {}".format(compress_range[0], compress_range[1])
    jpegcompress_value = random.randint(compress_range[0], compress_range[1])
    out = BytesIO()
    image.save(out, 'JPEG', quality=jpegcompress_value)
    out.seek(0)
    rgb_image = Image.open(out)
    return rgb_image


def transform_gaussian_noise(img, mean = 0.0, var = 30.0):
    '''
        Perform random gaussian noise
    '''
    img = np.array(img)
    height, width, channels = img.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma,(height, width, channels))
    noisy = img + gauss
    cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy = noisy.astype(np.uint8)
    return Image.fromarray(noisy)


def _motion_blur(img, kernel_size):
    # Specify the kernel size. 
    # The greater the size, the more the motion. 
    # Create the vertical kernel. 
    kernel_v = np.zeros((kernel_size, kernel_size)) 
    # Create a copy of the same for creating the horizontal kernel. 
    kernel_h = np.copy(kernel_v) 
    # Fill the middle row with ones. 
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
    # Normalize. 
    kernel_v /= kernel_size 
    kernel_h /= kernel_size 
    if np.random.uniform() > 0.5:
        # Apply the vertical kernel. 
        blurred = cv2.filter2D(img, -1, kernel_v) 
    else:
        # Apply the horizontal kernel. 
        blurred = cv2.filter2D(img, -1, kernel_h) 
    return blurred

def _unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def _increase_contrast(img, kernel_size):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=np.random.uniform(0.001, 4.0), tileGridSize=(kernel_size,kernel_size))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def transform_random_blur(img):
    img = np.array(img)
    flag = np.random.uniform()
    kernel_size = random.choice([3, 5, 7, 9, 11])
    if flag >= 0.75:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), np.random.uniform(0.0, 2.0))
    elif flag >= 0.5:
        img = _motion_blur(img, kernel_size)
    elif flag >= 0.4:
        img = cv2.blur(img, (kernel_size, kernel_size))
    elif flag >= 0.2:
        img = _unsharp_mask(img, kernel_size = kernel_size)
    elif flag >= 0.0:
        img = _increase_contrast(img, kernel_size)
    return Image.fromarray(img)

def transform_adjust_gamma(image, lower = 0.2, upper = 2.0):
    image = np.array(image)
    gamma = np.random.uniform(lower, upper)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return Image.fromarray(cv2.LUT(image, table))

# def transform_blur(img):
#     flag = np.random.uniform()
#     kernel_size = random.choice([3, 5, 7, 9])
#     img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
#     return img

def transform_to_gray(img):
    '''
        Perform random gaussian noise
    '''
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(gray)

def transform_resize(image, resize_range = (24, 112), target_size = 112):
    assert resize_range[0] < resize_range[1], "Lower and higher value not accepted: {} vs {}".format(resize_range[0], resize_range[1])
    resize_value = random.randint(resize_range[0], resize_range[1])
    w, h = image.size
    if w < h: 
        new_w = resize_value
        new_h = h / w * resize_value 
    else: 
        new_h = resize_value 
        new_w = w / h * resize_value 
    resize_image = image.resize((int(new_w), int(new_h)), Image.BICUBIC)

    return resize_image.resize((w, h), Image.BICUBIC)


# def transform_eraser(image):
#     if np.random.uniform() < 0.1:
#         mask_range = random.randint(0, 3)
#         image_array = np.array(image, dtype=np.uint8)
#         image_array[(7-mask_range)*16:, :, :] = 0
#         return Image.fromarray(image_array)
#     else:
#         return image

def transform_color_jiter(sample, brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1):
    photometric = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            photometric.get_params(photometric.brightness, photometric.contrast,
                                                  photometric.saturation, photometric.hue)
    for fn_id in fn_idx:
        if fn_id == 0 and brightness_factor is not None:
            sample = F.adjust_brightness(sample, brightness_factor)
        elif fn_id == 1 and contrast_factor is not None:
            sample = F.adjust_contrast(sample, contrast_factor)
        elif fn_id == 2 and saturation_factor is not None:
            sample = F.adjust_saturation(sample, saturation_factor)
        elif fn_id == 3 and hue_factor is not None:
            sample = F.adjust_hue(sample, hue_factor)
    return sample




class CustomTransform:
    def __init__(self, random_gray = 0.15, random_rotation = 0.15, random_flip = 0.1, is_train = True, is_padding_with_ratio = True):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.random_gray = random_gray
        self.random_rotation = random_rotation
        self.random_flip = random_flip
        self.is_train = is_train
        self.is_padding_with_ratio = is_padding_with_ratio
    def __call__(self, image):
        # Random grayscaling with a probability of 0.5
 
        # Blur augmentation 
        if np.random.uniform() < 0.3 and self.is_train: 
            image = transform_random_blur(image) 
 
        # Downscale augmentation 
        if np.random.uniform() < 0.3 and self.is_train: 
            image = transform_resize(image, resize_range = (64, 224), target_size = 224) 
 
        # Color augmentation 
        if np.random.uniform() < 0.3 and self.is_train: 
            image = transform_adjust_gamma(image) 
 
        if np.random.uniform() < 0.3 and self.is_train: 
            image = transform_color_jiter(image) 
         
 
        # Noise augmentation 
        if np.random.uniform() < 0.15 and self.is_train: 
            image = transform_gaussian_noise(image, mean = 0.0, var = 30.0) 
 
        # Gray augmentation 
        if np.random.uniform() < 0.2 and self.is_train: 
            image = transform_to_gray(image) 
         
        # JPEG augmentation 
        if np.random.uniform() < 0.5 and self.is_train: 
            image = transform_JPEGcompression(image, compress_range = (20, 80)) 
        # Random rotation (up to ±10 degrees)
        if random.random() < self.random_rotation and self.is_train:
            angle = random.uniform(-10, 10)
            image = transforms.functional.rotate(image, angle)
    

        if random.random() < self.random_flip and self.is_train:
            image = transforms.functional.hflip(image)

        if random.random() < 0.5 and self.is_padding_with_ratio: 
            # image, mask = self.add_padding_image(image)
            image = self._resize_and_random_paste(image, (224, 224))
        else:
            image = image.resize((224,224))
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        return image
    
    def _resize_and_random_paste(self, img, size):
        """Resize image while keeping aspect ratio and paste it at the top-left position on a background."""
        img.thumbnail(size, Image.ANTIALIAS)
        
        # Create a new background of the given size
        background = Image.new('RGB', size, (0, 0, 0))
        
        # Paste image onto background at the top-left corner
        background.paste(img, (0, 0))
        
        return background

# Example usage
if __name__ == "__main__":
    from PIL import Image
    img = Image.open("/home1/data/congvu/TAO/dataset/Cropped_Data_261_Class_1604/train_3/Audi_V8_Sedan/01180.jpg").convert('RGB')
    custom_transform = CustomTransform()
    img = custom_transform(img)
    # img.save('sample.jpg')


