"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
from scipy import ndimage
import numpy as np
import copy
from skimage.transform import resize
import imageio
import cv2
import matplotlib.pyplot as plt
import pandas as pd

try:
    _imread = scipy.misc.imread
except AttributeError:
    print('ATT')
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:          # self.images(image pool) 가 self.maxsize 이상이 될 때. image pool 에서 random으로 두개(A,B) select한 다음 return. 그 자리에는 input image로 대체.
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

#def load_test_data(image_path, fine_size=256):
#    img = imread(image_path)
#    img = scipy.misc.imresize(img, [fine_size, fine_size])
#    img = img/127.5 - 1
#    return img

def load_test_data(image_path, load_size=286, fine_size=256, channel=3):   #hcw
    if channel==3:
        img = imread(image_path)
    else:
        img = imread(image_path, is_grayscale=True)
        print('imread xxxx')
    
    #img = scipy.misc.imresize(img, [load_size, load_size])
    img = resize(img, [load_size, load_size])
    h1 = int((load_size-fine_size)/2)
    w1 = int((load_size-fine_size)/2)
    img = img[h1:h1+fine_size, w1:w1+fine_size]
    img = img/127.5 - 1

    return img

def load_test_data2(image_path, load_size=286, fine_size=256, channel=3):   #hcw
    if channel==3:
        img = imread(image_path)
    else:
        img = imread(image_path, is_grayscale=True)
        print('imread xxxx')
    
    #img = scipy.misc.imresize(img, [load_size, load_size])
    img = resize(img, [load_size, load_size])
    h1 = int((load_size-fine_size)/2)
    w1 = int((load_size-fine_size)/2)
    img = img[h1-30:h1+fine_size-30, w1-30:w1+fine_size-30]
    img = img/127.5 - 1

    return img

def restore_uint(img):
    img = (img+1)*127.5
    return img

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False, channel=3):
    if channel == 3:
        img_A = imread(image_path[0])
        img_B = imread(image_path[1])
    else:
        img_A = imread(image_path[0], is_grayscale = True)
        img_B = imread(image_path[1], is_grayscale = True)
    if not is_testing:
        #img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        #img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        img_A = resize(img_A, [load_size, load_size])
        img_B = resize(img_B, [load_size, load_size])
        if load_size != fine_size:                                                  # hcw
            h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
            img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
            img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() < 0.1:        # hcw
            img_A = ndimage.gaussian_filter(img_A, sigma=1)
            img_B = ndimage.gaussian_filter(img_B, sigma=1)
        #if np.random.random() > 0.5:
        #    img_A = np.fliplr(img_A)
        #    img_B = np.fliplr(img_B)
    else:
        #img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        #img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
        img_A = resize(img_A, [fine_size, fine_size])
        img_B = resize(img_B, [fine_size, fine_size])

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    ## test
    if np.random.random() < 0.3:
        img_B = add_sp_noise(img_B, 's&p')
    if np.random.random() < 0.3:
        img_B = add_sp_noise(img_B, 'gaussian')
    if np.random.random() < 0.3:
        img_B = add_black_square(img_B)
    #

    if channel==3:
        img_AB = np.concatenate((img_A, img_B), axis=2)
    else:
        img_A = np.expand_dims(img_A, axis=2)
        img_B = np.expand_dims(img_B, axis=2)
        img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    try:
        print('try inverse transform')
        return imsave(inverse_transform(images), size, image_path)
    except:
        return imsave(images, size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        #return _imread(path, as_gray=True).astype(np.float)
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float) # hcw

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    try:
        return imageio.imwrite(path, merge(images, size))       # hcw
    except:
        return imageio.imwrite(path, images)
    #return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def hist_shift_table(img): # hcw

    target_val = 160

    binwidth = 2
    #bin_ = plt.hist(img.ravel(),bins=np.arange(0, 255 + binwidth, binwidth))
    bin_ = np.histogram(img, bins=np.linspace(0,254,int(256/2)))

    #L = np.argsort(bin_[0])
    #if np.abs(L[-1]-L[-2]) < 1000:    
    #    s_point = int((L[-1]+L[-2])/2)*binwidth
    #else:
    #    s_point = L[-1]*binwidth
    
    #s_points = [s_point-5,np.min([s_point+5, 255])]
    try:
        s_points = [int(np.min(np.where(bin_[0]>2000))*binwidth), int(np.max(np.where(bin_[0]>2000))*binwidth)]
    except:
        s_points = [int(np.min(np.where(bin_[0]>1000))*binwidth), int(np.max(np.where(bin_[0]>1000))*binwidth)]
    n = s_points[1] - s_points[0]
    
    ax = s_points[0]
    ay = target_val-n
    bx = s_points[1]
    by = target_val

    tbl1 = np.array([(ay/ax)*i for i in range(0,ax)]).astype('uint8')
    tbl2 = np.array([((by - ay)/(bx - ax))*(i-ax)+ay for i in range(ax,bx)]).astype('uint8')
    tbl3 = np.array([((256 - by)/(256 - bx))*(i-bx)+by for i in range(bx,256)]).astype('uint8')
    table =  np.concatenate((tbl1, tbl2, tbl3))
    
    return table
    #return cv2.LUT(img, table)

def add_sp_noise(image, noise_type):
    if noise_type == 's&p':
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        coords = list(zip(coords[0], coords[1]))
        for i in coords:
            out[i] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        coords = list(zip(coords[0], coords[1]))
        for i in coords:
            out[i] = -1
        #print('s&p applied')
        return out
    elif noise_type == 'gaussian':
        row,col = image.shape
        mean = 0
        #var = 0.1
        #sigma = var**0.5
        gauss = np.random.normal(mean,0.2,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        noisy = np.clip(noisy, a_min = -1, a_max = 1)
        #print('gaussian applied')
        return noisy

def add_black_square(image):
    row, col = image.shape
    w = np.random.randint(1, 50)
    h = np.random.randint(1, 50)
    x = np.random.randint(0, row-w-1)
    y = np.random.randint(0, col-h-1)
    #intensity = np.random.randint(0,100)
    #intensity = intensity/50 - 1
    #image[y:y+h, x:x+w] = intensity
    image[y:y+h, x:x+w] = -1
    #print('bs applied')
    return image


def _load_data_and_split(data_path, train_ratio): # load data and train test split.
    ds = pd.read_csv(data_path)
    ds_1 = ds.sample(frac=1)
    ds_train = ds_1.iloc[:int(len(ds_1)*train_ratio),:]
    ds_test = ds_1.iloc[int(len(ds_1)*train_ratio):,:]
        
    ds_train.to_csv(data_path[:-4]+'_'+str(train_ratio)+'_train.csv', index=False)
    ds_test.to_csv(data_path[:-4]+'_'+str(train_ratio)+'_test.csv', index=False)