from PIL import Image, ImageEnhance
import skimage.color as color
import numpy as np
import time
action_size = 12
import math
from action_set import *
#from action_set_tf import *
def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def take_action(image_np, action_idx):
    # 选择要执行的动作，返回执行操作后的图像
    #image_pil = Image.fromarray(np.uint8(image_np))
    #image_pil = Image.fromarray(np.uint8((image_np+0.5)*255))
    # enhance contrast
    return_np = None
    # contrast 
    if action_idx == 0:# 调整对比度
        return_np = contrast(image_np+0.5, 0.95)
    elif action_idx == 1:
        return_np = contrast(image_np+0.5, 1.05)
    # enhance color
    elif action_idx == 2:
        return_np = color_saturation(image_np+0.5, 0.95)
    elif action_idx == 3:
        return_np = color_saturation(image_np+0.5, 1.05)
    # color brightness
    elif action_idx == 4:
        return_np = brightness(image_np+0.5, 0.93)
    elif action_idx == 5:
        return_np = brightness(image_np+0.5, 1.07)
    # color temperature : http://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop
    elif action_idx == 6:
        r,g,b = 240, 240, 255 # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np+0.5, r,g,b)
    elif action_idx == 7:
        r,g,b = 270, 270, 255 # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np+0.5, r,g,b)
    elif action_idx == 8:
        r,g,b = 255, 240, 240 # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np+0.5, r,g,b)
    elif action_idx == 9:
        r,g,b = 255, 270, 270 # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np+0.5, r,g,b)
    elif action_idx == 10:
        r,g,b = 240, 255, 240 # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np+0.5, r,g,b)
    elif action_idx == 11:
        r,g,b = 270, 255, 270 # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np+0.5, r,g,b)
    
    else:
        print "error"
    return return_np-0.5
