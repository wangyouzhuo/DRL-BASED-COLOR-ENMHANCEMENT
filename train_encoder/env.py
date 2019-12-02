#####################
# env for encoder
#####################

import numpy as np
import os
import random
from utils.load_image import *
from utils.retouch_op import *
from utils.train_op import *
import time
import utils.action 
import skimage.color as color 

from scipy.misc import imread, imresize


class Environment(object):

    def __init__(self, target_path):
        # self.dirty_path = dirty_path  # 脏数据
        self.target_path = target_path  # 干净数据

        self.target_name_list = os.listdir(self.target_path) # list 
        random.shuffle(self.target_name_list) # 将images打乱

        self.batch_size = 64
        self.action_size = 12

    def load_image(self, offset):
        '''
        加载batch_size个images
        input:指向当前image的指针
        return:batch_sise 个image_np 
        shape: [batch_size, 224, 224, 3]
        '''
        images_np = []
        img_list = self.target_name_list[offset:offset * self.batch_size]
        for path in img_list:
            img_path = os.path.join(self.target_path, path)
            images_np.append(imresize(imread(img_path, mode='RGB'), (224, 224))/255.0)

        return np.array(np.stack(images_np, axis=0))

    def take_action(self, images_np):
        '''
        image_np shape: [batch_size, 224, 224, 3], 已经归一化, 0~1
        images_np是原图，相当于target_image
        dirty_image是对原图执行action后的图，相当于对其进行失真操作
        action_list 是action 的 index

        return 
        target_image, raw_image, action_list

        或者直接使用lab颜色空间
        '''
        action_list = []
        target_image = []
        mse_scores = []
        raw_image = images_np
        for image in images_np:
            action_index = random.randint(0, 11)
            action_list.append(action_index)
            tgt_img = action.take_action(image, action_index)
            target_image.append(tgt_img)

            # mse
            dirty_lab = color.rgb2lab(tgt_img)
            target_lab = color.rgb2lab(image)
            mse = np.sqrt(np.sum((target_lab - dirty_lab)**2, axis=2)).mean()/10.0
            mse_scores.append(mse)

        return np.array(np.stack(target_image, axis=0)), np.array(np.stack(raw_image, axis=0)), action_list, mse_scores 





    def show(self):
        show_image(self.current_image)

    def show_target(self):
        show_image(self.target_image)

    def get_action_trajectory(self):
        return  self.action_trajectory

    def get_color_feature(self,image):
        channel_one,channel_two,channel_three = distribution_color(image)
        result = np.concatenate([channel_one,channel_two,channel_three],axis=0).reshape([1,-1])[0]
        # print("color_feature:",result)
        return result

    def get_gray_feature(self,image):
        result = distribution_gray(image).reshape([1,-1])[0]
        # print("gray_feature:",result)
        return result

    def save_env_image(self,success,epi):
        time_current = time.strftime("[%Y-%m-%d-%H-%M-%S]", time.localtime(time.time()))
        if success:
            save_image(self.current_image, filepath=result_save_path +str(epi)+'_'+'Success_'+ str(time_current) + '_No:'+ self.img_name )
        else:
            save_image(self.current_image, filepath=result_save_path +str(epi)+'_'+'False_'  + str(time_current) + '_No:'+ self.img_name )




if __name__ == "__main__":
       env = Environment(target_path=root + 'data/source_data/',source_path=root + 'data/train_data/')
       env.reset()