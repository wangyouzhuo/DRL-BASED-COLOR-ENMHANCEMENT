#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading 
import numpy as np 
import tensorflow as tf 
from env import Environment 
import os 
from model_of_encoder import Encoder_Network 

class Worker:
    def __init__(self):
        self.epoch      = 0
        self.num_thread = 5   
        self.batch_size = 64  
        self.coord      = tf.train.Coordinator()
        self.sess       = tf.Session()
        self.img_path   = 'path'
        self.img_offset = 0
        self.lr         = 0.0001
        self.opt        = tf.train.AdamOptimizer(self.lr)
        self.num_epochs     = 30

        self.global_params = self._build_global_params_dict()
        # self.max_step_in_episode = 100000

        target_img_list = os.listdir(self.img_path)
        total_img_numbers = len(target_img_list)
        self.train_batches_per_epoch = int(np.floor(total_img_numbers/self.batch_size))

##################
##################
    def _build_global_params_dict(self):
        '''
        定义全局weight，不同的线程更新同一个weight
        '''


        return params 


    def average(self, datas):
        data_sum = 0
        for item in datas:
            data_sum += item

        mean_data = data_sum * 1.0 / len(datas)
        return mean_data 

    def work(self, encoder_network):
        while not self.coord.should_stop() and self.epoch < self.num_epochs:

            step = 0 
            env = Environment(self.img_path)

            # 开始一个episode
            while step < self.train_batches_per_epoch:
                
                images_np = env.load_image(self.img_offset)
                self.img_offset += 1 
                raw_img, target_img, act_list, mse = env.take_action(images_np)   

                encoder_network.update_state_predicter(raw_img, act_list, target_img, self.lr)
                encoder_network.update_action_predicter(raw_img, act_list, target_img, self.lr)

                # 缺少自动编码器的Loss

                encoder_network.pull_all_params()

                step += 1 

            self.epoch += 1 


    def work_threader(self):
        
        global_net = Encoder_Network('Global',sess=sess) 
        encoder_network = Encoder_Network('local', sess, global_net=global_net)
        self.sess.run(tf.global_variables_initializer)
        worker_threader = []
        # 开启多线程
        for i in range(self.num_thread):
            t = threading.Thread(target=self.work, args=(encoder_network))
            t.start()
            worker_threader.append(t)

        self.coord.join(worker_threader)


if __name__ == '__main__':
    pass 