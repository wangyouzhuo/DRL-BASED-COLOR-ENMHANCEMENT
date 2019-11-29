import tensorflow as tf
import numpy as np
import tensorflow as tf
from train_encoder.config_of_encoder import *
from operations.generate_weight import *


class Encoder_Network(object):

    def __init__(self,name):

        with tf.name_scope(name):

            self.current_image = tf.placeholder(tf.float32,[None,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNEL], 'Current_image')

            self.next_image = tf.placeholder(tf.float32,[None,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNEL], 'Next_image')

            self.action = tf.placeholder(tf.int32 , [None,],'Action_index')

            self.learning_rate = tf.placeholder(tf.float32,None,'learning_rate')

            self.OPT = tf.train.RMSPropOptimizer(self.learning_rate , name='RMSPropA')

            self._prepare_weight()

            if 'global' in name:

                self.current_feature = self._build_encoder(input=self.current_image)
                self.next_feature    = self._build_encoder(input=self.next_image)




    def _prepare_weight(self):

            # weight_for_encoder
            self.conv1_weight = generate_conv2d_weight(shape=[3,3,IMAGE_CHANNEL,8]  ,name="conv1_weight_encode")
            self.conv1_bias   = generate_conv2d_bias(shape=8            ,name='conv1_bias_encode')
            self.conv2_weight = generate_conv2d_weight(shape=[3,3,8,16] ,name="conv2_weight_encode")
            self.conv2_bias   = generate_conv2d_bias(shape=16           ,name='conv2_bias_encode')
            self.conv3_weight = generate_conv2d_weight(shape=[2,2,16,32],name="conv3_weight_encode")
            self.conv3_bias   = generate_conv2d_bias(shape=32           ,name='conv3_bias_encode')
            self.conv4_weight = generate_conv2d_weight(shape=[2,2,32,64],name="conv4_weight_encode")
            self.conv4_bias   = generate_conv2d_bias(shape=64           ,name='conv4_bias_encode')
            self.fc_weight_encoder    = generate_fc_weight(shape=[1024,1024]    ,name='fc_weight_encode')
            self.fc_bias_encoder      = generate_fc_weight(shape=[1024]         ,name='fc_bias_encode')
            self.encoder_params = [
                self.conv1_weight       ,self.conv4_bias,
                self.conv2_weight       ,self.conv2_bias,
                self.conv3_weight       ,self.conv3_bias,
                self.conv4_weight       ,self.conv4_bias,
                self.fc_weight_encoder  ,self.fc_bias_encoder
            ]


            # weight_for_decoder
            self.fc_weight_decoder = generate_fc_weight(shape=[1024,1024]   ,name='fc_weight_decode')
            self.fc_bias_decoder    = generate_fc_bias(shape=[1024]     ,name='fc_bias_decode')
            self.conv5_weight = generate_conv2d_weight(shape=[2,2,64,32] ,name='conv5_weight_encode')
            self.conv5_bias   = generate_conv2d_bias(shape=32            ,name='conv5_weight_bias')
            self.conv6_weight = generate_conv2d_weight(shape=[2,2,32,16] ,name='conv6_weight_encode')
            self.conv6_bias   = generate_conv2d_bias(shape=16            ,name='conv6_weight_bias')
            self.conv7_weight = generate_conv2d_weight(shape=[2,2,16,8]  ,name='conv7_weight_encode')
            self.conv7_bias   = generate_conv2d_bias(shape=8             ,name='conv7_weight_bias')
            self.conv8_weight = generate_conv2d_weight(shape=[2,2,8,3]   ,name='conv8_weight_encode')
            self.conv8_bias   = generate_conv2d_bias(shape=3             ,name='conv8_weight_bias')
            self.conv9_weight = generate_conv2d_weight(shape=[2,2,3,3]   ,name='conv9_weight_encode')
            self.conv9_bias   = generate_conv2d_bias(shape=3             ,name='conv9_weight_bias')
            self.decode_params = [
                self.fc_weight_decoder,self.fc_bias_decoder,
                self.conv5_weight     ,self.conv5_bias,
                self.conv6_weight     ,self.conv6_bias,
                self.conv7_weight     ,self.conv7_bias,
                self.conv8_weight     ,self.conv8_bias,
                self.conv9_weight     ,self.conv9_bias
            ]

            # weight for action_predicter
            self.a_p_weight = generate_fc_weight(shape=[1024*2,ACTION_SIZE],name='a_p_weight')
            self.a_p_bias   = generate_fc_bias(shape=[ACTION_SIZE],name='a_p_bias')




            # weight for state predicter




            # weight for state_predicter

    def _build_encoder(self,input_image):
        conv1 = tf.nn.conv2d(input_image, self.conv1_weight, strides=[1, 2, 2, 1], padding='SAME')

        conv1 = tf.layers.max_pooling2d(conv1,(2,2),(2,2),padding='same')
        conv1 = tf.nn.elu(tf.nn.bias_add(conv1, self.conv1_bias))

        conv2 = tf.nn.conv2d(conv1, self.conv2_weight, strides=[1, 2, 2, 1], padding='SAME')

        conv2 = tf.layers.max_pooling2d(conv2,(2,2),(2,2),padding='same')
        conv2 = tf.nn.elu(tf.nn.bias_add(conv2, self.conv2_bias))

        conv3 = tf.nn.conv2d(conv2, self.conv3_weight, strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.elu(tf.nn.bias_add(conv3, self.conv3_bias))

        conv4 = tf.nn.conv2d(conv3, self.conv4_weight, strides=[1, 2, 2, 1], padding='SAME')
        conv4 = tf.nn.elu(tf.nn.bias_add(conv4,self.conv4_bias))

        flatten_feature = flatten(conv4) # 1024-d

        state_feature = tf.nn.elu(tf.matmul(flatten_feature, self.fc_weight_encoder) + self.fc_bias_encoder) # 1024

        return state_feature


    def _build_decoder(self,input_feature):
        input_feature = (tf.matmul(input_feature, self.fc_weight_decoder) + self.fc_bias_decoder)

        input_feature = inverse_flatten(input_feature,(-1,4,4,64))

        resize5 = tf.image.resize_images(input_feature,size=(7,7),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv5 = tf.nn.conv2d(resize5, self.conv5_weight,strides=[1,1,1,1],  padding='SAME')
        conv5 = tf.nn.elu(tf.nn.bias_add(conv5, self.conv5_bias))
        # 7x7x32

        resize6 = tf.image.resize_images(conv5,size=(14,14),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv6 = tf.nn.conv2d(resize6, self.conv6_weight,strides=[1,1,1,1] , padding='SAME')
        conv6 = tf.nn.elu(tf.nn.bias_add(conv6, self.conv6_bias))
        # 14x14x16

        resize7 = tf.image.resize_images(conv6,size=(28,28),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv7 = tf.nn.conv2d(resize7, self.conv7_weight, strides=[1,1,1,1],padding='SAME')
        conv7 = tf.nn.elu(tf.nn.bias_add(conv7, self.conv7_bias))
        # 28x28x8

        resize8 = tf.image.resize_images(conv7,size=(56,56),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv8 = tf.nn.conv2d(resize8, self.conv8_weight, strides=[1,1,1,1],padding='SAME')
        conv8 = tf.nn.elu(tf.nn.bias_add(conv8, self.conv8_bias))
        # 56x56x3

        resize9 = tf.image.resize_images(conv8,size=(112,112),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv9 = tf.nn.conv2d(resize9, self.conv7_weight, strides=[1,1,1,1],padding='SAME')
        conv9 = tf.nn.elu(tf.nn.bias_add(conv9, self.conv9_bias))
        # 112x112x3

        resize10 = tf.image.resize_images(conv9,size=(224,224),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return resize10






if __name__ == '__main__':

    Encoder_Network('global')