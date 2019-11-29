import tensorflow as tf
import numpy as np
import tensorflow as tf
from train_encoder.config_of_encoder import *
from operations.generate_weight import *


random_prob = []
for i in range(ACTION_SIZE):
    random_prob.append(1.0/ACTION_SIZE)


class Encoder_Network(object):

    def __init__(self,name,sess,global_net = None):

        with tf.name_scope(name):

            self.session = sess

            self.current_image = tf.placeholder(tf.float32,[None,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNEL], 'Current_image')

            self.next_image = tf.placeholder(tf.float32,[None,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNEL], 'Next_image')

            self.action = tf.placeholder(tf.int32 , [None,],'Action_index')

            self.learning_rate = tf.placeholder(tf.float32,None,'learning_rate')

            self.OPT = tf.train.RMSPropOptimizer(self.learning_rate , name='RMSPropA')

            self._prepare_weight()

            if 'global' in name:

                self.current_feature = self._build_encoder(input_image=self.current_image)
                self.next_feature    = self._build_encoder(input_image=self.next_image)

                if global_net :
                    self.global_net = global_net
                else:
                    print('global_net can not be None !!!!!')

                self.action_prob,self.a_p_loss,self.update_a_p_op,self.pull_a_p_op = self._prepare_action_predicter(current_feature=self.current_feature,next_feature=self.next_feature)

                self.loss,self.update_s_p_op,self.pull_s_p_op = self._prepare_state_predicter(current_feature=self.current_feature,action=self.action,next_image=self.next_image)



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
            self.action_predict_params = [self.a_p_weight,self.a_p_bias]

            # weight for state_predicter
            self.s_p_weight_1 = generate_fc_weight(shape=[1024+ACTION_SIZE,1024],name='s_p_weight_1')
            self.s_p_bias_1   = generate_fc_bias(shape=[1024],name='s_p_bias_1')
            self.s_p_weight_2 = generate_fc_weight(shape=[1024,1024],name='s_p_weight_2')
            self.s_p_bias_2   = generate_fc_bias(shape=[1024],name='s_p_bias_2')
            self.state_predict_params = [self.s_p_weight_1,self.s_p_bias_1,self.s_p_weight_2,self.s_p_bias_2]


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
        conv9 = tf.nn.conv2d(resize9, self.conv9_weight, strides=[1,1,1,1],padding='SAME')
        conv9 = tf.nn.elu(tf.nn.bias_add(conv9, self.conv9_bias))
        # 112x112x3

        resize10 = tf.image.resize_images(conv9,size=(224,224),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return resize10


    def _prepare_action_predicter(self,current_feature,next_feature):

        concat_feature = tf.concat([current_feature,next_feature],axis=1)  # 1024||1024 --> ACTION-SIZE
        action_prob = tf.nn.softmax(tf.matmul(concat_feature,self.a_p_weight) + self.a_p_bias)

        a_p_loss = -tf.reduce_mean(tf.one_hot(self.action,ACTION_SIZE,dtype=tf.float32)*
                                   tf.log(tf.clip_by_value(action_prob,1e-10,1.0)))

        a_p_grads = [tf.clip_by_norm(item, 40) for item in tf.gradients(self.action_predict_loss,
                                  self.encoder_params + self.action_predict_params )]

        update_a_p_op = self.OPT.apply_gradients(a_p_grads,
                                self.global_net.encoder_params + self.global_net.action_predict_params)

        pull_a_p_op = [l_p.assign(g_p) for l_p, g_p in zip(self.encoder_params + self.action_predict_params,
                                self.global_net.encoder_params + self.global_net.action_predict_params)]

        return action_prob,a_p_loss,update_a_p_op,pull_a_p_op

    def _prepare_state_predicter(self,current_feature,action,next_image):

        concat_feature = tf.concat([current_feature , tf.one_hot(action,4,dtype=tf.float32)])

        s_p_temp = (tf.matmul(concat_feature,self.s_p_weight_1) + self.s_p_bias_1)

        state_feature_predicted = (tf.matmul(s_p_temp,self.s_p_weight_2) + self.s_p_bias_2)

        next_image_predicted = self._build_decoder(state_feature_predicted)

        loss = tf.subtract(next_image,next_image_predicted)

        s_p_grads = [tf.clip_by_norm(item, 40) for item in
                     tf.gradients(loss , self.state_predict_params + self.encoder_params)]

        update_s_p_op = self.OPT.apply_gradients(s_p_grads,
                                         self.global_net.state_predict_params + self.global_net.encoder_params)

        pull_s_p_op = [l_p.assign(g_p) for l_p, g_p in zip(self.encoder_params + self.state_predict_params,
                                         self.global_net.encoder_params + self.global_net.state_predict_params)]

        return loss,update_s_p_op,pull_s_p_op


    # choose action:

    def random_choose_action(self):
        action = np.random.choice(range(ACTION_SIZE),p=random_prob)
        return action

    def update_state_predicter(self,current_image,action,next_image,learning_rate):
        self.session.run(self.update_s_p_op,
                         feed_dict = {self.current_image:current_image,
                                      self.action:action,
                                      self.next_image:next_image,
                                      self.learning_rate:learning_rate})

    def update_action_predicter(self,current_image,action,next_image,learning_rate):
        self.session.run(self.update_a_p_op,
                         feed_dict = {
                             self.current_image:current_image,
                             self.action:action,
                             self.next_image:next_image,
                             self.learning_rate:learning_rate
                         })


    def pull_all_params(self):
        self.session.run([self.pull_a_p_op,self.pull_s_p_op])





if __name__ == '__main__':

    Encoder_Network('global')