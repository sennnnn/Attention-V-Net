import os
import tensorflow as tf

from model import zigzag_unet
from loss_metric import *
from model_util import *

class zigzag(object):
    def __init__(self, base_path):
        self.ckpt_path = os.path.join(base_path, 'ckpt')
        self.pb_path = os.path.join(base_path, 'frozen_model')
        self.graph = tf.Graph()

    def graph_compose(self, num_class, initial_channel):
        with self.graph.as_default():
            data = tf.placeholder(tf.float32, [None, None, None, 1], name='data')
            self.label = {'label': tf.placeholder(tf.uint8, [None, None, None, num_class], name='label')}
            keep_prob = tf.placeholder(tf.float32, name='keep_probability')
            with tf.variable_scope('network'):
                output_list = zigzag_unet(data, num_class, initial_channel, keep_prob)
            tf.identity(output_list[-1], name='predict')
            self.predict = {'zigzag_1_output': output_list[0], 'zigzag_2_output': output_list[1], \
                            'zigzag_3_output': output_list[2], 'zigzag_4_output': output_list[3], \
                            'predict': tf.nn.softmax(output_list[3])}

        return self.graph

    def loss_compose(self):
        with self.graph.as_default():
            ce = lambda labels,logits: tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
            loss_1 = tf.reduce_mean(ce(labels=self.label['label'], logits=self.predict['zigzag_1_output']))
            loss_2 = tf.reduce_mean(ce(labels=self.label['label'], logits=self.predict['zigzag_2_output']))
            loss_3 = tf.reduce_mean(ce(labels=self.label['label'], logits=self.predict['zigzag_3_output']))
            loss_4 = tf.reduce_mean(ce(labels=self.label['label'], logits=self.predict['zigzag_4_output']))
            self.loss = {'multi-task integreted loss': tf.reduce_mean(loss_1 + loss_2 + loss_3 + loss_4), \
                         'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4, \
                         'loss': loss_4}
            
            return self.loss

    def metric_compose(self):
        with self.graph.as_default():
            so = lambda x: tf.nn.softmax(x)
            metric_1 = tf_dice(self.label['label'], so(self.predict['zigzag_1_output']))
            metric_2 = tf_dice(self.label['label'], so(self.predict['zigzag_2_output']))
            metric_3 = tf_dice(self.label['label'], so(self.predict['zigzag_3_output']))
            metric_4 = tf_dice(self.label['label'], so(self.predict['zigzag_4_output']))
            metric_o = tf_dice(self.label['label'], self.predict['predict'])
            self.metric = {'zigzag_1_metric': metric_1, 'zigzag_2_metric': metric_2, \
                           'zigzag_3_metric': metric_3, 'zigzag_4_metric': metric_4, \
                           'metric': metric_o}
            
            return self.metric

    def optimizer_compose(self):
        with self.graph.as_default():
            # learning rate relating things
            lr_input = tf.placeholder(tf.float32, name='lr_input')
            lr = tf.Variable(1., name='lr')

            lr_init_op = tf.assign(lr, lr_input, name='lr_initial_op')
            lr_decay_op = tf.assign(lr, lr/2, name='lr_decay_op')
            
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer')
            variables_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network/zigzag_1')
            variables_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network/zigzag_2')
            variables_3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network/zigzag_3')
            variables_4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network/zigzag_4')
            optimizer_1 = optimizer.minimize(self.loss['loss_1'], var_list=variables_1)
            optimizer_2 = optimizer.minimize(self.loss['loss_2'], var_list=variables_2)
            optimizer_3 = optimizer.minimize(self.loss['loss_3'], var_list=variables_3)
            optimizer_4 = optimizer.minimize(self.loss['loss_4'], var_list=variables_4)
            self.optimizer = tf.group(optimizer_1, optimizer_2, optimizer_3, optimizer_4)

            return self.optimizer, lr, lr_init_op, lr_decay_op

    def restore(self, pattern, sess, graph, saver):
        if(pattern == "ckpt"):
            try:
                saver.restore(sess,"{}/best_model".format(self.ckpt_path))
                print("The latest checkpoint model is loaded...")
            except:
                sess = restore_from_pb(sess, load_graph(get_newest("{}".format(self.pb_path))), graph)
        else:
            pb_name = get_newest("{}".format(self.pb_path))
            print("{},the latest frozen graph is loaded...".format(pb_name))
            pb_graph = load_graph(pb_name)
            sess = restore_from_pb(sess, pb_graph, graph)

    def save(self, sess, saver, epoch, pattern, one_epoch_avg_metric):
        return_string = ''
        if(pattern == 'ckpt'):
            saver.save(sess, "{}/best_model".format(self.ckpt_path))
            pb_name = "{}/{}_%.3f.pb".format(self.pb_path, epoch)%(one_epoch_avg_metric)
            return_string += 'frozen_model_save {}\n'.format(pb_name)
            return_string += frozen_graph(sess, pb_name)
        elif(pattern == 'pb'):
            # 因为是从 frozen_model 中 restore 所以应该将 restore_from_pb 中多出来的那些 assign 操作去掉
            saver.save(sess, "{}/best_model".format(self.ckpt_path))
            pb_name = "{}/{}_%.3f.pb".format(self.pb_path, epoch)%(one_epoch_avg_metric)
            return_string += 'frozen_model_save {}\n'.format(pb_name)
            return_string += frozen_graph(sess, pb_name)
        else:
            print('pattern must be ckpt or pb!')
            exit()

        return return_string