import tensorflow as tf
import numpy as np

import collections

# BIG TODO

NTMControllerState = collections.namedtuple('NTMControllerState', ('controller_state', 'read_list', 'w_list', 'M'))

class ntm(tf.keras.Model):
    def __init__(self):
        super(ntm, self).__init__()
    
    def call(self, inputs):
        pass


class aug_unet(tf.keras.layers.Layer):
    def __init__(self):
        super(aug_unet, self).__init__()

    def call(self, inputs):
        pass

class mem_conv_rnn(tf.keras.layers.Layer):
    def __init__(self):
        super(mem_conv_rnn, self).__init__()
    
    def call(self, inputs):
        pass

# https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/utils.py
def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

def learned_init(units):
    return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
        activation_fn=None, biases_initializer=None))

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)

# Big parts of this code taken from:
# https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/ntm.py
class NTMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, controller_layers, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num,
                 addressing_mode='content_and_location', shift_range=1, reuse=False, output_dim=None, clip_value=20,
                 init_mode='constant'):
        super(NTMCell, self).__init__()
        self.controller_layers = controller_layers
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.clip_value = clip_value

        # def single_cell(num_units):
        #     return tf.keras.layers.ConvLSTM2DCell(filters=num_units, kernel_size=1) # TODO is this the right way?

        # TODO trying this using the already done implementation of convlstm2d        
        self._controller = tf.keras.layers.ConvLSTM2DCell(filters=self.controller_units, kernel_size=3)
        # Thoughts on this:
        # how do we get the controlling parameters?
        # do we just use very long filters -> output is a vector?


        self.init_mode = init_mode

        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range

        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(self.controller_units + self.memory_vector_dim * self.read_head_num)
    
    def __call__(self, x, prev_state):
        prev_read_list = prev_state.read_list

        controller_input = tf.concat([x] + prev_read_list, axis=0) #TODO axis?
        with tf.variable_scope('controller', reuse=self.reuse):
            controller_output, controller_state = self._controller(controller_input, prev_state.controller_state)
        

    @property
    def state_size(self):
        return NTMControllerState()

    @property
    def output_size(self):
        return self.output_dim