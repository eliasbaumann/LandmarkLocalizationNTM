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
    # return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
    #     activation_fn=None, biases_initializer=None))
    return tf.squeeze(tf.keras.layers.Dense(units=units, activation=None, biases_initializer=None)(tf.ones([1, 1])))

def create_linear_initializer(input_size):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.keras.initializers.TruncatedNormal(stddev=stddev, seed=42)

# Big parts of this code taken from:
# https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/ntm.py
class NTMCell(tf.keras.layers.AbstractRNNCell):
    '''
    memory_mode: 'matrix' -> store matrices, 'embedding' -> create embedding and store
    '''
    def __init__(self, controller_layers, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num,
                 addressing_mode='content_and_location', shift_range=1, reuse=False, output_dim=None, clip_value=20,
                 init_mode='constant', memory_mode='matrix'):
        super(NTMCell, self).__init__()
        self.controller_layers = controller_layers
        self.controller_units = controller_units
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.shift_range = shift_range
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        self.reuse = reuse
        self.clip_value = clip_value

        self.num_heads = self.read_head_num+self.write_head_num


        self._ds = tf.keras.layers.MaxPooling2D(pool_size=(8,8))

        # ########### TODO: Matrix mode:
        # TODO trying this using the already done implementation of convlstm2d        
        self._controller = tf.keras.layers.ConvLSTM2DCell(filters=self.num_heads+2*self.write_head_num+1, kernel_size=3) # This needs to output CxC matrix K_t, and maybe a second matrix which is then used for other parameters?
        # with multiple heads this needs to scale up?
        self.num_params_per_head = 1+1+(self.shift_range*2+1)+1

        self._ctrl2p = tf.keras.layers.Dense(units=self.num_heads*self.num_params_per_head, activation=None) # Outputs: keystrength scalar beta_t, interpolation gate scalar g_t, shift weighting vector s_t (length n_memory rows), sharpening scalar gamma_t

        self._ctrl2o = tf.keras.layers.Conv2D()#TODO how should the output look like?
        
        # ########### TODO: Embedding mode:

        # self._controller = tf.keras.layers.LSTMCell(filters=x, kernel_size=3)
        # self._conv = tf.keras.layers.Conv2D(kernel_size=3) #TODO

        # self._flat = tf.keras.layers.Flatten()
        # self._rs = tf.keras.layers.Reshape(target_shape=TODO)
        # self._us = tf.keras.layers.Conv2DTranspose(size=[8,8])

        # self.num_params_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        # self.total_param_num = self.num_params_per_head * self.num_heads + self.memory_vector_dim * 2 * self.write_head_num



        self.init_mode = init_mode

        self.step = 0
        self.output_dim = output_dim
        self.shift_range = shift_range

        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(self.controller_units + self.memory_vector_dim * self.read_head_num)
    
    def __call__(self, x, prev_state):
        prev_read_list = prev_state.read_list

        controller_input = tf.concat([x] + prev_read_list, axis=0) #TODO this will not work for me, need to change
        #with tf.variable_scope('controller', reuse=self.reuse):
        controller_output, controller_state = self._controller(controller_input, prev_state.controller_state)
        

    @property
    def state_size(self):
        return NTMControllerState()

    @property
    def output_size(self):
        return self.output_dim