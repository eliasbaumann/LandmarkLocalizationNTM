import tensorflow as tf
import numpy as np

import collections


def create_linear_initializer(input_size):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.keras.initializers.TruncatedNormal(stddev=stddev, seed=42)

# Big parts of this code taken from:
# https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/ntm.py / https://github.com/snowkylin/ntm/blob/master/model_v2.py # they have lots of overlap
class NTMCell(tf.keras.layers.AbstractRNNCell):
    '''
    
    '''
    def __init__(self, controller_units, memory_size, memory_vector_dim, read_head_num, write_head_num, batch_size, layer,
                 addressing_mode='content_and_location', shift_range=1, reuse=False, output_dim=None, clip_value=20,
                 init_mode='constant', name='ntm_cell'):
        super(NTMCell, self).__init__(name=name)
        # self.controller_layers = controller_layers
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
        self.batch_size = batch_size
        self.layer = layer

        self._ds = tf.keras.layers.MaxPooling2D(pool_size=(8,8), name='ntm_pool2d')

        self._controller = tf.keras.layers.LSTMCell(units=self.controller_units)

        self.num_params_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        self.total_param_num = self.num_params_per_head * self.num_heads + self.memory_vector_dim * 2 * self.write_head_num

        self.init_mode = init_mode

        self.output_dim = output_dim
        self.shift_range = shift_range

        # o2p = controller output to parameters, o2o = controller outputs to output

        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(self.controller_units + self.memory_vector_dim * self.read_head_num)

        self.o2p = tf.keras.layers.Dense(units=self.total_param_num, use_bias=True, kernel_initializer=self.o2p_initializer, name='o2p')
        self.o2o = tf.keras.layers.Dense(units=self.output_dim, use_bias=True, kernel_initializer=self.o2o_initializer, name='o2o')

        # for initial state: Create variables:
        # from https://github.com/snowkylin/ntm/blob/master/ntm/ntm_cell_v2.py#L39 
        self.init_memory_state = self.add_weight(name='init_memory_state_%d' % self.layer,
                                                 shape=[self.controller_units],
                                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
        self.init_carry_state = self.add_weight(name='init_carry_state_%d' % self.layer,
                                                shape=[self.controller_units],
                                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
        self.init_r = [self.add_weight(name='init_r_%d_%d' % (self.layer, i),
                                       shape=[self.memory_vector_dim],
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
                       for i in range(self.read_head_num)]
        self.init_w = [self.add_weight(name='init_w_%d_%d' % (self.layer, i),
                                       shape=[self.memory_size],
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
                       for i in range(self.read_head_num + self.write_head_num)]
        self.init_M = self.add_weight(name='init_M_%d' % self.layer,
                                      shape=[self.memory_size, self.memory_vector_dim],
                                      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))


    
    # @tf.function
    def call(self, x, prev_state):
        prev_read_list = prev_state["read_vector_list"]
        #prev_read_list.set_shape([self.read_head_num, self.batch_size, self.memory_vector_dim])
        controller_input = tf.concat([x]+prev_read_list, axis=1, name='concat_ctrl_inp%d' % self.layer)
        controller_state = prev_state["controller_state"]
        # controller_state.set_shape([2, self.batch_size, self.controller_units])
        controller_output, controller_state = self._controller(controller_input, controller_state)
        parameters = self.o2p(controller_output)
        parameters = tf.clip_by_value(parameters, -self.clip_value, self.clip_value)

        head_parameter_list = tf.split(parameters[:, :self.num_params_per_head * self.num_heads], self.num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, self.num_params_per_head * self.num_heads:], 2 * self.write_head_num, axis=1)

        prev_w_list = prev_state["w_list"]
        # prev_w_list.set_shape([self.read_head_num + self.write_head_num, self.batch_size, self.memory_size])
        prev_M = prev_state["M"]
        # prev_M.set_shape([self.batch_size, self.memory_size, self.memory_vector_dim])
        w_list = []
        
        # create params
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = tf.nn.softplus(head_parameter[:, self.memory_vector_dim])
            g = tf.sigmoid(head_parameter[:, self.memory_vector_dim + 1])
            s = tf.nn.softmax(head_parameter[:, self.memory_vector_dim + 2:self.memory_vector_dim + 2 + (self.shift_range * 2 + 1)])
            gamma = tf.nn.softplus(head_parameter[:, -1]) + 1
            w = self._addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])
            w_list.append(w)
        
        # read
        read_w_list = w_list[:self.read_head_num]
        read_vector_list = [tf.reduce_sum(tf.expand_dims(read_w_list[i], axis=2)*prev_M, axis=1) for i in range(self.read_head_num)]
        
        # write
        write_w_list = w_list[self.read_head_num:]
        M = prev_M
        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)
            erase_vec = tf.expand_dims(tf.sigmoid(erase_add_list[i*2]), axis=1)
            add_vec = tf.expand_dims(tf.tanh(erase_add_list[i*2+1]), axis=1)
            M = M * (tf.ones(M.get_shape()) - tf.matmul(w, erase_vec)) + tf.matmul(w, add_vec)

        ntm_output = self.o2o(tf.concat([controller_output] + read_vector_list, axis=1, name='concat_ntm_out_%d' % self.layer))
        ntm_output = tf.clip_by_value(ntm_output, -self.clip_value, self.clip_value)
        state = {
            'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_list': w_list,
            'M': M
        }
        return ntm_output, state
    
    @tf.function
    def _addressing(self, k, beta, g, s, gamma, prev_M, prev_w):
        # content focussing:
        K = self._similarity(k, prev_M, method='cosine')
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1)*K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keepdims=True)

        # location focussing:
        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w

        # TODO i literally do not know whats happening here, we create the shift matrix i guess
        s = tf.concat([s[:, :self.shift_range + 1],
                       tf.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1, name='concat_adressing_%d_1' % self.layer)
        t = tf.concat([tf.reverse(s, axis=[1]), tf.reverse(s, axis=[1])], axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)],
            axis=1, name='concat_adressing_%d_2' % self.layer
        )

        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1)*s_matrix, axis=2)
        w_sharpened = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpened / tf.reduce_sum(w_sharpened, axis=1, keepdims=True)
        return w

    # u = k, v = M
    @tf.function
    def _similarity(self,u,v, method='cosine'):
        '''
        Evaluates similarity between key vector and every row in Memory
        Implemented: cosine similarity
        '''
        u = tf.expand_dims(u,axis=2)
        if method=='cosine':

            nom = tf.matmul(v, u)
            u_norm = tf.sqrt(tf.reduce_sum(tf.square(u), axis=1, keepdims=True))
            v_norm = tf.sqrt(tf.reduce_sum(tf.square(v), axis=2, keepdims=True))
            denom = v_norm * u_norm
            return tf.squeeze(nom / (denom + 1e-8))#tf.squeeze(tf.math.divide_no_nan(nom,denom)) # instead of adding 1e-8 
    
    @tf.function
    def get_initial_state(self):
        initial_state = {
            'controller_state': [self._expand(tf.tanh(self.init_memory_state), dim=0, N=self.batch_size),
                                 self._expand(tf.tanh(self.init_carry_state), dim=0, N=self.batch_size)],
            'read_vector_list': [self._expand(tf.nn.tanh(self.init_r[i]), dim=0, N=self.batch_size)
                                 for i in range(self.read_head_num)],
            'w_list': [self._expand(tf.nn.softmax(self.init_w[i]), dim=0, N=self.batch_size)
                       for i in range(self.read_head_num + self.write_head_num)],
            'M': self._expand(tf.tanh(self.init_M), dim=0, N=self.batch_size)
        }
        return initial_state

    @tf.function
    def _expand(self, x, dim, N):
        return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim, name='concat_expand_%d' % self.layer)

    @tf.function
    def _learned_init(self, units):
        return tf.squeeze(tf.keras.layers.Dense(units, activation_fn=None, biases_initializer=None, name='learned_init_%d' % self.layer)(tf.ones([1, 1])))

    @property
    def output_size(self):
        return self.output_dim

