import collections
import numpy as np
import tensorflow as tf
import access

DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state',
                                               'controller_state'))


class DNCCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, access_config, controller_config, output_size, clip_value=None, name='dnc'):
        super(DNCCell, self).__init__()
        self.controller_layers = controller_config['hidden_layers']
        self.controller_size = controller_config['hidden_size']
        
        def single_cell(num_units):
            return tf.keras.layers.LSTMCell(num_units)
        self._controller = tf.keras.layers.StackedRNNCells([single_cell(self.controller_size) for _ in range(self.controller_layers)])
        # self._controller = tf.keras.layers.LSTM(**controller_config) 
        self._access = access.MemoryAccess(**access_config) 

        self._access_output_size = np.prod(self._access.output_size.as_list())
        self._output_size = output_size
        self._clip_value = clip_value or 0

        self._output_size = tf.TensorShape([output_size])
        self._state_size = DNCState(access_output=self._access_output_size, access_state=self._access.state_size, controller_state=self.controller_size)

    def _clip_if_enabled(self, x):
        if self._clip_value > 0:
            return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        return x

    def __call__(self, inputs, prev_state):
        prev_access_output = prev_state.access_output
        prev_access_state = prev_state.access_state
        prev_controller_state = prev_state.controller_state

        controller_input = tf.concat([tf.keras.backend.batch_flatten(inputs), tf.keras.backend.batch_flatten(prev_access_output)],1)
        controller_output, controller_state = self._controller(controller_input, prev_controller_state)

        controller_output = self._clip_if_enabled(controller_output)
        controller_state = tf.nest.map_structure(self._clip_if_enabled, controller_state)

        access_output, access_state = self._access(controller_output, prev_access_state)

        output = tf.concat([controller_output, tf.keras.backend.batch_flatten(access_output)], 1)
        output = tf.keras.layers.Dense(self._output_size.as_list()[0], name='output_linear')(output)
        output = self._clip_if_enabled(output)
        return output, DNCState(access_output=access_output, access_state=access_state, controller_state=controller_state)

    def initial_state(self, batch_size, dtype=tf.float32):
        return DNCState(controller_state=self._controller.get_initial_state(inputs=None, batch_size=batch_size, dtype=dtype),
            access_state=self._access.get_initial_state(inputs=None, batch_size=batch_size, dtype=dtype),
            access_output=tf.zeros([batch_size] + self._access.output_size.as_list(), dtype))
    
    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

