import collections
import numpy as np
import tensorflow as tf
import access

DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state',
                                               'controller_state'))


class DNCCell(tf.keras.layers.SimpleRNNCell):
    def __init__(self, access_config, controller_config, output_size, clip_value= None, name='dnc'):
        # TODO
        super(DNCCell, self).__init__(name=name)
        self._controller = tf.keras.layers.LSTM(**controller_config)  # TODO what do we do here?
        self._access = access.MemoryAccess(**access_config) 

        self._access_output_size = np.prod(self._access_output_size.as_list())
        self._output_size = output_size
        self._clip_value = clip_value or 0

        self._output_size = tf.TensorShape([output_size])
        self._state_size = DNCState(access_output=self._access_output_size, access_state=self._access.state_size, controller_state=self._controller.state_size)

    def _clip_if_enabled(self, x):
        if self._clip_value > 0:
            return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        return x

    def __call__(self, inputs, prev_state):
        """ TODO 1. get read outputs from previous state and concatenate with input
                 2. run controller on inputs
                 3. use contoller outputs to do read write access stuff
                 4. output contoller output and access output
        """

