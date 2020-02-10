## Wrap NTM with encoder decoder network s.t. the ntm is fed vectors and not images.


import tensorflow as tf

from ntm import NTMCell

class Encoder_Decoder_Wrapper(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, pool_size, batch_size):
        super(Encoder_Decoder_Wrapper, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.conv = [tf.keras.layers.Conv2D(filters=num_filters, kernel_size=self.kernel_size, activation='relu', padding='same', data_format='channels_first') for _ in range(4)]

        self.conv_enc = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, activation='relu', padding='same', data_format='channels_first')
        self.conv_out = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, activation='relu', padding='same', data_format='channels_first')
        self.ds = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_size, data_format='channels_first')
        self.us = tf.keras.layers.UpSampling2D(size=pool_size, data_format='channels_first')

        self.cell = NTMCell(controller_units=256, memory_size=64, memory_vector_dim=256, read_head_num=1, write_head_num=1, output_dim=256) # TODO


    def call(self, inputs):
        x = self.conv[0](inputs) 
        x = self.ds(x)
        x = self.conv[1](x)
        x = self.ds(x)
        x = self.conv_enc(x)
        x = tf.reshape(x, [self.batch_size, -1])
        x.set_shape([self.batch_size, 256])

        # NTM here
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        ntm_out, state = self.cell(x, state)

        x = tf.reshape(ntm_out, [self.batch_size, 1, 16, 16]) 
        x = self.conv[2](x)
        x = self.us(x)
        x = self.conv[3](x)
        x = self.us(x)
        x = self.conv_out(x)
        return x
