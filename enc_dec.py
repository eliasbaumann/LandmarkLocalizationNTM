## Wrap NTM with encoder decoder network s.t. the ntm is fed vectors and not images.


import tensorflow as tf

from ntm import NTMCell

class Encoder_Decoder_Wrapper(tf.keras.layers.AbstractRNNCell):
    '''
    Wraps NTMcell in an encoder decoder structure that downsamples and then upsamples the input with the bottleneck being an ntm cell
    '''
    def __init__(self, ntm_config, batch_size, layer, name='enc_dec', **kwargs):
        super(Encoder_Decoder_Wrapper, self).__init__(name=name, **kwargs)
        self.layer = layer
        self.batch_size = batch_size
        
        if ntm_config["ntm_param"] is not None:
            self.controller_units = ntm_config["ntm_param"]["controller_units"]
            self.memory_size = ntm_config["ntm_param"]["memory_size"]
            self.memory_vector_dim = ntm_config["ntm_param"]["memory_vector_dim"]
            self.output_dim = ntm_config["ntm_param"]["output_dim"]
            self.read_head_num = ntm_config["ntm_param"]["read_head_num"]
            self.write_head_num = ntm_config["ntm_param"]["write_head_num"]
            
            self.cell = NTMCell(controller_units=self.controller_units, memory_size=self.memory_size, memory_vector_dim=self.memory_vector_dim, read_head_num=self.read_head_num, write_head_num=self.write_head_num, output_dim=self.output_dim, batch_size=batch_size, layer=self.layer) 
        else:
            self.controller_units = None
            self.memory_size = None
            self.memory_vector_dim = None
            self.output_dim = None
            self.read_head_num = None
            self.write_head_num = None
            
            self.cell = None
        self.num_filters = ntm_config["enc_dec_param"]["num_filters"]
        self.kernel_size = ntm_config["enc_dec_param"]["kernel_size"]
        self.pool_size = ntm_config["enc_dec_param"]["pool_size"]
        
        self.dim = tf.sqrt(tf.cast(self.output_dim, tf.float32))
        self.conv = [tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, activation='relu', padding='same', data_format='channels_first', name='enc_dec_conv_%d_%d' % (self.layer, i)) for i in range(len(self.pool_size)*2+1)]

        self.conv_enc = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, activation='relu', padding='same', data_format='channels_first', name='enc_dec_last_enc')
        self.ds = [tf.keras.layers.AveragePooling2D(pool_size=i, strides=self.pool_size, data_format='channels_first', name='enc_dec_ds_%d_%d' % (self.layer, i)) for i in self.pool_size]
        self.us = [tf.keras.layers.UpSampling2D(size=i, data_format='channels_first', name='enc_dec_us_%d_%d' % (self.layer, i)) for i in self.pool_size[::-1]]

        self.flat = tf.keras.layers.Flatten(data_format='channels_first', name='enc_dec_flat_%d' % self.layer)
        

    def call(self, x, state):
        for i in range(len(self.pool_size)):
            x = self.conv[i](x)
            x = self.ds[i](x)
        x = self.conv_enc(x)
        
        # NTM here
        if self.cell is not None:
            x = self.flat(x)
            # state = self.cell.get_initial_state(self.batch_size)
            ntm_out, state = self.cell(x, state)
            x = tf.reshape(ntm_out, [self.batch_size, -1, self.dim, self.dim]) # Output_dim always has to have a square root

        for i in range(len(self.pool_size)):
            x = self.conv[i+len(self.pool_size)](x)
            x = self.us[i](x)
        x = self.conv[len(self.pool_size)*2](x)
        return x, state

