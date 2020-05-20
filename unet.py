import tensorflow as tf
from enc_dec import Encoder_Decoder_Wrapper, Encoder_Decoder_Baseline

# TODO: Tasks 
# - fixed upsampling instead of deconv?? Is in, but needs tryiing

class convnet2d(tf.keras.Model):
    def __init__(self, num_fmaps, num_landmarks, name='convnet2d', **kwargs):
        super(convnet2d, self).__init__(name=name, **kwargs)
        self.num_fmaps = num_fmaps
        self.num_landmarks = num_landmarks
        self.convnet = conv_pass(11, 128, 5)
        self.heatmap = conv_pass(1, self.num_landmarks, 1, activation=None)

    def call(self, inputs):
        x = self.convnet(inputs)
        logits = self.heatmap(x)
        return logits


class unet2d(tf.keras.Model):
    def __init__(self, num_fmaps, fmap_inc_factor, downsample_factors, num_landmarks, ntm=False, ntm_pos=[0], enc_dec=False, batch_size=None, training=None, im_size=[256, 256], name='unet2d', **kwargs): #TODO ntm and enc_dec should not be truable at the same time
        super(unet2d, self).__init__(name=name, **kwargs)
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.num_landmarks = num_landmarks
        self.ntm = ntm
        self.enc_dec = enc_dec
        self.ntm_pos = ntm_pos
        self.batch_size = batch_size
        self.training = training
        self.im_size = im_size
        self.unet = unet(self.num_fmaps, self.fmap_inc_factor, self.downsample_factors, ntm=self.ntm, ntm_pos=self.ntm_pos, enc_dec=self.enc_dec, batch_size=self.batch_size, im_size=self.im_size)
        self.logits = conv_pass(1, self.num_landmarks, 1, activation=tf.keras.activations.tanh)

    @tf.function#(input_signature=[tf.TensorSpec(shape=[None,1,256,256], dtype=tf.float32)])
    def call(self, inputs):
        unet_2d = self.unet(inputs)
        res = self.logits(unet_2d) # TODO payer et al do no activation ?
        return res

class unet(tf.keras.layers.Layer):
    def __init__(self, num_fmaps, fmap_inc_factor, downsample_factors, ntm=False, ntm_pos=[0], memory_size=64, enc_dec=False, batch_size=None, activation=tf.keras.activations.relu, layer=0, im_size=[256, 256], name='unet', **kwargs):
        super(unet, self).__init__(name=name+'_'+str(layer), **kwargs)
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.ntm = ntm
        self.ntm_pos = ntm_pos
        self.enc_dec = enc_dec
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.activation = activation
        self.layer = layer
        self.im_size = im_size
        self.inp_conv = conv_pass(kernel_size=3,
                                  num_fmaps=self.num_fmaps,
                                  num_repetitions=2,
                                  activation=self.activation,
                                  name='unet_left_%i'%self.layer)


        if self.ntm and self.layer in self.ntm_pos:
            assert self.batch_size is not None, 'Please set batch_size in unet2d init'
            self.ntm_enc_dec = Encoder_Decoder_Wrapper(num_filters=64, kernel_size=3, pool_size=4, batch_size=self.batch_size, memory_size=self.memory_size, name="ntm_enc_dec_"+str(self.layer)) # TODO parameterizable 
        elif self.enc_dec and self.layer in self.ntm_pos:
            self.ntm_enc_dec = Encoder_Decoder_Baseline(num_filters=64, kernel_size=3, pool_size=4, batch_size=self.batch_size, name="enc_dec_"+str(self.layer)) # TODO parameterizable
        else:
            self.ntm_enc_dec = None

        if self.layer >= len(self.downsample_factors)-1:
            self.drop = tf.keras.layers.Dropout(.2,seed=42, name='dropout_%i'%self.layer)
        else:
            self.drop = None

        if self.layer < len(self.downsample_factors):
            self.unet_rec = unet(num_fmaps=self.num_fmaps*self.fmap_inc_factor,
                                 fmap_inc_factor=self.fmap_inc_factor,
                                 downsample_factors=self.downsample_factors,
                                 ntm=self.ntm,
                                 ntm_pos=self.ntm_pos,
                                 enc_dec=self.enc_dec,
                                 batch_size=self.batch_size,
                                 activation=self.activation,
                                 layer=self.layer+1,
                                 im_size=self.im_size)
            self.ds = downsample(factors=self.downsample_factors[self.layer], name='ds_%i'%self.layer)
            self.us = upsample(factors=self.downsample_factors[self.layer],
                               num_fmaps=self.num_fmaps,
                               activation=self.activation,
                               name='us_%i'%self.layer)
            self.crop = crop_spatial(name='crop_%i'%self.layer)
            self.out_conv = conv_pass(kernel_size=3, num_fmaps=self.num_fmaps, num_repetitions=2, name='unet_right_%i'%self.layer)
        else: 
            self.unet_rec = None
            self.ds = None
            self.us = None
            self.crop = None
            self.out_conv = None
    
    @tf.function
    def call(self, inputs):
        f_left = self.inp_conv(inputs)
        if self.ntm and self.layer in self.ntm_pos:
            mem = self.ntm_enc_dec(f_left)
            f_left = tf.concat([mem, f_left], axis=1)
        # bottom layer:
        if self.layer == len(self.downsample_factors):
            f_left = self.drop(f_left)
            return f_left
        # to add dropout to second to last layer as well: 
        elif self.layer == len(self.downsample_factors)-1:
            f_left = self.drop(f_left)
        g_in = self.ds(f_left)
        g_out = self.unet_rec(g_in)
        g_out_upsampled = self.us(g_out)
        f_left_cropped = self.crop(f_left, tf.shape(g_out_upsampled))
        f_right = tf.concat([f_left_cropped, g_out_upsampled],1)
        f_out = self.out_conv(f_right)
        return f_out

    def get_config(self):
        config = super(unet, self).get_config()
        config.update({'num_fmaps':self.num_fmaps, 'fmap_inc_factor':self.fmap_inc_factor, 'downsample_factors':self.downsample_factors, 'ntm':self.ntm, 'batch_size':self.batch_size, 'activation':self.activation, 'layer':self.layer})
        return config

        

class conv_pass(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_fmaps, num_repetitions, activation=tf.keras.activations.relu, name='conv_pass', **kwargs):
        super(conv_pass, self).__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.num_fmaps = num_fmaps
        self.num_repetitions = num_repetitions
        self.activation = activation
        self.conv = [tf.keras.layers.Conv2D(filters=self.num_fmaps, 
                                        kernel_size=self.kernel_size, 
                                        padding='same', 
                                        data_format='channels_first', 
                                        activation=self.activation, 
                                        name=self.name+'_%i'%i) for i in range(self.num_repetitions)]
    
    @tf.function
    def call(self, inputs):
        for i in range(self.num_repetitions):
            inputs = self.conv[i](inputs)
        return inputs

    def get_config(self):
        config = super(conv_pass,self).get_config()
        config.update({'kernel_size':self.kernel_size, 'num_fmaps':self.num_fmaps, 'num_repetitions':self.num_repetitions, 'activation':self.activation})
        return config

class downsample(tf.keras.layers.Layer):
    def __init__(self, factors, name='ds', **kwargs):
        super(downsample, self).__init__(name = name, **kwargs)
        self.factors = factors
        self.ds = tf.keras.layers.AveragePooling2D(pool_size=self.factors, strides=self.factors, padding='same', data_format='channels_first', name=self.name+'_internal')
    
    @tf.function
    def call(self, inputs):
        inputs = self.ds(inputs)
        return inputs

    def get_config(self):
        config = super(downsample, self).get_config()
        config.update({'factors':self.factors})
        return config

class upsample(tf.keras.layers.Layer):
    def __init__(self, factors, num_fmaps, activation=tf.keras.activations.relu, name='us', **kwargs):
        super(upsample, self).__init__(name=name, **kwargs)
        self.factors = factors
        self.num_fmaps = num_fmaps
        self.activation = activation
        # self.us = tf.keras.layers.Conv2DTranspose(filters=self.num_fmaps,
        #                                     kernel_size=self.factors,
        #                                     strides=self.factors,
        #                                     padding='valid',
        #                                     data_format='channels_first',
        #                                     activation=self.activation,
        #                                     name=self.name)
        self.us = tf.keras.layers.UpSampling2D(size=self.factors, data_format = "channels_first", name=name+'_internal')
    
    @tf.function
    def call(self, inputs):
        inputs = self.us(inputs)
        return inputs

    def get_config(self):
        config = super(upsample, self).get_config()
        config.update({'factors':self.factors, 'num_fmaps':self.num_fmaps, 'activation':self.activation})
        return config

class crop_spatial(tf.keras.layers.Layer):
    def __init__(self, name='crop_spatial', **kwargs):
        super(crop_spatial, self).__init__(name=name, **kwargs)
        
    @tf.function
    def call(self, inputs, shape):
        in_shape = tf.shape(inputs)
        offset = [0,0] + [(in_shape[i] - shape[i]) // 2 for i in range(2, shape.shape[0])]
        size = tf.concat([in_shape[0:2],shape[2:]],0)
        inputs = tf.slice(inputs, offset, size)
        return inputs
