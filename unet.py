import tensorflow as tf


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
    def __init__(self, num_fmaps, fmap_inc_factor, downsample_factors, num_landmarks, name='unet2d', **kwargs):
        super(unet2d, self).__init__(name=name, **kwargs)
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.num_landmarks = num_landmarks
        self.unet = unet(self.num_fmaps, self.fmap_inc_factor, self.downsample_factors)
        self.logits = conv_pass(1, self.num_landmarks, 1, activation=tf.keras.activations.tanh)

    def call(self, inputs):
        unet_2d = self.unet(inputs)
        res = self.logits(unet_2d) # TODO payer et al do no activation ?
        return res

class unet(tf.keras.layers.Layer):
    def __init__(self, num_fmaps, fmap_inc_factor, downsample_factors, activation=tf.keras.activations.relu, layer=0, name='unet', **kwargs):
        super(unet, self).__init__(name=name, **kwargs)
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.activation = activation
        self.layer = layer
        self.inp_conv = conv_pass(
                       kernel_size=3,
                       num_fmaps=self.num_fmaps,
                       num_repetitions=2,
                       activation=self.activation,
                       name='unet_left_%i'%self.layer)
        
        if (self.layer < len(self.downsample_factors)):
            self.unet_rec = unet(num_fmaps=self.num_fmaps*self.fmap_inc_factor, 
                            fmap_inc_factor=self.fmap_inc_factor, 
                            downsample_factors=self.downsample_factors, 
                            activation=self.activation, 
                            layer=self.layer+1)
            self.ds = downsample(factors=self.downsample_factors[self.layer])
            self.us = upsample(factors=self.downsample_factors[self.layer],
                                    num_fmaps=self.num_fmaps,
                                    activation=self.activation)
            self.crop = crop_spatial()
            self.out_conv = conv_pass(kernel_size=3, num_fmaps=self.num_fmaps, num_repetitions=2)
        else: 
            self.unet_rec = None
            self.ds = None
            self.us = None
            self.crop = None
            self.out_conv = None

        
    
    def call(self, inputs):
        f_left = self.inp_conv(inputs)
        # bottom layer:
        if (self.layer == len(self.downsample_factors)):
            return f_left
        g_in = self.ds(f_left)
        g_out = self.unet_rec(g_in)
        g_out_upsampled = self.us(g_out)
        f_left_cropped = self.crop(f_left,tf.shape(g_out_upsampled))
        f_right = tf.concat([f_left_cropped, g_out_upsampled],1)
        f_out = self.out_conv(f_right)
        return f_out

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
    
    def call(self, inputs):
        for i in range(self.num_repetitions):
            inputs = self.conv[i](inputs)
        return inputs

class downsample(tf.keras.layers.Layer):
    def __init__(self, factors, name='ds', **kwargs):
        super(downsample, self).__init__(name = name, **kwargs)
        self.factors = factors
        self.ds = tf.keras.layers.MaxPool2D(pool_size=self.factors, strides=self.factors, padding='same', data_format='channels_first', name=self.name)
    
    
    def call(self, inputs):
        inputs = self.ds(inputs)
        return inputs

class upsample(tf.keras.layers.Layer):
    def __init__(self, factors, num_fmaps, activation=tf.keras.activations.relu, name='us', **kwargs):
        super(upsample, self).__init__(name=name, *kwargs)
        self.factors = factors
        self.num_fmaps = num_fmaps
        self.activation = activation
        self.us = tf.keras.layers.Conv2DTranspose(filters=self.num_fmaps,
                                            kernel_size=self.factors,
                                            strides=self.factors,
                                            padding='valid',
                                            data_format='channels_first',
                                            activation=self.activation,
                                            name=self.name)

    def call(self, inputs):
        inputs = self.us(inputs)
        return inputs

class crop_spatial(tf.keras.layers.Layer):
    def __init__(self):
        super(crop_spatial, self).__init__(name='crop_spatial')
        
    
    def call(self, inputs, shape):
        in_shape = tf.shape(inputs)
        offset = [0,0] + [(in_shape[i] - shape[i]) // 2 for i in range(2, shape.shape[0])]
        size = tf.concat([in_shape[0:2],shape[2:]],0)
        inputs = tf.slice(inputs, offset, size)
        return inputs