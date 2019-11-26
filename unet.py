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

    def call(self, inputs):
        unet_2d = unet(self.num_fmaps, self.fmap_inc_factor, self.downsample_factors)(inputs)
        logits = conv_pass(1, self.num_landmarks, 1, activation=None)(unet_2d) # TODO payer et al do no activation ?
        return logits

class unet(tf.keras.layers.Layer):
    def __init__(self, num_fmaps, fmap_inc_factor, downsample_factors, activation=tf.keras.activations.relu, layer=0, name='unet', **kwargs):
        super(unet, self).__init__(name=name, **kwargs)
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.activation = activation
        self.layer = layer
    
    def call(self, inputs):
        f_left = conv_pass(
                       kernel_size=3,
                       num_fmaps=self.num_fmaps,
                       num_repetitions=2,
                       activation=self.activation,
                       name='unet_left_%i'%self.layer)(inputs)
        
        # bottom layer:
        if (self.layer == len(self.downsample_factors)):
            return f_left
        
        g_in = downsample(f_left, self.downsample_factors[self.layer])

        g_out = unet(num_fmaps=self.num_fmaps*self.fmap_inc_factor, 
                    fmap_inc_factor=self.fmap_inc_factor, 
                    downsample_factors=self.downsample_factors, 
                    activation=self.activation, 
                    layer=self.layer+1)(g_in)

        g_out_upsampled = upsample(g_out,
                                factors=self.downsample_factors[self.layer],
                                num_fmaps=self.num_fmaps,
                                activation=self.activation)

        f_left_cropped = crop_spatial(f_left, g_out_upsampled.get_shape().as_list())

        f_right = tf.concat([f_left_cropped, g_out_upsampled],1)

        f_out = conv_pass(kernel_size=3, num_fmaps=self.num_fmaps, num_repetitions=2)(f_right)

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
# def conv_pass(fmaps_in, kernel_size, num_fmaps, num_repetitions, activation=tf.keras.activations.relu, name='conv_pass'):
#     fmaps = fmaps_in
#     for i in range(num_repetitions):
#         fmaps = tf.keras.layers.Conv2D(filters=num_fmaps, 
#                                        kernel_size=kernel_size, 
#                                        padding='same', 
#                                        data_format='channels_first', 
#                                        activation=activation, 
#                                        name=name+'_%i'%i)(fmaps)
#     return fmaps

def downsample(fmaps_in, factors, name='ds'):
    fmaps = tf.keras.layers.MaxPool2D(pool_size=factors, strides=factors, padding='valid', data_format='channels_first', name=name)(fmaps_in)
    return fmaps

def upsample(fmaps_in, factors, num_fmaps, activation=tf.keras.activations.relu, name='us'):
    fmaps = tf.keras.layers.Conv2DTranspose(filters=num_fmaps,
                                            kernel_size=factors,
                                            strides=factors,
                                            padding='valid',
                                            data_format='channels_first',
                                            activation=activation,
                                            name=name)(fmaps_in)
    return fmaps

def crop_spatial(fmaps_in, shape):
    in_shape = fmaps_in.get_shape().as_list()

    offset = [0, 0] + [(in_shape[i] - shape[i]) // 2 for i in range(2, len(shape))]
    size = in_shape[0:2] + shape[2:]

    fmaps = tf.slice(fmaps_in, offset, size)

    return fmaps

# def unet(fmaps_in, num_fmaps, fmap_inc_factor, downsample_factors, activation=tf.keras.activations.relu, layer=0):
#     f_left = conv_pass(fmaps_in,
#                        kernel_size=3,
#                        num_fmaps=num_fmaps,
#                        num_repetitions=2,
#                        activation=activation,
#                        name='unet_left_%i'%layer)
    
#     # bottom layer:
#     if (layer == len(downsample_factors)):
#         return f_left
    
#     g_in = downsample(f_left, downsample_factors[layer])

#     g_out = unet(g_in, 
#                  num_fmaps=num_fmaps*fmap_inc_factor, 
#                  fmap_inc_factor=fmap_inc_factor, 
#                  downsample_factors=downsample_factors, 
#                  activation=activation, 
#                  layer=layer+1)

#     g_out_upsampled = upsample(g_out,
#                                factors=downsample_factors[layer],
#                                num_fmaps=num_fmaps,
#                                activation=activation)

#     f_left_cropped = crop_spatial(f_left, g_out_upsampled.get_shape().as_list())

#     f_right = tf.concat([f_left_cropped, g_out_upsampled],1)

#     f_out = conv_pass(f_right, kernel_size=3, num_fmaps=num_fmaps, num_repetitions=2)

#     return f_out
