import tensorflow as tf


class unet_base(object):
    def __init__(self): # TODO
        pass

    def conv_pass(self, fmaps_in, kernel_size, num_fmaps, num_repetitions, activation=tf.keras.activations.relu, name='conv_pass'):
        fmaps = fmaps_in
        for i in range(num_repetitions):
            fmaps = tf.keras.layers.Conv2D(filters=num_fmaps, kernel_size=kernel_size, padding='valid', data_format='channels_first', activation=activation, name=name+'_%i'%i)(fmaps) #TODO channels correct?
        return fmaps

    def downsample(self,fmaps_in, factors, name='ds'):
        fmaps = tf.keras.layers.MaxPool2D(pool_size=factors, strides=factors, padding='valid', data_format='channels_first', name=name)(fmaps_in)
        return fmaps

    def upsample(self,fmaps_in, factors, num_fmaps, activation=tf.keras.activations.relu, name='us'):
        fmaps = tf.keras.layers.Conv2DTranspose(filters=num_fmaps, kernel_size=factors, strides=factors, padding='valid', data_format='channels_first', activation=activation, name=name)(fmaps_in)
        return fmaps

    def crop_spatial(self, fmaps_in, shape):
        in_shape = fmaps_in.get_shape().as_list()

        offset = [0, 0] + [(in_shape[i] - shape[i]) // 2 for i in range(2, len(shape))]
        size = in_shape[0:2] + shape[2:]

        fmaps = tf.slice(fmaps_in, offset, size)

        return fmaps

    def unet(self, fmaps_in, num_fmaps, fmap_inc_factor, downsample_factors, activation=tf.keras.activations.relu, layer=0):
        f_left = self.conv_pass(fmaps_in, kernel_size=3, num_fmaps=num_fmaps, num_repetitions=2, activation=activation, name='unet_left_%i'%layer)
        
        # bottom layer:
        if (layer == len(downsample_factors)):
            return f_left
        
        g_in = self.downsample(f_left, downsample_factors[layer])

        g_out = self.unet(g_in, num_fmaps=num_fmaps*fmap_inc_factor, fmap_inc_factor=fmap_inc_factor, downsample_factors=downsample_factors, activation=activation, layer=layer+1)

        g_out_upsampled = self.upsample(g_out, factors = downsample_factors[layer],num_fmaps=num_fmaps, activation=activation)

        f_left_cropped = self.crop_spatial(f_left, g_out_upsampled.get_shape().as_list())

        f_right = tf.concat([f_left_cropped, g_out_upsampled],1)

        f_out = self.conv_pass(f_right, kernel_size=3, num_fmaps=num_fmaps, num_repetitions=2)

        return f_out
