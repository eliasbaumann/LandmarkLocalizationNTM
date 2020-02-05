## Wrap NTM with encoder decoder network s.t. the ntm is fed vectors and not images.

# pool_size should be 4 for 256 -> 64 -> 16 -> 4

class Encoder_Decoder_Wrapper(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, pool_size):
        super(Encoder_Decoder_Wrapper, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=self.kernel_size, activation='relu', padding='same')
        self.conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size, activation='relu', padding='same')
        self.ds = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_size)
        self.us = tf.keras.layers.UpSampling2D(size=pool_size)


    def call(self, inputs):
        # TODO:
        # - Apply encoding with convnet
        # - feed encoded vector as input into NTM
        # - apply decoding to output (but only to the not controller params part? need to check this)
        # return decoded image in same format as input
        x = self.conv(inputs)
        x = self.ds(x)
        x = self.conv(x)
        x = self.ds(x)
        x = self.conv_1(x)
        x = self.ds(x)
        x = tf.reshape(x, [-1])

        # NTM here

        x = tf.reshape(ntm_out, [1,4,4]) # we on
        x = self.conv(x)
        x = self.us(x)
        x = self.conv(x)
        x = self.us(x)
        x = self.conv(x)
        x = self.us(x)
        x = self.conv_1(x)
        return x
