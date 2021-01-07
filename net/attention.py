import tensorflow as tf

class AttentionGate(tf.keras.layers.Layer):
    '''
    Attention Gates as defined by Oktay et al. 2018
    '''
    def __init__(self, n_filters, kernel_size, name='attn', **kwargs):
        super(AttentionGate, self).__init__(name=name, **kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        
        # wx+wg+bg
        self.conv_wx = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, data_format="channels_first", use_bias=False, name=self.name+"_attn_wx")
        self.conv_wg = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, data_format="channels_first", use_bias=True, name=self.name+"_attn_wg_")
        self.conv_psi = tf.keras.layers.Conv2D(filters=1, kernel_size=kernel_size, activation=tf.keras.activations.sigmoid, data_format="channels_first", use_bias=True, name=self.name+"_attn_psi_")
        self.activation = tf.keras.layers.ReLU(name=self.name+"_attn_relu_")

    def call(self, inputs, in_gate):
        wx = self.conv_wx(inputs)
        wg = self.conv_wg(in_gate)
        sigma_1 = self.activation(tf.add(wx,wg))
        alpha = self.conv_psi(sigma_1)
        return alpha * inputs, alpha



        