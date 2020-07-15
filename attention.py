import tensorflow as tf


class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, layer, name='attn', **kwargs):
        super(AttentionGate, self).__init__(name=name, **kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.layer = layer
        
        # wx+wg+bg
        self.conv_wx = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, data_format="channels_first", use_bias=False, name="attn_wx_"+str(self.layer))
        self.conv_wg = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, data_format="channels_first", use_bias=True, name="attn_wg_"+str(self.layer))
        self.conv_psi = tf.keras.layers.Conv2D(filters=1, kernel_size=kernel_size, activation=tf.keras.activations.sigmoid, data_format="channels_first", use_bias=True, name="attn_psi_"+str(self.layer))
        self.activation = tf.keras.layers.ReLU(name="attn_relu_"+str(self.layer))

    def call(self, inputs, in_gate):
        wx = self.conv_wx(inputs)
        wg = self.conv_wg(in_gate)
        sigma_1 = self.activation(tf.add(wx,wg))
        alpha = self.conv_psi(sigma_1)
        return alpha * inputs, alpha



        