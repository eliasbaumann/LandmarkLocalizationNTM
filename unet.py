import tensorflow as tf
from enc_dec import Encoder_Decoder_Wrapper
from attention import AttentionGate

class unet2d(tf.keras.Model):
    def __init__(self, num_fmaps, fmap_inc_factor, downsample_factors, num_landmarks, seq_len=None, ntm_config=None, attn_config=None, batch_size=None, training=None, im_size=[256, 256], name='unet2d', **kwargs): 
        super(unet2d, self).__init__(name=name, **kwargs)
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.num_landmarks = num_landmarks
        self.ntm_config = ntm_config
        self.attn_config = attn_config
        self.batch_size = batch_size
        self.training = training
        self.im_size = im_size
        self.seq_len = seq_len
        self.unet_rec = unet(self.num_fmaps, self.fmap_inc_factor, self.downsample_factors, ntm_config=self.ntm_config, attn_config=self.attn_config, batch_size=self.batch_size, im_size=self.im_size)
        self.logits = conv_pass(1, self.num_landmarks, 1, activation=tf.keras.activations.tanh) #, payer et al do no activation

    def call(self, inputs, training=True):
        states = self.setup_states()
        out = []
        mem_out = []
        attn_out = []
        for i in range(self.seq_len):
            unet_2d, states, attn_maps = self.unet_rec(inputs[i], states, training)
            res = self.logits(unet_2d) 
            out.append(res)
            mem_out.append(states)
            attn_out.append(attn_maps)
        return tf.stack(out, axis=0), mem_out, attn_out

    def pred_test(self, inputs, training=False):
        states = self.setup_states()
        out = []
        mem_out = []
        attn_out = []
        lab = inputs[0,:,1:,:,:]
        img = inputs[0,:,:1,:,:]
        for _ in range(self.seq_len):
            unet_2d, states, attn_maps = self.unet_rec(tf.concat([img,lab], axis=1), states, training=training)
            res = self.logits(unet_2d) 
            out.append(res)
            mem_out.append(states)
            attn_out.append(attn_maps)
            lab = res
        return tf.stack(out, axis=0), mem_out, attn_out

    def setup_states(self):
        states = []
        _unet = self.unet_rec
        while _unet is not None:
            states = [*states, *_unet.get_initial_states()]
            _unet = _unet.unet_rec
        return states




class unet(tf.keras.layers.AbstractRNNCell):
    def __init__(self, num_fmaps, fmap_inc_factor, downsample_factors, ntm_config=None, attn_config=None, batch_size=None, activation=tf.nn.leaky_relu, layer=0, im_size=[256, 256], name='unet', **kwargs):
        super(unet, self).__init__(name=name+'_'+str(layer), **kwargs)
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factors = downsample_factors
        self.ntm_config = ntm_config
        self.attn_config = attn_config
        self.batch_size = batch_size
        self.activation = activation
        self.layer = layer
        self.im_size = im_size
        self.inp_conv = conv_pass(kernel_size=3,
                                  num_fmaps=self.num_fmaps,
                                  num_repetitions=2,
                                  activation=self.activation,
                                  name='unet_left_%i'%self.layer)
        self.ntm_l = None
        self.ntm_r = None
        self.add = tf.keras.layers.Add(name='ntm_out_add_%i' %self.layer)
        if self.ntm_config is not None:
            if self.layer in list(map(int, self.ntm_config.keys())):
                conf = self.ntm_config[str(self.layer)]
                assert self.batch_size is not None, 'Please set batch_size in unet2d init'
                assert conf["enc_dec_param"] is not None, "Please define parameters for the encoder-decoder part"
                if conf["enc_dec_param"]["pos"] == "l":
                    self.ntm_l = Encoder_Decoder_Wrapper(ntm_config=conf, batch_size=self.batch_size, layer=self.layer, name="ntm_l_"+str(self.layer))
                elif conf["enc_dec_param"]["pos"] == "r":
                    self.ntm_r = Encoder_Decoder_Wrapper(ntm_config=conf, batch_size=self.batch_size, layer=self.layer, name="ntm_r_"+str(self.layer))
                else:
                    self.ntm_l = Encoder_Decoder_Wrapper(ntm_config=conf, batch_size=self.batch_size, layer=self.layer, name="ntm_l_"+str(self.layer))
                    self.ntm_r = Encoder_Decoder_Wrapper(ntm_config=conf, batch_size=self.batch_size, layer=self.layer, name="ntm_r_"+str(self.layer))   
        
        self.attention = None
        if self.attn_config is not None:
            if self.layer in list(map(int, self.attn_config.keys())):
                atn = self.attn_config[str(self.layer)]
                self.attention = AttentionGate(n_filters=atn["num_filters"], kernel_size=1, layer=self.layer, name='attn_'+str(self.layer))
        
        if self.layer >= len(self.downsample_factors)-1:
            self.drop = tf.keras.layers.Dropout(.2,seed=42, name='dropout_%i'%self.layer)
        else:
            self.drop = None

        if self.layer < len(self.downsample_factors):
            self.unet_rec = unet(num_fmaps=self.num_fmaps*self.fmap_inc_factor,
                                 fmap_inc_factor=self.fmap_inc_factor,
                                 downsample_factors=self.downsample_factors,
                                 ntm_config=self.ntm_config,
                                 attn_config = self.attn_config,
                                 batch_size=self.batch_size,
                                 activation=self.activation,
                                 layer=self.layer+1,
                                 im_size=self.im_size)
            self.ds = downsample(factors=self.downsample_factors[self.layer], name='ds_%i'%self.layer)
            self.us = upsample(factors=self.downsample_factors[self.layer],
                               name='us_%i'%self.layer)
            self.crop = crop_spatial(name='crop_%i'%self.layer)
            self.out_conv = conv_pass(kernel_size=3, num_fmaps=self.num_fmaps, num_repetitions=2, activation=self.activation, name='unet_right_%i'%self.layer)
        else: 
            self.unet_rec = None
            self.ds = None
            self.us = None
            self.crop = None
            self.out_conv = None
    
    def call(self, inputs, prev_state, training=True):
        state_l = tf.constant(0.)
        state_r = tf.constant(0.)
        attn_map = tf.constant(0.)
        f_left = self.inp_conv(inputs)
        if self.ntm_l is not None:
            mem_l, state_l = self.ntm_l(f_left, prev_state[self.layer*2])
            f_left = tf.concat([mem_l, f_left], axis=1)
        # bottom layer:
        if self.layer == len(self.downsample_factors):
            f_left = self.drop(f_left, training=training)
            return f_left, [state_l], [attn_map]
        # to add dropout to second to last layer as well: 
        elif self.layer == len(self.downsample_factors)-1:
            f_left = self.drop(f_left, training=training)
        g_in = self.ds(f_left)
        g_out, state_rec, attn_rec = self.unet_rec(g_in, prev_state)

        g_out_upsampled = self.us(g_out)
        f_left_cropped = self.crop(f_left, tf.shape(g_out_upsampled))
        
        
        if self.attention is not None:
            f_left_cropped, attn_map = self.attention(g_out_upsampled, f_left_cropped)

        f_right = tf.concat([f_left_cropped, g_out_upsampled],1)
        if self.ntm_r is not None:
            mem_r, state_r = self.ntm_r(f_right, prev_state[self.layer*2+1])
            f_right = tf.concat([mem_r, f_right], axis=1)
        f_out = self.out_conv(f_right)
        return f_out, [state_l, state_r, *state_rec], [attn_map, *attn_rec]

    def get_config(self):
        config = super(unet, self).get_config()
        config.update({'num_fmaps':self.num_fmaps, 'fmap_inc_factor':self.fmap_inc_factor, 'downsample_factors':self.downsample_factors, 'ntm_config':self.ntm_config, 'batch_size':self.batch_size, 'activation':self.activation, 'layer':self.layer})
        return config

    def get_initial_states(self):
        state_l = tf.constant(0.)
        state_r = tf.constant(0.)
        if self.ntm_l is not None:
            if self.ntm_l.cell is not None:
                state_l = self.ntm_l.cell.get_initial_state()
        if self.ntm_r is not None:
            if self.ntm_r.cell is not None:
                state_r = self.ntm_r.cell.get_initial_state()
        return state_l, state_r


        

class conv_pass(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_fmaps, num_repetitions, activation=tf.nn.relu, name='conv_pass', **kwargs):
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

    def get_config(self):
        config = super(conv_pass,self).get_config()
        config.update({'kernel_size':self.kernel_size, 'num_fmaps':self.num_fmaps, 'num_repetitions':self.num_repetitions, 'activation':self.activation})
        return config

class downsample(tf.keras.layers.Layer):
    def __init__(self, factors, name='ds', **kwargs):
        super(downsample, self).__init__(name = name, **kwargs)
        self.factors = factors
        self.ds = tf.keras.layers.AveragePooling2D(pool_size=self.factors, strides=self.factors, padding='same', data_format='channels_first', name=self.name+'_internal')
    
    def call(self, inputs):
        inputs = self.ds(inputs)
        return inputs

    def get_config(self):
        config = super(downsample, self).get_config()
        config.update({'factors':self.factors})
        return config

class upsample(tf.keras.layers.Layer):
    def __init__(self, factors, name='us', **kwargs):
        super(upsample, self).__init__(name=name, **kwargs)
        self.factors = factors
        self.us = tf.keras.layers.UpSampling2D(size=self.factors, data_format = "channels_first", name=name+'_internal')
    
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
        
    def call(self, inputs, shape):
        in_shape = tf.shape(inputs)
        offset = [0,0] + [(in_shape[i] - shape[i]) // 2 for i in range(2, shape.shape[0])]
        size = tf.concat([in_shape[0:2],shape[2:]],0)
        inputs = tf.slice(inputs, offset, size)
        return inputs
