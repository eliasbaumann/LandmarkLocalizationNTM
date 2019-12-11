import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

class Heatmap_Generator(object):
    def __init__(self, image_shape, n_landmarks, sigma):
        self.image_shape = image_shape
        self.n_landmarks = n_landmarks
        self.sigma = sigma
        
    def _rescale(self, img, epsilon):
        e = tf.constant(epsilon, dtype=tf.float32)
        img = tf.divide(tf.add(e, img), tf.constant(epsilon+1., dtype=tf.float32))
        img = tf.subtract(tf.scalar_mul(tf.constant(2., dtype=tf.float32), img),tf.constant(1., dtype=tf.float32))
        return img

    def _generate_heatmap(self, coords):        
        X,Y = tf.meshgrid(tf.range(0,self.image_shape[0],1.), tf.range(0,self.image_shape[1],1.))
        idx = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y,[-1,1])], axis =1)
        gaussian = tfp.distributions.MultivariateNormalDiag(loc=coords, scale_diag=tf.ones(tf.shape(coords),tf.float32)*self.sigma)
        prob = tf.reshape(gaussian.prob(idx),tf.shape(X))
        #prob = self._rescale(prob, .001)
        prob = tf.expand_dims(prob,0)
        return tf.transpose(prob)
    
    def generate_heatmaps(self, coord_list):
        hm_list = tf.map_fn(self._generate_heatmap, coord_list)
        paddings = [[0,self.n_landmarks-tf.shape(hm_list)[0]],[0,0],[0,0],[0,0]] # TODO maybe we can skip this
        hm_list = tf.pad(hm_list, paddings , 'CONSTANT', 0.)
        hm_list = tf.image.per_image_standardization(hm_list)
        return tf.squeeze(hm_list)#tf.squeeze(hm_list)#tf.stack(hm_list,axis=0)

