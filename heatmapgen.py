import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

class Heatmap_Generator(object):
    def __init__(self, image_shape, sigma):
        self.image_shape = image_shape
        self.sigma = sigma
        

    def _generate_heatmap(self, coords):        
        X,Y = tf.meshgrid(tf.range(0,self.image_shape[0],1.), tf.range(0,self.image_shape[1],1.))
        idx = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y,[-1,1])], axis =1)
        gaussian = tfp.distributions.MultivariateNormalDiag(loc=coords, scale_diag=tf.ones(tf.shape(coords),tf.float32)*self.sigma)
        prob = tf.reshape(gaussian.prob(idx),tf.shape(X))
        return prob

    def generate_heatmaps(self, coord_list):
        hm_list = [self._generate_heatmap(x) for x in coord_list]
        return tf.stack(hm_list,axis=0)



# if __name__ == "__main__":
#     hmg = Heatmap_Generator([100,100],2.)
#     hm = hmg.generate_heatmaps([[4,4],[40,60]])
#     plt.matshow(hm[1])
#     plt.show()