import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def _generate_heatmap(coords, imx, sigma=3): #square        
    X,Y = tf.meshgrid(tf.range(0,imx,1.), tf.range(0,imx,1.))
    idx = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y,[-1,1])], axis =1)
    gaussian = tfp.distributions.MultivariateNormalDiag(loc=coords, scale_diag=tf.ones(tf.shape(coords),tf.float32)*sigma)
    prob = tf.reshape(gaussian.prob(idx),tf.shape(X))
    prob = tf.transpose(prob) # to make it HW instead of WH 
    prob = tf.expand_dims(prob,0)
    return prob 

@tf.function
def generate_heatmaps(coord_list, imx, n_landmarks, sigma):
    hm_list = tf.map_fn(lambda x: _generate_heatmap(x, imx, sigma), coord_list) # 3 = size of gaussian blob
    paddings = [[0,n_landmarks-tf.shape(hm_list)[0]],[0,0],[0,0],[0,0]] 
    hm_list = tf.pad(hm_list, paddings , 'CONSTANT', 0.)
    hm_list = tf.image.per_image_standardization(hm_list)
    return tf.squeeze(hm_list)

