import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import dnccell
import data
import unet

import argparse
import os

parser = argparse.ArgumentParser()

# Task
parser.add_argument('--dataset', type=str, default='droso', help='select dataset based on name (droso, cepha, ?hands?)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

# Model parameters
parser.add_argument('--hidden_size', type=int, default=64, help='Size of LSTM hidden layer.')
parser.add_argument('--hidden_layers', type=int, default=2, help='Number of LSTM hidden layers')
parser.add_argument('--memory_size', type=int, default=16, help='The number of memory slots.')
parser.add_argument('--word_size', type=int, default=16, help='The width of each memory slot.')
parser.add_argument('--num_write_heads', type=int, default=1, help='Number of memory write heads.')
parser.add_argument('--num_read_heads', type=int, default=4, help='Number of memory read heads.')
parser.add_argument('--clip_value', type=int, default=20,
                        help='Maximum absolute value of controller and dnc outputs.')

# Optimizer parameters.
parser.add_argument('--max_grad_norm', type=float, default=50, help='Gradient clipping norm limit.')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Optimizer learning rate.')
parser.add_argument('--optimizer_epsilon', type=float, default=1e-10,
                      help='Epsilon used for RMSProp optimizer.')

# Training options.
parser.add_argument('--num_training_iterations', type=int, default=1000,
                        help='Number of iterations to train for.')
parser.add_argument('--report_interval', type=int, default=10,
                        help='Iterations between reports (samples, valid loss).')
parser.add_argument('--checkpoint_dir', type=str, default='/tmp/tf/dnc',
                       help='Checkpointing directory.')
parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='Checkpointing step interval.')

args = parser.parse_args()

LOG_PATH = 'C:\\Users\\Elias\\Desktop\\MA_logs'
CP_PATH = LOG_PATH+'\\models\\cp-{epoch:04d}.ckpt'
CP_DIR = os.path.dirname(CP_PATH)

def vis_points(image, points, diameter=5, given_kp=None):
    im = image.copy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for (y, x) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (255, 0, 0), -1)
    if given_kp is not None:
        for (y,x) in given_kp:
            cv2.circle(im, (int(x), int(y)), diameter, (0, 255, 0), -1)
    plt.imshow(im)

class dnc_model(tf.keras.Model):
    def __init__(self, output_size):
        super(dnc_model, self).__init__()
        self.output_size = output_size
    
    def call(self, inputs):
        access_config = {
            "memory_size": args.memory_size,
            "word_size": args.word_size,
            "num_reads": args.num_read_heads,
            "num_writes": args.num_write_heads}
        controller_config = {
            "hidden_layers": args.hidden_layers,
            "hidden_size": args.hidden_size}
        clip_value = args.clip_value
        cell = dnccell.DNCCell(access_config, controller_config, self.output_size, clip_value)
        initial_state = cell.initial_state(args.batch_size)
        output_sequence, _ = tf.keras.layers.RNN(cell, time_major=True)(inputs, initial_state=initial_state)
        return output_sequence

def loss_func(gt_labels, logits):
    loss = tf.nn.l2_loss(gt_labels-logits) / args.batch_size
    return loss

def coord_dist(y_true, y_pred):
    y_pred = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), y_pred)
    y_true = tf.map_fn(lambda y: tf.map_fn(get_max_indices, y), y_true)
    return tf.keras.losses.MeanAbsoluteError()(y_true,y_pred)#tf.nn.l2_loss(y_true-y_pred) /args.batch_size


def get_max_indices(logits):
    coords = tf.squeeze(tf.where(tf.equal(logits, tf.reduce_max(logits))))
    coords = tf.cond(tf.greater(tf.rank(coords),tf.constant(1)),true_fn=lambda:tf.gather(coords,0),false_fn=lambda:coords)
    return tf.cast(coords, tf.float32)

def unstack_img_lab(img, kp_list):
    given_kp = tf.gather(img, tf.constant(range(1,len(kp_list))), axis=0)
    given_kp = tf.map_fn(get_max_indices, given_kp)
    image = tf.gather(img, tf.constant(0), axis=0)
    return image, given_kp

def vis_results(img,label,model,kp_list):
    given_kp = None
    # if kp_list is not None:
    #     given_kp = tf.gather(img, tf.constant(range(1,len(kp_list))), axis=0)
    #     given_kp = tf.map_fn(get_max_indices, given_kp)
    #     img = tf.gather(img, tf.constant(0), axis=0)
    img = tf.expand_dims(img, 0)

    label = tf.expand_dims(label, 0)
    pred = model.predict(img)
    if kp_list is not None:
        img, given_kp = unstack_img_lab(tf.squeeze(img), kp_list)

    pred_keypoints = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), pred)
    lab_kp = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), label)
    vis_points(img.numpy().squeeze(), pred_keypoints.numpy()[0], 5, given_kp)
    plt.show()
    vis_points(img.numpy().squeeze(), lab_kp.numpy()[0], 5, given_kp)
    plt.show()

def predict_from_cp(kp_list=None):
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset()
    latest = tf.train.latest_checkpoint(CP_DIR)
    unet_model = unet.unet2d(128,2,[[2,2],[2,2],[2,2],[2,2]],dataset.n_landmarks)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    unet_model.compile(optimizer, loss = loss_func, metrics= [coord_dist])
    unet_model.load_weights(latest)
    iterator = iter(dataset.test_data)
    for _ in range(5):
        img, label = next(iterator)
        vis_results(img, label, unet_model,kp_list)

def train_unet(num_training_iterations, kp_list=None):
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset(keypoints=kp_list)
    unet_model = unet.unet2d(128,2,[[2,2],[2,2],[2,2],[2,2]],dataset.n_landmarks-(len(kp_list)-1))
    #unet_model = unet.convnet2d(128, dataset.n_landmarks)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)    
    n_epochs = 40 # TODO
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CP_PATH,verbose=1, save_weights_only=True, save_freq=args.checkpoint_interval*args.batch_size*num_training_iterations//n_epochs) #ugly way of saving every 5 epochs :)
    unet_model.compile(optimizer, loss = loss_func, metrics= [coord_dist])
    unet_model.save_weights(CP_PATH.format(epoch=0))
    unet_model.fit(x=dataset.data,
                   epochs=n_epochs,
                   validation_data=dataset.val_data,
                   steps_per_epoch=num_training_iterations//n_epochs,
                   validation_steps=1,
                   callbacks=[tb_callback, cp_callback]) 
    iterator = iter(dataset.test_data)
    for _ in range(5):
        img, label = next(iterator)
        vis_results(img, label, unet_model,kp_list)


def train_dnc(num_training_iterations, report_interval):
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset()
    iterator = iter(dataset.data)
    model = dnc_model(dataset) # TODO dataset has no outputsize anymore, because we predict heatmaps, we need to construct it differently now
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate, epsilon=args.optimizer_epsilon)
    for iteration in range(num_training_iterations):
        observation, label = next(iterator)
        with tf.GradientTape() as tape:
            output_logits = model(observation)
            loss = loss_func(output_logits, label)
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, model.trainable_weights), args.max_grad_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        if iteration % report_interval == 0:
            print(loss)

if __name__ == "__main__":
    # train_dnc(args.num_training_iterations, args.report_interval)
    train_unet(args.num_training_iterations, kp_list = [0,1,2])
    # predict_from_cp()
