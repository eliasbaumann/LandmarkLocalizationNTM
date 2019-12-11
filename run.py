import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import dnccell
import data
import unet

import argparse

parser = argparse.ArgumentParser()

# Task
parser.add_argument('--dataset', type=str, default='droso', help='TODO')
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
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate.')
parser.add_argument('--optimizer_epsilon', type=float, default=1e-10,
                      help='Epsilon used for RMSProp optimizer.')

# Training options.
parser.add_argument('--num_training_iterations', type=int, default=100,
                        help='Number of iterations to train for.')
parser.add_argument('--report_interval', type=int, default=10,
                        help='Iterations between reports (samples, valid loss).')
parser.add_argument('--checkpoint_dir', type=str, default='/tmp/tf/dnc',
                       help='Checkpointing directory.')
parser.add_argument('--checkpoint_interval', type=int, default=-1,
                        help='Checkpointing step interval.')

args = parser.parse_args()

def vis_points(image, points, diameter=10):
    im = image.copy()

    for (x, y) in points:
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

# TODO: replace by pixelwise differnce
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

def train_unet(num_training_iterations, report_interval):
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset()
    # unet_model = unet.unet2d(128,2,[[2,2],[2,2],[2,2],[2,2]],dataset.n_landmarks)
    unet_model = unet.convnet2d(128, dataset.n_landmarks)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    
    n_epochs = 5 # TODO

    unet_model.compile(optimizer, loss = tf.keras.losses.MeanSquaredError(), metrics= [coord_dist])
    unet_model.fit(x=dataset.data, epochs=n_epochs, validation_data = dataset.val_data, steps_per_epoch=num_training_iterations//n_epochs, validation_steps=1) #TODO ntraining epochs
    # for epoch in range(n_epochs):
    #     for iteration in range(num_training_iterations):
    #         img, label = next(iterator)
    #         with tf.GradientTape() as tape:
    #             logits = unet_model(img)
    #             loss = loss_func(label, logits)
    #         grads = tape.gradient(loss, unet_model.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, unet_model.trainable_weights)) #issue: trainable weights is empty

    #         if(iteration % report_interval == 0):
    #             img, label = next(val_iterator)
    #             logits = unet_model(img)
    #             val_loss = coord_dist(label, logits)
    #             tf.print('train_loss:')
    #             tf.print(loss)
    #             tf.print('val_loss:')
    #             tf.print(val_loss)
    #             tf.print('---------')
    iterator = iter(dataset.data)
    for _ in range(5):
        img, label = next(iterator)
        pred = unet_model.predict(img)
        pred_keypoints = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), pred)
        lab_kp = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), label)
        vis_points(img.numpy().squeeze()[0], pred_keypoints.numpy()[0],5)
        plt.show()
        vis_points(img.numpy().squeeze()[0], lab_kp.numpy()[0], 5)
        plt.show()


def train_dnc(num_training_iterations, report_interval):
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset()
    iterator = iter(dataset.data)
    model = dnc_model(dataset.im_size) # TODO dataset has no outputsize anymore, because we predict heatmaps, we need to construct it differently now
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
    # tf.config.experimental_run_functions_eagerly(True)
    # train_dnc(args.num_training_iterations, args.report_interval)
    # tf.python.eager.profiler.start_profiler_server(6009)
    # logdir = "C:\\Users\\Elias\\Desktop\\log"
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=True)
    train_unet(args.num_training_iterations, args.report_interval)
    # with writer.as_default():
    #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)