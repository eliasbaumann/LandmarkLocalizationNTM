import argparse
import os
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import data
import unet



parser = argparse.ArgumentParser()

# Task
parser.add_argument('--dataset', type=str, default='droso', help='select dataset based on name (droso, cepha, ?hands?)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--num_test_samples', type=int, default=5, help='Number of samples from test to predict and save')

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
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Optimizer learning rate.')

# Training options.
parser.add_argument('--num_training_iterations', type=int, default=10000,
                        help='Number of iterations to train for.')
parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of iterations to train for.')
parser.add_argument('--validation_steps', type=int, default=5,
                        help='Number of validation steps after every epoch.')
parser.add_argument('--report_interval', type=int, default=50,
                        help='Iterations between reports (samples, valid loss).')
parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Checkpointing step interval.')

args = parser.parse_args()

def vis_points(image, points, diameter=5, given_kp=None):
    im = image.copy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for (y, x) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (255, 0, 0), -1)
    if given_kp is not None:
        for (y,x) in given_kp:
            cv2.circle(im, (int(x), int(y)), diameter, (0, 255, 0), -1)
    plt.imshow(im)

@tf.function
def ssd_loss(gt_labels, logits):
    loss = tf.nn.l2_loss(gt_labels-logits) / args.batch_size
    return loss

@tf.function
def coord_dist(y_true, y_pred):
    y_pred = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), y_pred)
    y_true = tf.map_fn(lambda y: tf.map_fn(get_max_indices, y), y_true)
    return tf.keras.losses.MeanAbsoluteError()(y_true,y_pred)#tf.nn.l2_loss(y_true-y_pred) /args.batch_size

@tf.function
def get_max_indices(logits):
    coords = tf.squeeze(tf.where(tf.equal(logits, tf.reduce_max(logits))))
    coords = tf.cond(tf.greater(tf.rank(coords),tf.constant(1)),true_fn=lambda:tf.gather(coords,0),false_fn=lambda:coords)
    return tf.cast(coords, tf.float32)

@tf.function
def per_kp_stats(y_true, y_pred, margin):
    y_pred = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), y_pred)
    y_true = tf.map_fn(lambda y: tf.map_fn(get_max_indices, y), y_true)
    exp_y_pred = tf.expand_dims(y_pred, 1)
    exp_y_true = tf.expand_dims(y_true, 2)
    exp_closest = tf.argmin(tf.reduce_mean(tf.square(tf.abs(tf.subtract(exp_y_pred, exp_y_true))), axis=-1), axis=-1)
    closest_to_nearest = tf.reduce_mean(tf.cast(tf.equal(exp_closest, tf.range(tf.shape(exp_closest)[-1], dtype=tf.int64)), dtype=tf.float32), axis=0)
    within_margin = tf.reduce_mean(tf.cast(tf.reduce_all(tf.greater_equal(margin, tf.abs(tf.subtract(y_true, y_pred))), axis=2), tf.float32), axis=0)
    return within_margin, closest_to_nearest

@tf.function
def unstack_img_lab(img, kp_list):
    given_kp = tf.gather(img, tf.constant([0]), axis=0)    
    given_kp = tf.gather(img, tf.constant(range(1,len(kp_list))), axis=0)
    given_kp = tf.map_fn(get_max_indices, given_kp)
    image = tf.gather(img, tf.constant(0), axis=0)
    return image, given_kp

def store_results(img, label, model, kp_list, i, path):
    given_kp = None
    img = tf.expand_dims(img, 0)
    label = tf.expand_dims(label, 0)
    pred = model(img)
    if kp_list is not None:
        img, given_kp = unstack_img_lab(tf.squeeze(img), kp_list)
    pred_keypoints = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), pred)
    lab_kp = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), label)
    vis_points(img.numpy().squeeze(), pred_keypoints.numpy()[0], 5, given_kp)
    if not os.path.exists(path+'\\samples\\'):
        os.makedirs(path+'\\samples\\')
    plt.savefig(path+'\\samples\\%02d_pred.png' % i)
    vis_points(img.numpy().squeeze(), lab_kp.numpy()[0], 5, given_kp)
    plt.savefig(path+'\\samples\\%02d_gt.png' % i)

# TODO Switch over to custom loop
def predict_from_cp(path, run_number, num_filters, fmap_inc_factor, ds_factors, kp_list=None, ntm=False):
    if kp_list == [0]:
        kp_list = None
    log_dir, cp_dir = load_dir(path, run_number)
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset(keypoints=kp_list)
    latest = tf.train.latest_checkpoint(cp_dir)
    len_kp = (len(kp_list)-1) if kp_list is not None else 0
    unet_model = unet.unet2d(num_filters, fmap_inc_factor, ds_factors, dataset.n_landmarks-len_kp, ntm=ntm, batch_size=args.batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    unet_model.compile(optimizer, loss=ssd_loss, metrics= [coord_dist])
    unet_model.load_weights(latest)
    iterator = iter(dataset.test_data)
    for i in range(5):
        img, label = next(iterator)
        store_results(img, label, unet_model, kp_list, i, log_dir)

def train_unet_custom(path, num_filters, fmap_inc_factor, ds_factors, kp_list=None, ntm=False, load_from_cp=False, run_number=None, kp_metric_margin=3):
    if kp_list == [0]:
        kp_list = None
    kp_margin = tf.constant(kp_metric_margin, dtype=tf.float32)
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset(keypoints=kp_list)
    len_kp = (len(kp_list)-1) if kp_list is not None else 0
    unet_model = unet.unet2d(num_filters, fmap_inc_factor, ds_factors, dataset.n_landmarks-len_kp, ntm=ntm, batch_size=args.batch_size)
    if load_from_cp:
        log_path, cp_path = load_dir(path, run_number)
        latest = tf.train.latest_checkpoint(log_path+"\\cp\\")
        unet_model.load_weights(latest)
    else:
        log_path, cp_path = create_dir(path)
    train_writer = tf.summary.create_file_writer(log_path+"\\train\\")
    val_writer = tf.summary.create_file_writer(log_path+"\\val\\")
    @tf.function
    def predict(img):
        return unet_model(img)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    train = iter(dataset.data)
    val = iter(dataset.val_data)
    train_loss = []
    val_loss = []
    train_coord_dist = []
    val_coord_dist = []
    mrg = []
    cgt = []
    tf.print("Starting train loop...")
    start_time = time.time()
    for step in range(args.num_training_iterations):
        with tf.GradientTape() as tape:
            img, lab = next(train)
            pred = predict(img)
            loss = ssd_loss(lab, pred)
        grad = tape.gradient(loss, unet_model.trainable_weights)
        optimizer.apply_gradients(zip(grad, unet_model.trainable_weights)) # tf.clip_by_global_norm
        train_loss.append(loss)
        train_coord_dist.append(coord_dist(lab, pred))
        
        if step % args.report_interval == 0:
            for _ in range(args.validation_steps):
                img_v, lab_v = next(val)
                pred_v = predict(img_v)
                within_margin, closest_to_gt = per_kp_stats(lab_v, pred_v, kp_margin)
                mrg.append(within_margin)
                cgt.append(closest_to_gt)
                val_loss.append(ssd_loss(lab_v, pred_v))
                val_coord_dist.append(coord_dist(lab_v, pred_v))
            tl_mean = tf.reduce_mean(train_loss)
            tcd_mean = tf.reduce_mean(train_coord_dist)
            vl_mean = tf.reduce_mean(val_loss)
            vcd_mean = tf.reduce_mean(val_coord_dist)
            mrg_mean = tf.reduce_mean(mrg, axis=0)
            cgt_mean = tf.reduce_mean(cgt, axis=0)
            
            elapsed_time = int(time.time() - start_time)
            tf.print("Iteration", step , "(Elapsed: ", elapsed_time, "s): Train: ssd: ", tl_mean, ", coord_dist: ", tcd_mean, ", Val: ssd: ", vl_mean, ", coord_dist: ", vcd_mean)
            tf.print("% within margin: ", mrg_mean, summarize=-1)
            with open(os.path.join(log_path,'within_margin.txt'), 'ab') as mrgtxt:
                np.savetxt(mrgtxt, [np.array(mrg_mean)], fmt='%3.3f', delimiter=",")
            mrgtxt.close()
            tf.print("% closest to gt", cgt_mean, summarize=-1)
            with open(os.path.join(log_path,'closest_gt.txt'), 'ab') as cgttxt:
                np.savetxt(cgttxt, [np.array(cgt_mean)], fmt='%3.3f', delimiter=",")
            cgttxt.close()
            with train_writer.as_default():
                tf.summary.scalar("ssd_loss", tl_mean, step=step)
                tf.summary.scalar("coord_dist", tcd_mean, step=step)
                train_writer.flush()
            with val_writer.as_default():
                tf.summary.scalar("ssd_loss", tl_mean, step=step)
                tf.summary.scalar("coord_dist", vcd_mean, step=step)
                val_writer.flush()
            train_loss = []
            val_loss = []
            train_coord_dist = []
            val_coord_dist = []
            mrg = []
            cgt = []
        if step % args.checkpoint_interval == 0:
            unet_model.save_weights(cp_path.format(step=step))
            print("saved cp-{:05d}.ckpt".format(step))
    test = iter(dataset.test_data)
    for j in range(args.num_test_samples):
        img, lab = next(test)
        store_results(img, lab, unet_model, kp_list, j, log_path)

# TODO rework this, so its consistent.
def create_dir(path):
    previous_runs = os.listdir(path)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

    logdir = 'run_%02d' % run_number
    l_dir = os.path.join(path, logdir)
    cp_dir = l_dir +'\\cp\\cp-{step:04d}.ckpt'
    return l_dir, cp_dir

def load_dir(path, run_number):
    logdir = 'run_%02d' % run_number
    l_dir = os.path.join(path, logdir)
    cp_pth = l_dir+'\\cp\\cp-{step:04d}.ckpt'
    cp_dir = os.path.dirname(cp_pth)
    return l_dir, cp_dir


if __name__ == "__main__":
    PATH = 'C:\\Users\\Elias\\Desktop\\MA_logs'
    train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], kp_list=None, ntm=False)
    # predict_from_cp(PATH, 3, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2]], kp_list = None, ntm=False)


# kp_list: 0 is image, remaining numpers are keypoints. If you dont want to include keypoints in input, set to None