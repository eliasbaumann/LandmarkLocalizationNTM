import argparse
import os
import time

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
    pred = model.predict(img)
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

def store_results2(img, label, model, kp_list, i, path):
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

def train_unet(path, num_filters, fmap_inc_factor, ds_factors, kp_list=None, ntm=False):
    if kp_list == [0]:
        kp_list = None
    log_path, cp_path = create_dir(path)
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset(keypoints=kp_list)
    len_kp = (len(kp_list)-1) if kp_list is not None else 0
    unet_model = unet.unet2d(num_filters, fmap_inc_factor, ds_factors, dataset.n_landmarks-len_kp, ntm=ntm, batch_size=args.batch_size)
    #unet_model = unet.convnet2d(128, dataset.n_landmarks)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)    
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,verbose=1, save_weights_only=True, save_freq=args.checkpoint_interval*args.batch_size*args.num_training_iterations//args.num_epochs) #ugly way of saving every 5 epochs :)
    unet_model.compile(optimizer, loss = ssd_loss, metrics= [coord_dist])
    unet_model.save_weights(cp_path.format(epoch=0))
    unet_model.fit(x=dataset.data,
                   epochs=args.num_epochs,
                   validation_data=dataset.val_data,
                   steps_per_epoch=args.num_training_iterations//args.num_epochs,
                   validation_steps=10,
                   callbacks=[tb_callback, cp_callback]) 
    unet_model.summary()
    iterator = iter(dataset.test_data)
    for i in range(5): # TODO the problem might be that i can only predict a batch, not a single file anymore? because of previous_read_list?
        img, label = next(iterator)
        store_results(img, label, unet_model,kp_list, i, log_path)


def train_unet_custom(path, num_filters, fmap_inc_factor, ds_factors, kp_list=None, ntm=False):
    if kp_list == [0]:
        kp_list = None
    log_path, cp_path = create_dir(path)
    train_writer = tf.summary.create_file_writer(log_path+"\\train\\")
    val_writer = tf.summary.create_file_writer(log_path+"\\val\\")
    dataset = data.Data_Loader(args.dataset, args.batch_size)
    dataset(keypoints=kp_list)
    len_kp = (len(kp_list)-1) if kp_list is not None else 0
    unet_model = unet.unet2d(num_filters, fmap_inc_factor, ds_factors, dataset.n_landmarks-len_kp, ntm=ntm, batch_size=args.batch_size)

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
                val_loss.append(ssd_loss(lab_v, pred_v))
                val_coord_dist.append(coord_dist(lab_v, pred_v))
            tl_mean = tf.reduce_mean(train_loss)
            tcd_mean = tf.reduce_mean(train_coord_dist)
            vl_mean = tf.reduce_mean(val_loss)
            vcd_mean = tf.reduce_mean(val_coord_dist)
            elapsed_time = int(time.time() - start_time)
            tf.print("Iteration", step , "(Elapsed: ", elapsed_time, "s): Train: ssd: ", tl_mean, ", coord_dist: ", tcd_mean, ", Val: ssd: ", vl_mean, ", coord_dist: ", vcd_mean)
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
        if step % args.checkpoint_interval == 0:
            unet_model.save_weights(cp_path.format(step=step))
            print("saved cp-{:04d}.ckpt".format(step))
    test = iter(dataset.test_data)
    for j in range(args.num_test_samples):
        img, lab = next(test)
        store_results2(img, lab, unet_model, kp_list, j, log_path)

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