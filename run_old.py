import argparse
import os
import time
import random
import json

import numpy as np
import tensorflow as tf
from cyclic_learning import ExponentialCyclicalLearningRate# from tensorflow_addons.optimizers import ExponentialCyclicalLearningRate

import matplotlib.pyplot as plt
import cv2

import data
import unet

from ntm_configs import CONF_POS_LIST, CONF_MEM_LIST

parser = argparse.ArgumentParser()

strategy = tf.distribute.MirroredStrategy()

unet_model = None
GLOBAL_BATCH_SIZE = None # has to be set

physical_devices = tf.config.list_physical_devices('GPU')
try:
  [tf.config.experimental.set_memory_growth(i, True) for i in physical_devices]
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# tf.config.experimental_run_functions_eagerly(True)

# Task
parser.add_argument('--dataset', type=str, default='droso', help='select dataset based on name (droso, cepha, ?hands?)')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_test_samples', type=int, default=5, help='Number of samples from test to predict and save')

# Optimizer parameters.
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate.') # TODO figure something out here, maybe cyclic learning rate to get out of local minima?

# Training options.
parser.add_argument('--num_training_iterations', type=int, default=1,
                        help='Number of iterations to train for.')
parser.add_argument('--validation_steps', type=int, default=5,
                        help='Number of validation steps after every epoch.')
parser.add_argument('--report_interval', type=int, default=50,
                        help='Iterations between reports (samples, valid loss).')
parser.add_argument('--checkpoint_interval', type=int, default=500,
                        help='Checkpointing step interval.')

args = parser.parse_args()


def vis_points(image, points, diameter=5, given_kp=None):
    im = image.copy() # h,w
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for (w, h) in points:
        cv2.circle(im, (int(h), int(w)), diameter, (255, 0, 0), -1)
    if given_kp is not None:
        for (w, h) in given_kp:
            cv2.circle(im, (int(h), int(w)), diameter, (0, 255, 0), -1)
    plt.imshow(im)

@tf.function
def ssd_loss(gt_labels, logits):
    loss = tf.nn.l2_loss(gt_labels-logits) / args.batch_size
    return loss

with strategy.scope():
    def dist_ssd_loss(gt_labels, logits):
        return tf.reduce_sum(tf.square(gt_labels-logits), axis=None)
    
    def compute_loss(gt_labels, logits):
        per_example_loss = dist_ssd_loss(gt_labels, logits)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE) 

    def dist_coord_dist(gt_labels, logits):
        logits = tf.cast(get_max_indices_argmax(logits), tf.float32)
        gt_labels = tf.cast(get_max_indices_argmax(gt_labels), tf.float32)
        loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.abs(gt_labels-logits)), axis=-1)), axis=-1) #pythagoras, mean batch, not entirely accurate for pixel images?
        return loss * (1./GLOBAL_BATCH_SIZE)

    def dist_per_kp_stats_iter(gt_labels, logits, margin):
        y_pred_n = get_max_indices_argmax(logits)
        y_true_n = get_max_indices_argmax(gt_labels)
        exp_y_pred = tf.expand_dims(y_pred_n, 0)
        exp_y_true = tf.expand_dims(y_true_n, 1)
        closest = tf.square(tf.subtract(exp_y_pred, exp_y_true)) # using TF broadcast to create distance table
        closest_red = tf.argmin(tf.reduce_mean(closest, axis=-1), axis=1) # find min distance
        closest_to_nearest = tf.reduce_sum(tf.cast(tf.equal(tf.transpose(tf.cast(closest_red, tf.int32)), tf.range(tf.shape(closest_red)[0], dtype=tf.int32)), dtype=tf.float32), axis=0) * (1./GLOBAL_BATCH_SIZE)
        within_margin = tf.reduce_sum(tf.cast(tf.reduce_all(tf.greater_equal(margin, tf.abs(tf.subtract(y_pred_n, y_true_n))), axis=-1), tf.float32), axis=-1) * (1./GLOBAL_BATCH_SIZE) 
        return within_margin, closest_to_nearest 



@tf.function
def coord_dist(y_true, y_pred):
    y_pred = tf.cast(get_max_indices_argmax(y_pred), tf.float32)
    y_true = tf.cast(get_max_indices_argmax(y_true), tf.float32)
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.abs(y_true-y_pred)), axis=-1)), axis=-1) #pythagoras, mean batch, not entirely accurate for pixel images?
    return loss

@tf.function
def get_max_indices(logits):
    '''
    Returns coordinates for maximum values per axis 1 (for N,C,H,W input)
    '''
    coords = tf.squeeze(tf.where(tf.equal(logits, tf.reduce_max(logits))))
    coords = tf.cond(tf.greater(tf.rank(coords),tf.constant(1)),true_fn=lambda:tf.gather(coords,0),false_fn=lambda:coords)
    return tf.cast(coords, tf.float32)

# TODO use this for other functions than per_kp_stats_iter
@tf.function
def get_max_indices_argmax(logits): # TODO correctly make use of H,W dimensions
    flat_logits = tf.reshape(logits, tf.concat([tf.shape(logits)[:-2], [-1]], axis=0))
    max_val = tf.cast(tf.argmax(flat_logits, axis=-1), tf.int32)
    w = max_val // tf.shape(logits)[-1]
    h = max_val % tf.shape(logits)[-1]
    res =  tf.concat((w,h), axis=-1)
    return res

@tf.function
def per_kp_stats(y_true, y_pred, margin): # input_shape: (batch_size, n_landmarks, im_size, im_size)
    y_pred = tf.map_fn(lambda x: tf.map_fn(get_max_indices, x), y_pred)
    y_true = tf.map_fn(lambda y: tf.map_fn(get_max_indices, y), y_true)
    exp_y_pred = tf.expand_dims(y_pred, 1)
    exp_y_true = tf.expand_dims(y_true, 2)
    exp_closest = tf.argmin(tf.reduce_mean(tf.square(tf.abs(tf.subtract(exp_y_pred, exp_y_true))), axis=-1), axis=-1) # get index of closest landmark
    closest_to_nearest = tf.reduce_mean(tf.cast(tf.equal(tf.cast(exp_closest, dtype=tf.int32), tf.range(tf.shape(exp_closest)[-1], dtype=tf.int32)), dtype=tf.float32), axis=0) # check if that closest landmark is the target
    within_margin = tf.reduce_mean(tf.cast(tf.reduce_all(tf.greater_equal(margin, tf.abs(tf.subtract(y_true, y_pred))), axis=2), tf.float32), axis=0) # check distance of pred to target and if its within margin
    return within_margin, closest_to_nearest

@tf.function # (lm, batch, C, H, W)
def per_kp_stats_iter(y_true, y_pred, margin):
    y_pred_n = get_max_indices_argmax(y_pred)
    y_true_n = get_max_indices_argmax(y_true)
    exp_y_pred = tf.expand_dims(y_pred_n, 0)
    exp_y_true = tf.expand_dims(y_true_n, 1)
    closest = tf.square(tf.subtract(exp_y_pred, exp_y_true)) # using TF broadcast to create distance table
    closest_red = tf.argmin(tf.reduce_mean(closest, axis=-1), axis=1) # find min distance
    closest_to_nearest = tf.reduce_mean(tf.cast(tf.equal(tf.transpose(tf.cast(closest_red, tf.int32)), tf.range(tf.shape(closest_red)[0], dtype=tf.int32)), dtype=tf.float32), axis=0) 
    within_margin = tf.reduce_mean(tf.cast(tf.reduce_all(tf.greater_equal(margin, tf.abs(tf.subtract(y_pred_n, y_true_n))), axis=-1), tf.float32), axis=-1) # 
    return within_margin, closest_to_nearest 

@tf.function
def unstack_img_lab(img, kp_list):
    given_kp = tf.gather(img, tf.constant([0]), axis=0)    
    given_kp = tf.gather(img, tf.constant(range(1,len(kp_list))), axis=0)
    given_kp = tf.map_fn(get_max_indices, given_kp)
    image = tf.gather(img, tf.constant(0), axis=0)
    return image, given_kp

def store_results(img, label, model, kp_list, fn, path):
    given_kp = None
    filename = fn.numpy().decode('UTF-8')
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
    plt.savefig(path+'\\samples\\'+filename+'_pred.png')
    vis_points(img.numpy().squeeze(), lab_kp.numpy()[0], 5, given_kp)
    plt.savefig(path+'\\samples\\'+filename+'_gt.png')

def store_results_iter(img, label, model, fn, path, lm_count, n_landmarks, im_size, kp_margin, test_mode=True):
    filenames = [i.decode('UTF-8') for i in fn.numpy()]
    inp, lab = convert_input(img, label, n_landmarks, lm_count)
    if test_mode:
        pred = model.pred_test(inp)
    else:
        pred = model(inp)

    loss = tf.map_fn(lambda x: ssd_loss(x[0], x[1]), (lab, pred), dtype=tf.float32)

    lab = tf.expand_dims(tf.reshape(tf.transpose(lab, [0,2,1,3,4]), [-1, args.batch_size, im_size[0], im_size[1]]), axis=2)
    pred = tf.expand_dims(tf.reshape(tf.transpose(pred, [0,2,1,3,4]), [-1, args.batch_size, im_size[0], im_size[1]]), axis=2) # 40,2,1,256,256

    c_dist = coord_dist(lab, pred)
    within_margin, closest_to_gt = per_kp_stats_iter(lab, pred, kp_margin)

    pred_keypoints = tf.transpose(get_max_indices_argmax(pred), [1,0,2])
    lab_keypoints = tf.transpose(get_max_indices_argmax(lab), [1,0,2])
    
    img = img.numpy().squeeze() #np.sum(lab.numpy().squeeze(), axis=0)#
    if len(img.shape)<3: # for batcH_size = 1
        img = np.expand_dims(img,axis=0)
    pred_logits = np.sum(pred.numpy().squeeze(), axis=0)

    pred_keypoints = pred_keypoints.numpy()
    lab_keypoints = lab_keypoints.numpy()
    if not os.path.exists(path+'\\samples\\'):
        os.makedirs(path+'\\samples\\')
    for i in range(img.shape[0]):
        # vis_points(img[i], pred_keypoints[i], 3, None)
        vis_points(img[i], pred_keypoints[i], 3, None)
        
        plt.savefig(path+'\\samples\\'+filenames[i]+'_pred.png')
        vis_points(img[i], lab_keypoints[i], 3, None)
        plt.savefig(path+'\\samples\\'+filenames[i]+'_gt.png')
        plt.imshow(cv2.cvtColor(pred_logits[i], cv2.COLOR_GRAY2BGR))
        plt.savefig(path+'\\samples\\'+filenames[i]+'_pred_logits.png')
    return [loss], [c_dist], within_margin, closest_to_gt
    
@tf.function
def convert_input(img, lab, lm, lm_count):
    ep_lab_0 = tf.fill(tf.shape(lab[:,0:lm_count,:,:]), -1e-4)
    # ep_lab = tf.zeros_like(lab)[:,0:lm_count,:,:] # t-1 label (4,lm_count,256,256)
    img = tf.expand_dims(tf.repeat(img, lm//lm_count, axis=1), axis=2)
    ep_lab = tf.concat([ep_lab_0, lab[:,0:lm-lm_count,:,:]], axis=1)
    ep_lab = tf.stack(tf.split(ep_lab, lm//lm_count, axis=1), axis=1)
    inp = tf.concat([img, ep_lab], axis=2)
    lab = tf.split(lab, lm//lm_count, axis=1)
    lab = tf.stack(lab, axis=1)
    inp = tf.transpose(inp, [1,0,2,3,4])
    lab = tf.transpose(lab, [1,0,2,3,4])
    return inp, lab


def test_pipeline(path, num_filters, fmap_inc_factor, ds_factors, lm_count, im_size=None, train_pct=80, val_pct=10, test_pct=10, ntm_config=None, run_number=None, start_steps=0, kp_metric_margin=3):
    kp_margin = tf.constant(kp_metric_margin, dtype=tf.int32)
    dataset = data.Data_Loader(args.dataset, args.batch_size, train_pct=train_pct, val_pct=val_pct, test_pct=test_pct, n_aug_rounds=10)
    dataset(im_size=im_size)
    unet_model = unet.unet2d(num_filters, fmap_inc_factor, ds_factors, lm_count, seq_len=dataset.n_landmarks//lm_count, ntm_config=ntm_config, batch_size=args.batch_size)
    if start_steps > 0:
        if start_steps % args.checkpoint_interval != 0:
            start_steps = int(np.round(float(start_steps) / args.checkpoint_interval, 0) * args.checkpoint_interval)
        log_path, cp_path, cp_dir = load_dir(path, run_number, start_steps)
        unet_model.load_weights(cp_dir)
    else:
        log_path, cp_path = create_dir(path)
    train_writer = tf.summary.create_file_writer(log_path+"\\train\\")
    val_writer = tf.summary.create_file_writer(log_path+"\\val\\")

    lr_schedule = ExponentialCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=200,
            scale_mode="cycle",
            gamma=0.96,
            name="exp_cyclic_scheduler")

    lr_schedule_2 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.learning_rate, decay_steps=500, decay_rate=.9)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_2, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=.9)
    train = iter(dataset.train_data)
    val = iter(dataset.val_data)
    test = iter(dataset.test_data)
    for i in range(3):
        img, lab, fn = next(train)
        store_results_iter(img, lab, unet_model, fn, log_path, lm_count, dataset.n_landmarks, im_size, kp_margin)

def store_parameters(data_config, opti_config, unet_config, ntm_config, training_params, log_path):
    params = {"data_config":data_config,
              "opti_config":opti_config,
              "unet_config":unet_config,
              "ntm_config":ntm_config,
              "training_params":training_params
             }
    with open(os.path.join(log_path,'params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)
    fp.close()

# https://www.tensorflow.org/tutorials/distribute/custom_training see this
def load_parameters(log_path):
    with open(os.path.join(log_path,'params.json'), 'r') as fp:
        params = json.load(fp)
    return params["data_config"], params["opti_config"], params["unet_config"], params["ntm_config"], params["training_params"]

def train_step(inp, lab): #TODO
    with tf.GradientTape() as tape:
        pred = unet_model(inp)
        loss = compute_loss(lab, pred)
    grad = tape.gradient(loss, unet_model.trainable_weights)
    clipped_grad, _ = tf.clip_by_global_norm(grad, 10000.0)
    optimizer.apply_gradients(zip(clipped_grad, unet_model.trainable_weights))

    lab = tf.reshape(lab, [-1, data_config["batch_size"]//strategy.num_replicas_in_sync, 1, im_size[0], im_size[0]])
    pred = tf.reshape(pred, [-1, data_config["batch_size"]//strategy.num_replicas_in_sync, 1, im_size[0], im_size[0]])
    kp_loss = tf.map_fn(lambda y: dist_ssd_loss(y[0], y[1]), (lab, pred), dtype=tf.float32) 
    c_dist = tf.map_fn(lambda y: dist_coord_dist(y[0], y[1]), (lab, pred), dtype=tf.float32)
    return loss, kp_loss, c_dist 

def val_step(inp, lab): #TODO
    pred = unet_model()

def test_step(): # TODO
    pass

def distributed_train_step(inp,lab):
    per_replica_loss, pr_kp_loss, pr_c_dist = strategy.run(train_step, args=(inp,lab,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, pr_kp_loss, axis=0), strategy.reduce(tf.distribute.ReduceOp.SUM, pr_c_dist, axis=0)

def iterative_train_loop(path, data_config, opti_config, unet_config, ntm_config, training_params, run_number=None, start_steps=0):
    '''
    Try a curriculum kind of approach, where we iteratively learn a landmark and then the next, with the solution of the last as input.
    lm_count: how many landmarks at once
    '''    
    global unet_model
    if start_steps > 0:
        if start_steps % training_params["checkpoint_interval"] != 0:
            start_steps = int(np.round(float(start_steps) / training_params["checkpoint_interval"], 0) * training_params["checkpoint_interval"])
        log_path, cp_path, cp_dir = load_dir(path, run_number, start_steps)
        unet_model.load_weights(cp_dir)
        data_config, opti_config, unet_config, ntm_config, training_params = load_parameters(log_path) #this now always overwrites any given params, TODO
    else:
        log_path, cp_path = create_dir(path)
    train_writer = tf.summary.create_file_writer(log_path+"\\train\\")
    val_writer = tf.summary.create_file_writer(log_path+"\\val\\")
    
    if run_number is None:
        store_parameters(data_config, opti_config, unet_config, ntm_config, training_params, log_path)

    kp_margin = tf.constant(training_params["kp_metric_margin"], dtype=tf.int32)
    dataset = data.Data_Loader(name=data_config['dataset'],
                               batch_size=data_config["batch_size"],
                               train_pct=data_config["train_pct"],
                               val_pct=data_config["val_pct"],
                               test_pct=data_config["test_pct"],
                               n_aug_rounds=data_config["n_aug_rounds"],
                               repeat=data_config["repeat"],
                               prefetch=data_config["prefetch"])

    dataset(im_size=data_config["im_size"])
    im_size = data_config["im_size"] # to save space...
    with strategy.scope():
        unet_model = unet.unet2d(num_fmaps=unet_config["num_filters"],
                                 fmap_inc_factor=unet_config["fmap_inc_factor"],
                                 downsample_factors=unet_config["ds_factors"],
                                 num_landmarks=data_config["lm_count"],
                                 seq_len=dataset.n_landmarks//data_config["lm_count"],
                                 ntm_config=ntm_config,
                                 batch_size=data_config["batch_size"],
                                 im_size=data_config["im_size"]
                                 )

        lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opti_config["learning_rate"], decay_steps=opti_config["decay_steps"], decay_rate=opti_config["decay_rate"])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay, beta_1=opti_config["adam_beta_1"], beta_2=opti_config["adam_beta_2"], epsilon=opti_config["adam_epsilon"])

        train = iter(strategy.experimental_distribute_dataset(dataset.train_data))
        val = iter(strategy.experimental_distribute_dataset(dataset.val_data))
        test = iter(strategy.experimental_distribute_dataset(dataset.test_data))

        train_loss = []
        train_loss_lm = []
        val_loss = []
        val_loss_lm = []
        train_coord_dist_lm = []
        val_coord_dist_lm = []
        mrg_lm = []
        cgt_lm = []

        tf.print("Starting train loop...")
        start_time = time.time()
        for step in range(start_steps, training_params["num_training_iterations"]+1):
            img, lab, _ = next(train) 
            inp, lab = convert_input(img, lab, dataset.n_landmarks, data_config["lm_count"])
            loss, kp_loss, c_dist = distributed_train_step(inp, lab)
            # with tf.GradientTape() as tape:
            #     pred = unet_model(inp)
            #     loss = ssd_loss(lab, pred)
            #     grad = tape.gradient(loss, unet_model.trainable_weights)
            #     clipped_grad, _ = tf.clip_by_global_norm(grad, 10000.0)
            #     optimizer.apply_gradients(zip(clipped_grad, unet_model.trainable_weights))


            #     lab = tf.reshape(lab, [-1, data_config["batch_size"], 1, im_size[0], im_size[0]])
            #     pred = tf.reshape(pred, [-1, data_config["batch_size"], 1, im_size[0], im_size[0]])
            #     kp_loss = tf.map_fn(lambda y: ssd_loss(y[0], y[1]), (lab, pred), dtype=tf.float32) 
            #     c_dist = tf.map_fn(lambda y: coord_dist(y[0], y[1]), (lab, pred), dtype=tf.float32)

            train_loss.append(loss)
            train_loss_lm.append([kp_loss])
            train_coord_dist_lm.append([c_dist])
            
            if step % training_params["report_interval"] == 0:
                for _ in range(training_params["validation_steps"]):
                    val_coord_dist = []
                    mrg = []
                    cgt = []
                    img_v, lab_v, _ = next(val)
                    inp_v, lab_v = convert_input(img_v, lab_v, dataset.n_landmarks, data_config["lm_count"])
                    
                    pred_v = unet_model(inp_v) # (lm, batch, C, H, W)
                    val_loss.append(ssd_loss(lab_v, pred_v))

                    lab_v = tf.reshape(lab_v, [-1, args.batch_size, 1, im_size[0], im_size[0]])
                    pred_v = tf.reshape(pred_v, [-1, args.batch_size, 1, im_size[0], im_size[0]])
                    
                    kp_val_loss = tf.map_fn(lambda x: ssd_loss(x[0], x[1]), (lab_v, pred_v), dtype=tf.float32)
                    val_c_dist = tf.map_fn(lambda y: coord_dist(y[0], y[1]), (lab_v, pred_v), dtype=tf.float32)
                    within_margin, closest_to_gt = per_kp_stats_iter(lab_v, pred_v, kp_margin)
                    
                    mrg.append(within_margin) #????
                    cgt.append(closest_to_gt)
                    val_loss_lm.append([kp_val_loss])
                    val_coord_dist_lm.append([val_c_dist])
                    mrg_lm.append(mrg)
                    cgt_lm.append(cgt)
            
                t_mean = tf.reduce_mean(train_loss)
                v_mean = tf.reduce_mean(val_loss)
                tl_mean = tf.squeeze(tf.reduce_mean(train_loss_lm, axis=0))
                tcd_mean = tf.squeeze(tf.reduce_mean(train_coord_dist_lm, axis=0))
                vl_mean = tf.squeeze(tf.reduce_mean(val_loss_lm, axis=0))
                vcd_mean = tf.squeeze(tf.reduce_mean(val_coord_dist_lm, axis=0))
                mrg_mean = tf.squeeze(tf.reduce_mean(mrg_lm, axis=0))
                cgt_mean = tf.squeeze(tf.reduce_mean(cgt_lm, axis=0))
                
                elapsed_time = int(time.time() - start_time)
                tf.print("Iteration", step , "(Elapsed: ", elapsed_time, "s):")
                tf.print("mean train loss since last update:", t_mean, summarize=-1)
                with open(os.path.join(log_path, 'train_loss.txt'), 'ab') as tltxt:
                    np.savetxt(tltxt, [np.array(t_mean)], fmt='%.3f', delimiter=",")

                tf.print("train loss per kp (ssd): ", tl_mean, summarize=-1)
                with open(os.path.join(log_path, 'train_loss_kp.txt'), 'ab') as tlkptxt:
                    np.savetxt(tlkptxt, [np.array(tl_mean)], fmt='%.3f', delimiter=",")
                tf.print("train coordinate distance: ", tcd_mean, summarize=-1)
                with open(os.path.join(log_path, 'train_coordd.txt'), 'ab') as tcdtxt:
                    np.savetxt(tcdtxt, [np.array(tcd_mean)], fmt='%.3f', delimiter=",")
                
                tf.print("mean val loss since last update:", v_mean, summarize=-1)
                with open(os.path.join(log_path, 'val_loss.txt'), 'ab') as vltxt:
                    np.savetxt(vltxt, [np.array(v_mean)], fmt='%.3f', delimiter=",")

                tf.print("validation loss per kp (ssd): ", vl_mean, summarize=-1)
                with open(os.path.join(log_path, 'val_loss_kp.txt'), 'ab') as vlkptxt:
                    np.savetxt(vlkptxt, [np.array(vl_mean)], fmt='%.3f', delimiter=",")
                tf.print("validation coordinate distance: ", vcd_mean, summarize=-1)
                with open(os.path.join(log_path, 'val_coordd.txt'), 'ab') as vcdtxt:
                    np.savetxt(vcdtxt, [np.array(vcd_mean)], fmt='%.3f', delimiter=",")

                tf.print("% within margin: ", mrg_mean, summarize=-1)
                with open(os.path.join(log_path, 'vaL_within_margin.txt'), 'ab') as mrgtxt:
                    np.savetxt(mrgtxt, [np.array(mrg_mean)], fmt='%3.3f', delimiter=",")
                mrgtxt.close()
                tf.print("% closest to gt", cgt_mean, summarize=-1)
                with open(os.path.join(log_path, 'val_closest_gt.txt'), 'ab') as cgttxt:
                    np.savetxt(cgttxt, [np.array(cgt_mean)], fmt='%3.3f', delimiter=",")
                cgttxt.close()

                train_loss =[]
                train_loss_lm = []
                val_loss = []
                val_loss_lm = []
                train_coord_dist_lm = []
                val_coord_dist_lm = []
                mrg_lm = [] 
                cgt_lm = []
            
        
            
            if step % training_params["checkpoint_interval"] == 0:
                unet_model.save_weights(cp_path.format(step=step))
                print("saved cp-{:04d}".format(step))
        
        test_loss = []
        test_c_dist = []
        test_mrg = []
        test_cgt = []
        for _ in range(training_params["num_test_samples"]):
            img_t, lab_t, fn = next(test) # img: 1,1,64,64 , lab: 1,40,64,64
            loss_t, c_dist_t, mrg_t, cgt_t = store_results_iter(img_t, lab_t, unet_model, fn, log_path, data_config["lm_count"], dataset.n_landmarks, im_size, kp_margin)
            test_loss.append(loss_t)
            test_c_dist.append(c_dist_t)
            test_mrg.append(mrg_t)
            test_cgt.append(cgt_t)
        test_res = [np.array(tf.squeeze(tf.reduce_mean(i, axis=0))) for i in [test_loss, test_c_dist, test_mrg, test_cgt]]
        with open(os.path.join(log_path, 'test_res.txt'), 'ab') as testtxt:
            for i in test_res:
                np.savetxt(testtxt, [i], fmt='%3.3f', delimiter=",")
        testtxt.close()
    
        
# TODO rewrite this to match the iterative training -> maybe we first do the distribute because that changes a lot again
def train_unet_custom(path, num_filters, fmap_inc_factor, ds_factors, im_size=None, train_pct=80, val_pct=10, test_pct=10, kp_list_in=None, ntm_config=None, run_number=None, start_steps=0, kp_metric_margin=3):
    assert im_size is not None, "Please provide an image size to which to rescale the input to"
    if kp_list_in == [0]:
        kp_list_in = None
    kp_margin = tf.constant(kp_metric_margin, dtype=tf.float32)
    dataset = data.Data_Loader(args.dataset, args.batch_size, train_pct=train_pct, val_pct=val_pct, test_pct=test_pct)
    dataset(im_size=im_size, keypoints=kp_list_in)
    len_kp = (len(kp_list_in)-1) if kp_list_in is not None else 0
    unet_model = unet.unet2d(num_filters, fmap_inc_factor, ds_factors, dataset.n_landmarks-len_kp, ntm_config=ntm_config, batch_size=args.batch_size)
    if start_steps > 0:
        if start_steps % args.checkpoint_interval != 0:
            start_steps = int(np.round(float(start_steps) / args.checkpoint_interval, 0) * args.checkpoint_interval)
        log_path, cp_path, cp_dir = load_dir(path, run_number, start_steps)
        unet_model.load_weights(cp_dir)
    else:
        log_path, cp_path = create_dir(path)
    train_writer = tf.summary.create_file_writer(log_path+"\\train\\")
    val_writer = tf.summary.create_file_writer(log_path+"\\val\\")
    
    @tf.function
    def predict(img):
        return unet_model(img)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    train = iter(dataset.train_data)
    val = iter(dataset.val_data)
    train_loss = []
    val_loss = []
    train_coord_dist = []
    val_coord_dist = []
    mrg = []
    cgt = []
    tf.print("Starting train loop...")
    start_time = time.time()
    for step in range(start_steps, args.num_training_iterations+1):
        with tf.GradientTape() as tape:
            img, lab, _ = next(train) 
            pred = predict(img)
            loss = ssd_loss(lab, pred)
        grad = tape.gradient(loss, unet_model.trainable_weights)
        optimizer.apply_gradients(zip(grad, unet_model.trainable_weights)) # tf.clip_by_global_norm
        train_loss.append(loss)
        train_coord_dist.append(coord_dist(lab, pred))
        
        if step % args.report_interval == 0:
            for _ in range(args.validation_steps):
                img_v, lab_v, _ = next(val)
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
                tf.summary.scalar("ssd_loss", vl_mean, step=step)
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
            print("saved cp-{:04d}".format(step))
    test = iter(dataset.test_data)
    for _ in range(args.num_test_samples):
        img, lab, fn = next(test)
        store_results(img, lab, unet_model, kp_list_in, fn, log_path)

# TODO rework this, so its consistent.
def create_dir(path):
    previous_runs = os.listdir(path)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    logdir = 'run_%02d' % run_number
    l_dir = os.path.join(path, logdir)
    cp_dir = l_dir +'\\cp\\cp-{step:04d}'
    return l_dir, cp_dir

def load_dir(path, run_number, step):
    logdir = 'run_%02d' % run_number
    l_dir = os.path.join(path, logdir)
    cp_pth = l_dir+'\\cp\\cp-{step:04d}'
    cp_dir = l_dir+'\\cp\\cp-{step:04d}'.format(step=step)
    #cp_dir = os.path.dirname(cp_pth)
    return l_dir, cp_pth, cp_dir


if __name__ == "__main__":
    '''
    config explanation:
    Config is stored and loaded as .json into a python dict

    #### data_config
    dataset: 'droso', str, (TODO maybe another dataset at some point), choose which dataset to run experiment o n
    batch_size: 2, int8, batch size
    im_size: [256,256], int tuple, resize images to this size
    lm_count: 5, int8, how many landmarks to put into iterative learning task at each iteration
    kp_list_in: [0,1,3,5], list of int8, which landmarks to put into input for non-iterative learning task (i believe it starts at 1, 0=image itself, None works for image itself as well)
    train_pct, 10  \\
    val_pct,   10   | --- int from (0,100], sum cant be > 100, how much of the dataset is train, test, validation set.
    test_pct,  10   /
    repeat, true, bool, whether to repeat the dataset infinitely
    prefetch, true, bool, whether to prefetch samples from the tf.data dataset
    n_aug_rounds, 10, int8, How many altered versions of the train dataset to append, i.e. 10 = 10 transformed versions of the same dataset are appended to the original (also transformed) dataset

    #### opti_config
    learning_rate, 1e-3, float, optimizer (adam) learning rate
    adam_beta_1, 0.9, float The exponential decay rate for the 1st moment estimates # TODO check whether this decays the learning rate
    adam_beta_2,  0.999, float, The exponential decay rate for the 2nd moment estimates
    adam_epsilon, 1e-7, float, A small constant for numerical stability 
    decay_rate: .9, float, rate at which to decay # see: initial_learning_rate * decay_rate ^ (step / decay_steps)
    decay_steps: 500, int, exponent to accelerate decay after x steps # ^ 

    #### unet_config
    num_filters, 16, int, number of filters on first layer of Unet
    fmap_inc_factor, 2, int, multiplier on how much to increase number of filters at each level
    ds_factors, [[2,2],[2,2],[2,2],[2,2],[2,2]], list of int tuples, defines depth of Unet, also defines how much is downsampled at each depth step

    #### ntm_config
    !! contains multiple configs, depending on how many ntm layers you want to include in the unet i.e. a ntm layer  !!
    !! on first unet level is provided by feeding a dict with name = unet level {"0":{"enc_dec_param":{...}}}        !!
    !! ntm_config further contains two sub-dictionaries: enc_dec_param, ntm_param                                    !!
    
        ###### end_dec_param, this is finnicky and needs to match the unet position
        num_filters, 16, int,  number of conv filters at each layer
        kernel_size, 3, int, conv_layer kernel size
        pool_size, [4,4], list of int, defines number of encoder decoder layers and how much is pooled
    
        ###### ntm_param, set to None if no NTM, just encoder-decoder architecture
        controller_units, 256, int, number of units of controller dense layer, controller defines what and how to write and read
        memory_size, 64, int, length of a single memory entry
        memory_vector_dim, 256, int, number of memory entries
        output_dim, 256, int, output dimensions of NTM, has to be square-rootable as it is reshaped to square format and then upsampled needs to match the Unet layer output
        read_head_num, 3, int, number of simultaneous read heads
        write_head_num, 3 int, number of simultaneous write heads
    
    #### training_params
    num_training_iterations, 10000, int, number of training steps, for iterative approach this also means, number of samples put in (does not get reduced by multiple iterations over the same image)
    validation_steps, 5, int, number of validations iterations over which metrics are averaged and reported
    report_interval, 50, int, number of steps after which a validation step happens
    kp_metric_margin, 3, int, pixel margin at which a kp is counted as close enough
    checkpoint_interval, 500, int, number of steps at which a model checkpoints happens, -> can only be used to reuse model for predictions, not to resume training, because no way to ensure data consistency currently
    num_test_samples, 5, int, number of samples (samples*batch_size) to test the model on, these samples are also printed and stored
    '''
    global GLOBAL_BATCH_SIZE
    PATH = 'C:\\Users\\Elias\\Desktop\\MA_logs'
    

    data_config = {"dataset":'droso',
                   "batch_size": 2,
                   "im_size": [256,256],
                   "lm_count": 5,
                   "kp_list_in": None,
                   "train_pct":10,
                   "val_pct":10,
                   "test_pct":10,
                   "repeat":True,
                   "prefetch":True,
                   "n_aug_rounds":10
                   }
    
    opti_config = {"learning_rate": 1e-3,
                   "adam_beta_1": 0.9,
                   "adam_beta_2": 0.999,
                   "adam_epsilon": 1e-7,
                   "decay_rate": .9, 
                   "decay_steps": 500
                   }
    
    unet_config = {"num_filters": 16,
                   "fmap_inc_factor": 2,
                   "ds_factors":[[2,2],[2,2],[2,2],[2,2],[2,2]]}


    ntm_config = {"0":{"enc_dec_param":{"num_filters":16,
                                               "kernel_size":3,
                                               "pool_size":[4,4]},
                              "ntm_param":{"controller_units":256,
                                           "memory_size":64,
                                           "memory_vector_dim":256,
                                           "output_dim":256,
                                           "read_head_num":3,
                                           "write_head_num":3}}
                        }
    
    training_params = {"num_training_iterations": 1,
                       "validation_steps": 5,
                       "report_interval": 50,
                       "kp_metric_margin": 3,
                       "checkpoint_interval": 500,
                       "num_test_samples": 5
                       }

    data_config["batch_size"] = data_config["batch_size"]*strategy.num_replicas_in_sync
    # List of experiments:
    # BIG TODO: check how long a training takes, adjust list of experiments accordingly
    # 1. Baseline (Unet):
    # 	- Train with full train test split (85/5/10) with all metrics

    # train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256, 256], train_pct=85, val_pct=5, test_pct=10)

    # 	- Train with 80/70/60/50/40/30/20/10/5 with all metrics?
    # for i in [80,70,60,50,40,30,20,10,5]:
    #     train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256, 256], train_pct=i, val_pct=5, test_pct=10)

    # 2. Unet with NTM: # NOTE: given that there is not best baseline for this whole thing, we simply define one and then compare against that
    # 	- Train with full train test split (80/20)

    # train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]],im_size=[256, 256], train_pct=85, val_pct=5, test_pct=10, ntm_config=standard_ntm_conf)

    # 		- compare with slightly larger Unet
    # train_unet_custom(PATH, num_filters=72, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256,256], train_pct=85, val_pct=5, test_pct=10) # TODO evaluate model size? Needs to be same number of parameters as ntm net
    # 		- compare with Unet with Encoder Decoder
    # train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256,256], train_pct=85, val_pct=5, test_pct=10, ntm_config=standard_ed_conf)
    # 	- NTM at different positions (5? positions, then multiple ones?) (80%/5%)
    # for i in [85, 5]:
    #     for j in CONF_POS_LIST:
    #         train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256,256], train_pct=i, val_pct=5, test_pct=10, ntm_conf=j)
    # - Different memory sizes (a,b,c,d,e,f)
    #     for k in CONF_MEM_LIST:
    #         train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256,256], train_pct=i, val_pct=5, test_pct=10, ntm_conf=k)
    # TODO pick best POS

    # 		- Train with 80/70/60/50/40/30/20/10/5 
    # for i in [80,70,60,50,40,30,20,10,5]:
    #     train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256,256], train_pct=i, val_pct=5, test_pct=10, ntm_conf=standard_ntm_conf)
    

    # 3. Give landmarks (5%) (unet, ntm)
        
    # 		- random
    # for n in [2,5]:            
    #     for _ in range(3):
    #         rand_kp = random.sample(range(1,40), k=n)
    #         train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256, 256], train_pct=5, val_pct=5, test_pct=10, kp_list_in=[0]+rand_kp)
    #         train_unet_custom(PATH, num_filters=64, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], im_size=[256, 256], train_pct=5, val_pct=5, test_pct=10, kp_list_in=[0]+rand_kp, ntm_config=standard_ntm_conf)

    # TODO: maybe one attempt where we select the landmarks?
    

    # 4. Iterative learning approach: (5%) (unet, ntm)
    # 	- Iterative feed with solution in t+1
    iterative_train_loop(PATH, data_config=data_config, opti_config=opti_config, unet_config=unet_config, ntm_config=ntm_config, training_params=training_params)    # 	- batched, not batched
    # test_pipeline(PATH, num_filters=16, fmap_inc_factor=2, ds_factors=[[2,2],[2,2],[2,2],[2,2],[2,2]], lm_count=5, im_size=[256, 256], train_pct=10, val_pct=10, test_pct=10, ntm_config=standard_ntm_conf)    # 	- batched, not batched
	


# kp_list: 0 is image, remaining numpers are keypoints. If you dont want to include keypoints in input, set to None