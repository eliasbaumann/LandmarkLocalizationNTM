import os
import argparse

import json
import time
import matplotlib.pyplot as plt
import cv2

import numpy as np
import tensorflow as tf

import unet
import data

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, default=None, help='Select a directory in which to search for config.json to execute')
args = parser.parse_args()

# tf.config.experimental_run_functions_eagerly(True)

class Train(object):
    def __init__(self, model, strategy, data_config, opti_config, training_params, data_landmarks, dataset_test_size, log_path, cp_path):
        self.model = model
        self.strategy = strategy
        lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opti_config["learning_rate"],
                                                                  decay_steps=opti_config["decay_steps"],
                                                                  decay_rate=opti_config["decay_rate"])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay,
                                                  beta_1=opti_config["adam_beta_1"],
                                                  beta_2=opti_config["adam_beta_2"],
                                                  epsilon=opti_config["adam_epsilon"])
        
        self.im_size = data_config["im_size"]
        self.global_batch_size = data_config["batch_size"]*strategy.num_replicas_in_sync
        self.batch_div = 1./self.global_batch_size
        self.kp_margin = tf.constant(training_params["kp_metric_margin"], dtype=tf.int32)
        self.data_config = data_config
        self.opti_config = opti_config
        self.training_params = training_params
        self.data_landmarks = data_landmarks
        self.dataset_test_size = dataset_test_size
        self.log_path = log_path
        self.cp_path = cp_path
        self.iter = True if training_params["mode"]=="iter" else False
        self.print_samples = True if training_params["store_samples"]==1 else False

    def dist_ssd_loss(self, gt_labels, logits):
        return tf.reduce_sum(tf.square(gt_labels-logits), axis=None)
    
    def compute_loss(self, gt_labels, logits):
        per_example_loss = self.dist_ssd_loss(gt_labels, logits)
        return per_example_loss * self.batch_div

    def dist_coord_dist(self, gt_labels, logits):
        logits = tf.cast(self.get_max_indices_argmax(logits), tf.float32)
        gt_labels = tf.cast(self.get_max_indices_argmax(gt_labels), tf.float32)
        loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(gt_labels-logits), axis=-1)), axis=-1) #pythagoras, mean batch, not entirely accurate for pixel images?
        return loss

    def dist_per_kp_stats_iter(self, gt_labels, logits, margin):
        y_pred_n = self.get_max_indices_argmax(logits)
        y_true_n = self.get_max_indices_argmax(gt_labels)
        exp_y_pred = tf.expand_dims(y_pred_n, 0)
        exp_y_true = tf.expand_dims(y_true_n, 1)
        closest = tf.square(tf.subtract(exp_y_pred, exp_y_true)) # using TF broadcast to create distance table
        closest_red = tf.argmin(tf.reduce_mean(closest, axis=-1), axis=1) # find min distance
        closest_to_nearest = tf.reduce_sum(tf.cast(tf.equal(tf.transpose(tf.cast(closest_red, tf.int32)), tf.range(tf.shape(closest_red)[0], dtype=tf.int32)), dtype=tf.float32), axis=0) 
        within_margin = tf.reduce_sum(tf.cast(tf.reduce_all(tf.greater_equal(margin, tf.abs(tf.subtract(y_pred_n, y_true_n))), axis=-1), tf.float32), axis=-1) 
        return within_margin*self.batch_div, closest_to_nearest*self.batch_div
    
    def get_max_indices_argmax(self, logits): # TODO correctly make use of H,W dimensions
        flat_logits = tf.reshape(logits, tf.concat([tf.shape(logits)[:-2], [-1]], axis=0))
        max_val = tf.cast(tf.argmax(flat_logits, axis=-1), tf.int32)
        w = max_val // tf.shape(logits)[-1]
        h = max_val % tf.shape(logits)[-1]
        res =  tf.concat((w,h), axis=-1)
        return res

    def convert_input(self, img, lab, lm, lm_count, iter):
        if not self.iter:
            inp = tf.expand_dims(img, axis=0)
            lab = tf.expand_dims(lab, axis=0)
            return inp, lab
        ep_lab_0 = tf.fill(tf.shape(lab[:,0:lm_count,:,:]), -1e-4)
        # ep_lab = tf.zeros_like(lab)[:,0:lm_count,:,:] # t-1 label (4,lm_count,256,256)
        img = tf.expand_dims(tf.repeat(img, lm//lm_count, axis=1), axis=2)
        ep_lab = tf.concat([ep_lab_0, lab[:,0:lm-lm_count,:,:]], axis=1)
        ep_lab = tf.stack(tf.split(ep_lab, lm//lm_count, axis=1), axis=1)
        inp = tf.concat([img, ep_lab], axis=2)
        lab = tf.split(lab, lm//lm_count, axis=1)
        lab = tf.stack(lab, axis=1)
        inp = tf.transpose(inp, [1,0,2,3,4]) # Seq, Batch, C, H, W
        lab = tf.transpose(lab, [1,0,2,3,4])
        return inp, lab

    def kp_loss_c_dist(self, lab, pred):
        lab = tf.expand_dims(tf.reshape(tf.transpose(lab, [0,2,1,3,4]), [-1, self.data_config["batch_size"], self.im_size[0], self.im_size[0]]), axis=2)
        pred = tf.expand_dims(tf.reshape(tf.transpose(pred, [0,2,1,3,4]), [-1, self.data_config["batch_size"], self.im_size[0], self.im_size[0]]), axis=2)
        kp_loss = tf.map_fn(lambda y: self.dist_ssd_loss(y[0], y[1]), (lab, pred), dtype=tf.float32)*self.batch_div
        c_dist = tf.map_fn(lambda y: self.dist_coord_dist(y[0], y[1]), (lab, pred), dtype=tf.float32)*self.batch_div
        return lab, pred, kp_loss, c_dist

    def train_step(self, inp, lab): #TODO
        inp, lab = self.convert_input(inp, lab, self.data_landmarks, self.data_config["lm_count"], self.iter)
        with tf.GradientTape() as tape:
            pred, _, _ = self.model(inp)
            loss = self.compute_loss(lab, pred)
        grad = tape.gradient(loss, self.model.trainable_weights)
        clipped_grad, _ = tf.clip_by_global_norm(grad, 10000.0)
        self.optimizer.apply_gradients(zip(clipped_grad, self.model.trainable_weights))
        _, _, kp_loss, c_dist = self.kp_loss_c_dist(lab, pred)
        return loss, kp_loss, c_dist

    def val_step(self, inp, lab):
        inp, lab = self.convert_input(inp, lab, self.data_landmarks, self.data_config["lm_count"], self.iter)
        pred, _, _ = self.model(inp, training=False)
        loss = self.compute_loss(lab, pred)
        lab, pred, kp_loss, c_dist = self.kp_loss_c_dist(lab, pred)
        within_margin, closest_to_gt = self.dist_per_kp_stats_iter(lab, pred, self.kp_margin)
        return  loss, kp_loss, c_dist, within_margin, closest_to_gt
    
    def test_step(self, inp, lab, fn):
        inp, lab = self.convert_input(inp, lab, self.data_landmarks, self.data_config["lm_count"], self.iter)
        img = inp[0,:,:1,:,:]
        tf.print(fn)
        print(fn)
        
        if not (self.data_config["kp_list_in"] is None or self.data_config["kp_list_in"] == [0]):
            given_kp = tf.expand_dims(tf.reshape(tf.transpose(tf.expand_dims(inp[0,:,1:,:,:],axis=0), [0,2,1,3,4]), [-1, self.data_config["batch_size"], self.im_size[0], self.im_size[0]]), axis=2)
        else:
            given_kp = tf.constant(0.)
        pred, states, attn_maps = self.model.pred_test(inp, training=False) if self.iter else self.model(inp, training=False)
        loss = self.compute_loss(lab, pred)
        lab, pred, kp_loss, c_dist = self.kp_loss_c_dist(lab, pred)
        within_margin, closest_to_gt = self.dist_per_kp_stats_iter(lab, pred, self.kp_margin)
        if self.print_samples:
            tf.py_function(func=self.store_samples, inp=[img, pred, lab, fn, given_kp, states, attn_maps], Tout=[])
        # self.store_samples(img, pred, lab, fn, given_kp, states, attn_maps)
        return loss, kp_loss, c_dist, within_margin, closest_to_gt
    
    def min_max_scale(self, inp):
        return (inp-inp.min())/(inp.max()-inp.min())

    def store_samples(self, img, pred, lab, filenames, given_kp = None, states=None, attn_maps=None):
        filenames = [i.decode('UTF-8') for i in filenames.numpy()]
        pred_keypoints = tf.transpose(self.get_max_indices_argmax(pred), [1,0,2])
        lab_keypoints = tf.transpose(self.get_max_indices_argmax(lab), [1,0,2])
        
        img = img.numpy().squeeze() #np.sum(lab.numpy().squeeze(), axis=0)#
        pred = self.min_max_scale(pred.numpy().squeeze())
        
        if len(img.shape)<3: # for batcH_size = 1
            img = np.expand_dims(img, axis=0)
            pred = np.expand_dims(pred, axis=0)
        pred_logits = np.max(pred, axis=1)
        # lab_lg_comp = np.max((lab.numpy().squeeze()+1)/2., axis=0)

        pred_keypoints = pred_keypoints.numpy()
        lab_keypoints = lab_keypoints.numpy()
        if not os.path.exists(self.log_path+'/samples/'):
            os.makedirs(self.log_path+'/samples/')
        if given_kp.numpy()!=0.:
            given_kp = tf.transpose(self.get_max_indices_argmax(given_kp), [1,0,2]).numpy()
        else:
            given_kp = np.repeat(None, img.shape[0])
        for i in range(img.shape[0]): # per batch iteration                
            vis_points(img[i], pred_keypoints[i], 3, given_kp=given_kp[i])
            plt.savefig(self.log_path+'/samples/'+filenames[i]+'_pred.png')
            vis_points(img[i], lab_keypoints[i], 3, given_kp=given_kp[i])
            plt.savefig(self.log_path+'/samples/'+filenames[i]+'_gt.png')
            plt.imshow(cv2.cvtColor(pred_logits[i], cv2.COLOR_GRAY2BGR))
            plt.savefig(self.log_path+'/samples/'+filenames[i]+'_pred_logits.png')
            # plt.imshow(cv2.cvtColor(lab_lg_comp[i], cv2.COLOR_GRAY2BGR))
            # plt.savefig(self.log_path+'/samples/'+filenames[i]+'_gt_logits.png')
            self.store_mem(states, filenames[i], i)
            self.store_attn(attn_maps, filenames[i], i)

    def store_mem(self, states, fn, batch_no):
        if self.model.ntm_config is None: 
            return
        keys = list(map(int, self.model.ntm_config.keys()))
        os.makedirs(self.log_path+'/samples/'+fn)
        count = 0
        for state in states:
            for key in keys:
                if self.model.ntm_config[str(key)]["ntm_param"] is None:
                    break
                pos = self.model.ntm_config[str(key)]['enc_dec_param']['pos']
                if pos == "b":
                    M_l = state[key*2]['M'][batch_no].numpy()
                    M_l = (M_l+1)/2.
                    plt.imshow(cv2.cvtColor(M_l, cv2.COLOR_GRAY2BGR))
                    plt.savefig(self.log_path+'/samples/'+fn+'/'+str(key)+'_l_'+str(count)+'mem.png')
                    M_r = state[key*2+1]['M'][batch_no].numpy()
                    M_r = (M_r+1)/2.
                    plt.imshow(cv2.cvtColor(M_r, cv2.COLOR_GRAY2BGR))
                    plt.savefig(self.log_path+'/samples/'+fn+'/'+str(key)+'_r_'+str(count)+'mem.png')
                else:
                    key = key*2 if pos=="l" else key*2+1
                    M = state[key]['M'][batch_no].numpy()
                    M = (M+1)/2.
                    plt.imshow(cv2.cvtColor(M, cv2.COLOR_GRAY2BGR))
                    plt.savefig(self.log_path+'/samples/'+fn+'/'+str(key)+'_'+pos+'_'+str(count)+'mem.png')
            count += 1
    
    def store_attn(self, attn, fn, batch_no):
        if self.model.attn_config is None:
            return
        keys = list(map(int, self.model.attn_config.keys()))
        os.makedirs(self.log_path+'/samples/'+fn)
        count = 0
        for attn_map in attn:
            for key in keys:
                M = tf.squeeze(attn_map[key][batch_no]).numpy()
                plt.imshow(cv2.cvtColor(M, cv2.COLOR_GRAY2BGR))
                plt.savefig(self.log_path+'/samples/'+fn+'/'+str(key)+'_'+str(count)+'attn.png')
            count +=1


    @tf.function
    def distributed_train_step(self, inp, lab):
        per_replica_loss, pr_kp_loss, pr_c_dist = self.strategy.experimental_run_v2(self.train_step, args=(inp, lab,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None), self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_kp_loss, axis=None), self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_c_dist, axis=None)

    @tf.function
    def distributed_val_step(self, inp, lab):
        pr_loss, pr_kp_loss, pr_c_dist, pr_wm, pr_ctgt = self.strategy.experimental_run_v2(self.val_step, args=(inp, lab,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_loss, axis=None),self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_kp_loss, axis=None),self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_c_dist, axis=None),self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_wm, axis=None),self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_ctgt, axis=None)
    
    @tf.function
    def distributed_test_step(self, inp, lab, fn):
        pr_loss, pr_kp_loss, pr_c_dist, pr_wm, pr_ctgt = self.strategy.experimental_run_v2(self.test_step, args=(inp, lab, fn, ))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_loss, axis=None),self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_kp_loss, axis=None),self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_c_dist, axis=None),self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_wm, axis=None),self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_ctgt, axis=None)

    def val_store(self, step, elapsed_time, t, tl, tcd, v, vl, vcd, mrg, cgt):
        tf.print("Iteration", step , "(Elapsed: ", elapsed_time, "s):")
        tf.print("mean train loss since last validation:", t[0], "std:", t[1], summarize=-1)
        with open(os.path.join(self.log_path, 'train_loss.txt'), 'ab') as tltxt:
            np.savetxt(tltxt, [np.array(t[0])], fmt='%.3f', delimiter=",")
            np.savetxt(tltxt, [np.array(t[1])], fmt='%.3f', delimiter=",")

        #tf.print("train loss per kp (ssd): ", tl[0], summarize=-1)
        with open(os.path.join(self.log_path, 'train_loss_kp.txt'), 'ab') as tlkptxt:
            np.savetxt(tlkptxt, [np.array(tl[0])], fmt='%.3f', delimiter=",")
            np.savetxt(tlkptxt, [np.array(tl[1])], fmt='%.3f', delimiter=",")
        #tf.print("train coordinate distance: ", tcd[0], summarize=-1)
        with open(os.path.join(self.log_path, 'train_coordd.txt'), 'ab') as tcdtxt:
            np.savetxt(tcdtxt, [np.array(tcd[0])], fmt='%.3f', delimiter=",")
            np.savetxt(tcdtxt, [np.array(tcd[1])], fmt='%.3f', delimiter=",")
        
        tf.print("mean validation loss:", v[0], "std:", v[1], summarize=-1)
        with open(os.path.join(self.log_path, 'val_loss.txt'), 'ab') as vltxt:
            np.savetxt(vltxt, [np.array(v[0])], fmt='%.3f', delimiter=",")
            np.savetxt(vltxt, [np.array(v[1])], fmt='%.3f', delimiter=",")

        #tf.print("validation loss per kp (ssd): ", vl[0], summarize=-1)
        with open(os.path.join(self.log_path, 'val_loss_kp.txt'), 'ab') as vlkptxt:
            np.savetxt(vlkptxt, [np.array(vl[0])], fmt='%.3f', delimiter=",")
            np.savetxt(vlkptxt, [np.array(vl[1])], fmt='%.3f', delimiter=",")
        #tf.print("validation coordinate distance: ", vcd[0], summarize=-1)
        with open(os.path.join(self.log_path, 'val_coordd.txt'), 'ab') as vcdtxt:
            np.savetxt(vcdtxt, [np.array(vcd[0])], fmt='%.3f', delimiter=",")
            np.savetxt(vcdtxt, [np.array(vcd[1])], fmt='%.3f', delimiter=",")

        #tf.print("% within margin: ", mrg[0], summarize=-1)
        with open(os.path.join(self.log_path, 'vaL_within_margin.txt'), 'ab') as mrgtxt:
            np.savetxt(mrgtxt, [np.array(mrg[0])], fmt='%3.3f', delimiter=",")
            np.savetxt(mrgtxt, [np.array(mrg[1])], fmt='%3.3f', delimiter=",")

        #tf.print("% closest to gt", cgt[0], summarize=-1)
        with open(os.path.join(self.log_path, 'val_closest_gt.txt'), 'ab') as cgttxt:
            np.savetxt(cgttxt, [np.array(cgt[0])], fmt='%3.3f', delimiter=",")
            np.savetxt(cgttxt, [np.array(cgt[1])], fmt='%3.3f', delimiter=",")

    def iter_loop(self, train, val, test, start_steps):
        train = iter(train)
        val = iter(val)
        test = iter(test)

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
        for step in range(start_steps, self.training_params["num_training_iterations"]+1):
            img, lab, _ = next(train)
            loss, kp_loss, c_dist = self.distributed_train_step(img, lab)
            
            train_loss.append(loss)
            train_loss_lm.append([kp_loss])
            train_coord_dist_lm.append([c_dist])

            if step % training_params["report_interval"] == 0:
                for _ in range(training_params["validation_steps"]):
                    img_v, lab_v, _ = next(val)
                    v_loss, v_kp_loss, v_c_dist, v_within_margin, v_closest_to_gt = self.distributed_val_step(img_v, lab_v)

                    mrg_lm.append(v_within_margin)
                    cgt_lm.append(v_closest_to_gt)
                    val_loss.append(v_loss)
                    val_loss_lm.append([v_kp_loss])
                    val_coord_dist_lm.append([v_c_dist])

                t_mean = tf.reduce_mean(train_loss)
                t_std = tf.math.reduce_std(train_loss)
                tl_mean = tf.squeeze(tf.reduce_mean(train_loss_lm, axis=0))
                tl_std = tf.squeeze(tf.math.reduce_std(train_loss_lm, axis=0))
                tcd_mean = tf.squeeze(tf.reduce_mean(train_coord_dist_lm, axis=0))
                tcd_std = tf.squeeze(tf.math.reduce_std(train_coord_dist_lm, axis=0))

                v_mean = tf.reduce_mean(val_loss)
                v_std = tf.math.reduce_std(val_loss)
                vl_mean = tf.squeeze(tf.reduce_mean(val_loss_lm, axis=0))
                vl_std = tf.squeeze(tf.math.reduce_std(val_loss_lm, axis=0))
                vcd_mean = tf.squeeze(tf.reduce_mean(val_coord_dist_lm, axis=0))
                vcd_std = tf.squeeze(tf.math.reduce_std(val_coord_dist_lm, axis=0))
                mrg_mean = tf.squeeze(tf.reduce_mean(mrg_lm, axis=0))
                mrg_std = tf.squeeze(tf.math.reduce_std(mrg_lm, axis=0))
                cgt_mean = tf.squeeze(tf.reduce_mean(cgt_lm, axis=0))
                cgt_std = tf.squeeze(tf.math.reduce_std(cgt_lm, axis=0))
                
                elapsed_time = int(time.time() - start_time)

                self.val_store(step,
                               elapsed_time,
                               (t_mean, t_std), 
                               (tl_mean, tl_std), 
                               (tcd_mean, tcd_std),
                               (v_mean, v_std),
                               (vl_mean, vl_std),
                               (vcd_mean, vcd_std),
                               (mrg_mean, mrg_std),
                               (cgt_mean, cgt_std))
                

                train_loss =[]
                train_loss_lm = []
                val_loss = []
                val_loss_lm = []
                train_coord_dist_lm = []
                val_coord_dist_lm = []
                mrg_lm = [] 
                cgt_lm = []
            
        
            
            if step % training_params["checkpoint_interval"] == 0:
                self.model.save_weights(self.cp_path.format(step=step))
                print("saved cp-{:04d}".format(step))
        
        test_loss = [] # TODO : fix test for simul loop variant, fix loss for simul loop variant (high value in beginning ? )
        test_kp_loss = []
        test_c_dist = []
        test_mrg = []
        test_cgt = []
        for _ in range(self.dataset_test_size // self.strategy.num_replicas_in_sync):
            img_t, lab_t, fn = next(test)
            t_loss, t_kp_loss, t_c_dist, t_within_margin, t_closest_to_gt = self.distributed_test_step(img_t, lab_t, fn)
            test_loss.append(t_loss)
            test_kp_loss.append([t_kp_loss])
            test_c_dist.append([t_c_dist])
            test_mrg.append(t_within_margin)
            test_cgt.append(t_closest_to_gt)
        
        test_res = [np.array(tf.squeeze(tf.reduce_mean(i, axis=0))) for i in [test_loss, test_kp_loss, test_c_dist, test_mrg, test_cgt]]
        test_std = [np.array(tf.squeeze(tf.math.reduce_std(i, axis=0))) for i in [test_loss, test_kp_loss, test_c_dist, test_mrg, test_cgt]]
        total_time = int(time.time() - start_time)
        with open(os.path.join(self.log_path, 'test_res.txt'), 'ab') as testtxt:
            for i in range(len(test_res)):
                np.savetxt(testtxt, [test_res[i]], fmt='%3.3f', delimiter=",")
                np.savetxt(testtxt, [test_std[i]], fmt='%3.3f', delimiter=",")
            np.savetxt(testtxt, [total_time], fmt='%3.3f', delimiter=",")


def vis_points(image, points, diameter=5, given_kp=None):
    im = (image.copy()+1.)/2. # h,w
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for (w, h) in points:
        cv2.circle(im, (int(h), int(w)), diameter, (1., 0., 0.), -1)
    if given_kp is not None:
        for (w, h) in given_kp:
            cv2.circle(im, (int(h), int(w)), diameter, (0., 1., 0.), -1)
    plt.imshow(im)

def create_dir(path, fold):
    previous_runs = [i for i in os.listdir(path) if "run_" in i]
    if fold != 0:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs])
    else:
        if len(previous_runs) == 0:
            run_number = 1
        else:
            run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    logdir = 'run_%02d/fold_%02d' % (run_number, fold)
    l_dir = os.path.join(path, logdir)
    os.makedirs(l_dir)
    cp_dir = l_dir +'//cp/cp-{step:04d}'
    return l_dir, cp_dir

# TODO fix when actually considering loading a checkpoint -> for now this just does not work
def load_dir(path, run_number, step):
    logdir = 'run_%02d' % run_number
    l_dir = os.path.join(path, logdir)
    cp_pth = l_dir+'/cp/cp-{step:04d}'
    cp_dir = l_dir+'/cp/cp-{step:04d}'.format(step=step)
    #cp_dir = os.path.dirname(cp_pth)
    return l_dir, cp_pth, cp_dir

def store_parameters(data_config, opti_config, unet_config, ntm_config, attn_config, training_params, log_path):
    params = {"data_config":data_config,
              "opti_config":opti_config,
              "unet_config":unet_config,
              "ntm_config":ntm_config,
              "attn_config":attn_config,
              "training_params":training_params
             }
    with open(os.path.join(log_path,'params.json'), 'w') as fp:
        json.dump(params, fp)
    fp.close()

def main(path, data_dir, data_config, opti_config, unet_config, ntm_config, attn_config, training_params, run_number=None, start_steps=0):
    devices = ['/device:GPU:{}'.format(i) for i in range(training_params["num_gpu"])]
    strategy = tf.distribute.MirroredStrategy(devices)
    dataset = data.Data_Loader(data_path=data_dir,
                               name=data_config['dataset'],
                               batch_size=data_config["batch_size"]*strategy.num_replicas_in_sync,
                               train_pct=data_config["train_pct"],
                               n_folds=data_config["n_folds"],
                               n_aug_rounds=data_config["n_aug_rounds"],
                               repeat=data_config["repeat"],
                               prefetch=data_config["prefetch"],
                               sigma=data_config["sigma"])

    dataset(im_size=data_config["im_size"], keypoints=data_config["kp_list_in"])
    if training_params["mode"] == "iter":
        seq_len = dataset.n_landmarks//data_config["lm_count"]
        num_landmarks = data_config["lm_count"]
    elif training_params["mode"] == "simul":
        seq_len = 1
        len_kp = (len(data_config["kp_list_in"])-1) if data_config["kp_list_in"] is not None else 0
        num_landmarks = dataset.n_landmarks - len_kp
    else:
        raise ValueError("training_params[\"mode\"] must be either iter or simul")

    cp_dir = None
    # TODO make checkpoint loading work again -> todo when you actually get somewhere with this code...
    # if start_steps > 0:
    #     if start_steps % training_params["checkpoint_interval"] != 0:
    #         start_steps = int(np.round(float(start_steps) / training_params["checkpoint_interval"], 0) * training_params["checkpoint_interval"])
    #     log_path, cp_path, cp_dir = load_dir(path, run_number, start_steps)
    #     data_config, opti_config, unet_config, ntm_config, training_params = load_parameters(log_path) #this now always overwrites any given params, TODO
    # else:
    for fold in range(dataset.n_folds):
        log_path, cp_path = create_dir(path, fold)
        dataset.prep_fold(fold)

        if run_number is None:
            store_parameters(data_config, opti_config, unet_config, ntm_config, attn_config, training_params, log_path)
        
        with strategy.scope():
            model = unet.unet2d(num_fmaps=unet_config["num_filters"],
                                kernel_size=unet_config["kernel_size"],
                                fmap_inc_factor=unet_config["fmap_inc_factor"],
                                downsample_factors=unet_config["ds_factors"],
                                num_landmarks=num_landmarks,
                                seq_len=seq_len,
                                ntm_config=ntm_config,
                                attn_config=attn_config,
                                batch_size=data_config["batch_size"],
                                im_size=data_config["im_size"]
                                )

            if cp_dir is not None:
                model.load_weights(cp_dir)
            
            train_data = strategy.experimental_distribute_dataset(dataset.train_data)
            val_data = strategy.experimental_distribute_dataset(dataset.val_data)
            test_data = strategy.experimental_distribute_dataset(dataset.test_data)

            trainer = Train(model=model, strategy=strategy, data_config=data_config, opti_config=opti_config, training_params=training_params, data_landmarks=dataset.n_landmarks, dataset_test_size=dataset.test_size, log_path=log_path, cp_path=cp_path)
            trainer.iter_loop(train=train_data, val=val_data, test=test_data, start_steps=start_steps)

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data["data_config"], data["opti_config"], data["unet_config"], data["ntm_config"], data["attn_config"], data["training_params"]

if __name__ == "__main__":
    '''
    config explanation:
    Config is stored and loaded as .json into a python dict

    #### data_config
    dataset: 'droso' or 'cephal', str, choose which dataset to run experiment with
    batch_size: 2, int8, batch size, needs to be divisible by number of GPUs -> batch_size = GLOBAL_BATCH_SIZE
    im_size: [256,256], int tuple, resize images to this size
    sigma: 1., float, size of gaussian blob on heatmap
    lm_count: 5, int8, how many landmarks to put does output layer predict (used for iterative and non iterative loop) #TODO is this true?
    kp_list_in: [0,1,3,5], list of int8, which landmarks to put into input for non-iterative learning task, None or [0] for no input kps -> this is only for non iterative loop
    train_pct, 10, int, percent of total data to take as training set (will be split by N_folds again) 
    n_folds, 3, int, how many CV folds
    repeat, true, bool, whether to repeat the dataset infinitely
    prefetch, true, bool, whether to prefetch samples from the tf.data dataset
    n_aug_rounds, 10, int8, How many altered versions of the train dataset to append, i.e. 10 = 10 transformed versions of the same dataset are appended to the original (also transformed) dataset

    #### opti_config
    learning_rate, 1e-3, float, optimizer (adam) learning rate
    adam_beta_1, 0.9, float The exponential decay rate for the 1st moment estimates   
    adam_beta_2,  0.999, float, The exponential decay rate for the 2nd moment estimates
    adam_epsilon, 1e-7, float, A small constant for numerical stability 
    decay_rate: .9, float, rate at which to decay lr -> adam only adjusts learning rate by last n gradients not globally # see: initial_learning_rate * decay_rate ^ (step / decay_steps)
    decay_steps: 500, int, exponent to accelerate decay after x steps # ^ 

    #### unet_config
    num_filters, 16, int, number of filters on first layer of Unet
    kernel_size, 3, int, kernel size for all convolutional layers (future maybe list)
    fmap_inc_factor, 2, int, multiplier on how much to increase number of filters at each level
    ds_factors, [[2,2],[2,2],[2,2],[2,2],[2,2]], list of int tuples, defines depth of Unet, also defines how much is downsampled at each depth step

    #### ntm_config
    !! contains multiple configs, depending on how many ntm layers you want to include in the unet i.e. a ntm layer  !!
    !! on first unet level is provided by feeding a dict with name = unet level {"0":{"enc_dec_param":{...}}}        !!
    !! ntm_config further contains two sub-dictionaries: enc_dec_param, ntm_param                                    !!
    
        ###### enc_dec_param, this is finnicky and needs to match the unet position
        num_filters, 16, int,  number of conv filters at each layer
        kernel_size, 3, int, conv_layer kernel size
        pool_size, [4,4], list of int, defines number of encoder decoder layers and how much is pooled
        pos:, one of:["l","r","b"], str, defines the lateral position of the enc dec -> either at downsampling, upsampling or both sides
    
        ###### ntm_param, set to None if no NTM, just encoder-decoder architecture
        controller_units, 256, int, number of units of controller dense layer, controller defines what and how to write and read
        memory_size, 64, int, length of a single memory entry
        memory_vector_dim, 256, int, number of memory entries
        output_dim, 256, int, output dimensions of NTM, has to be square-rootable as it is reshaped to square format and then upsampled needs to match the Unet layer output
        read_head_num, 3, int, number of simultaneous read heads
        write_head_num, 3 int, number of simultaneous write heads
        init_mode, "constant", str, constant for constant init, anything else for random (missing learned init, but that just adds complexity without much gain)

    #### attn_config
    # same structure as ntm_config
    num_filters, 256, int, number of intermediate filters in the attention gate (see https://arxiv.org/pdf/1804.03999.pdf)
    
    #### training_params
    num_training_iterations, 10000, int, number of training steps, for iterative approach this also means, number of samples put in (does not get reduced by multiple iterations over the same image)
    validation_steps, 5, int, number of validations iterations over which metrics are averaged and reported
    report_interval, 50, int, number of steps after which a validation step happens
    kp_metric_margin, 3, int, pixel margin at which a kp is counted as close enough
    checkpoint_interval, 500, int, number of steps at which a model checkpoints happens, -> can only be used to reuse model for predictions, not to resume training, because no way to ensure data consistency currently
    num_test_samples, 5, int, number of samples (samples*batch_size) to test the model on, these samples are also printed and stored
    mode, ["iter", "simul"], one of the two strings, defines in which mode the network learns / predicts, either iteratively learning landmarks or all landmarks simultaneously (or with kp_list_in input landmarks)
    store_samples, 1 for True anything else for false, int, defines if samples are being stored at test time
    num_gpu, 4, int, number of gpus to use
    '''
    PATH = '/fast/AG_Kainmueller/elbauma/landmark-ntm/experiments' # can define this explicitely to be an experiment folder to re-run select experiments
    DATA_DIR = '/fast/AG_Kainmueller/elbauma/landmark-ntm/datasets/'

    if args.conf is not None:
        PATH = args.conf

    path_list = [(dirpath,filename) for dirpath, _, filenames in os.walk(PATH) for filename in filenames if filename.endswith('config.json')] # searching for all experiments excluding stored jsons of ran experiments
    for experiment in path_list:
        data_config, opti_config, unet_config, ntm_config, attn_config, training_params = read_json(os.path.join(experiment[0], experiment[1]))
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = [str(i) for i in range(training_params["num_gpu"])]
        main(experiment[0], DATA_DIR, data_config, opti_config, unet_config, ntm_config, attn_config, training_params)
    

    
    