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

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"



parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, default=None, help='Select a directory in which to search for config.json to execute')
parser.add_argument('--load', action='store_true', help="Load from checkpoint")
parser.add_argument('--gpus', type=str, default="4,5,6,7", help="Choose which gpus to run experiment on")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

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
        self.print_samples = True if training_params["store_samples"]>=1 else False
        self.predict_self = True if training_params["self_predict"]==1 else False

        self.w_scale = 0.1 * 1935.0 / self.im_size[0] if data_config["dataset"] == "cephal" else 0.645796 * 3840.0 / self.im_size[0] # for mm/µm results
        self.h_scale = 0.1 * 2400.0 / self.im_size[0] if data_config["dataset"] == "cephal" else 0.645796 * 3234.0 / self.im_size[0] # for mm/µm results

    def dist_ssd_loss(self, gt_labels, logits):
        return tf.reduce_sum(tf.square(gt_labels-logits), axis=None)
    
    def compute_loss(self, gt_labels, logits):
        per_example_loss = self.dist_ssd_loss(gt_labels, logits)
        return per_example_loss * self.batch_div

    def dist_coord_dist(self, gt_labels, logits):
        logits = self.get_max_indices_argmax(logits)
        gt_labels = self.get_max_indices_argmax(gt_labels)
        crd = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(gt_labels-logits), axis=-1)), axis=-1) #pythagoras, mean batch, not entirely accurate for pixel images?
        return crd

    def dist_coord_dist_mm(self, gt_labels, logits): # same as above, fix later
        logits_mm = self.get_max_indices_argmax(logits, True)
        gt_labels_mm = self.get_max_indices_argmax(gt_labels, True)
        crd_mm = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(gt_labels_mm-logits_mm), axis=-1)), axis=-1) 
        return crd_mm

    def dist_cgt(self, gt_labels, logits):
        y_pred_n = self.get_max_indices_argmax(logits) # shape = n_lm,1,1,2
        y_true_n = self.get_max_indices_argmax(gt_labels)
        exp_y_pred = tf.expand_dims(y_pred_n, 0)
        exp_y_true = tf.expand_dims(y_true_n, 1)
        closest = tf.square(tf.subtract(exp_y_pred, exp_y_true)) # using TF broadcast to create distance table
        closest_red = tf.argmin(tf.reduce_mean(closest, axis=-1), axis=1) # find min distance
        closest_to_nearest = tf.reduce_sum(tf.cast(tf.equal(tf.transpose(tf.cast(closest_red, tf.int32)), tf.range(tf.shape(closest_red)[0], dtype=tf.int32)), dtype=tf.float32), axis=0) 
        return closest_to_nearest*self.batch_div

    def dist_outliers(self, gt_labels, logits, margin): # TODO could use already calculated crd from above -> fix later
        y_pred = self.get_max_indices_argmax(logits) # shape = n_lm,1,1,2
        y_true = self.get_max_indices_argmax(gt_labels)
        cdist = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))
        o_1 = tf.reduce_sum(tf.cast(tf.less(tf.cast(margin,tf.float32), cdist), tf.float32),axis=-1)
        o_2 = tf.reduce_sum(tf.cast(tf.less(tf.cast(margin*2,tf.float32), cdist), tf.float32),axis=-1) # TODO make this definable in config by e.g. list or something
        o_3 = tf.reduce_sum(tf.cast(tf.less(tf.cast(margin*5,tf.float32), cdist), tf.float32),axis=-1)
        return o_1*self.batch_div, o_2*self.batch_div, o_3*self.batch_div
    
    def dist_outliers_mm(self, gt_labels, logits, margin): # TODO remove mm duplicate functions
        y_pred = self.get_max_indices_argmax(logits, True) # shape = n_lm,1,1,2
        y_true = self.get_max_indices_argmax(gt_labels, True)
        cdist = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))
        o_1 = tf.reduce_sum(tf.cast(tf.less(tf.cast(margin,tf.float32), cdist), tf.float32),axis=-1)
        o_2 = tf.reduce_sum(tf.cast(tf.less(tf.cast(margin*2,tf.float32), cdist), tf.float32),axis=-1) 
        o_3 = tf.reduce_sum(tf.cast(tf.less(tf.cast(margin*5,tf.float32), cdist), tf.float32),axis=-1)
        return o_1*self.batch_div, o_2*self.batch_div, o_3*self.batch_div
    
    def get_max_indices_argmax(self, logits, mm=False): # TODO correctly make use of H,W dimensions
        flat_logits = tf.reshape(logits, tf.concat([tf.shape(logits)[:-2], [-1]], axis=0))
        max_val = tf.cast(tf.argmax(flat_logits, axis=-1), tf.int32)
        w = max_val // tf.shape(logits)[-1]
        h = max_val % tf.shape(logits)[-1]
        if mm:
            res = tf.concat((tf.cast(w,tf.float32)*self.w_scale,tf.cast(h,tf.float32)*self.h_scale), axis=-1)
        else:
            res = tf.concat((w,h), axis=-1)
        return tf.cast(res, tf.float32)

    def convert_input(self, img, lab, lm, lm_count, iter):
        if not self.iter:
            inp = tf.expand_dims(img, axis=0)
            lab = tf.expand_dims(lab, axis=0)
            return inp, lab
        ep_lab_0 = tf.fill(tf.shape(lab[:,0:lm_count,:,:]), 1e-4)
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
        c_dist_mm = tf.map_fn(lambda y: self.dist_coord_dist_mm(y[0], y[1]), (lab, pred), dtype=tf.float32)*self.batch_div
        return lab, pred, kp_loss, c_dist, c_dist_mm

    def train_step(self, inp, lab): #TODO
        inp, lab = self.convert_input(inp, lab, self.data_landmarks, self.data_config["lm_count"], self.iter)
        with tf.GradientTape() as tape:
            pred, _, _ = self.model(inp)
            loss = self.compute_loss(lab, pred)
        grad = tape.gradient(loss, self.model.trainable_weights)
        clipped_grad, _ = tf.clip_by_global_norm(grad, 10000.0)
        self.optimizer.apply_gradients(zip(clipped_grad, self.model.trainable_weights))
        _, _, kp_loss, c_dist, c_dist_mm = self.kp_loss_c_dist(lab, pred)
        return loss, kp_loss, c_dist, c_dist_mm

    def train_step_sp(self, inp, lab): #TODO
        inp, lab = self.convert_input(inp, lab, self.data_landmarks, self.data_config["lm_count"], self.iter)
        with tf.GradientTape() as tape:
            pred, _, _ = self.model.pred_self(inp)
            loss = self.compute_loss(lab, pred)
            loss += tf.nn.scale_regularization_loss(tf.reduce_sum(self.model.losses))
        grad = tape.gradient(loss, self.model.trainable_weights)
        clipped_grad, _ = tf.clip_by_global_norm(grad, 10000.0)
        self.optimizer.apply_gradients(zip(clipped_grad, self.model.trainable_weights))
        _, _, kp_loss, c_dist, c_dist_mm = self.kp_loss_c_dist(lab, pred)
        return loss, kp_loss, c_dist, c_dist_mm

    def val_step(self, inp, lab):
        inp, lab = self.convert_input(inp, lab, self.data_landmarks, self.data_config["lm_count"], self.iter)
        pred, _, _ = self.model.pred_self(inp, training=False)
        loss = self.compute_loss(lab, pred)
        lab, pred, kp_loss, c_dist, c_dist_mm = self.kp_loss_c_dist(lab, pred)
        closest_to_gt = self.dist_cgt(lab, pred)
        o_1, o_2, o_3 = self.dist_outliers(lab, pred, self.kp_margin)
        o_1_mm, o_2_mm, o_3_mm = self.dist_outliers_mm(lab, pred, self.kp_margin)
        return  loss, kp_loss, c_dist, c_dist_mm, o_1, o_2, o_3, o_1_mm, o_2_mm, o_3_mm, closest_to_gt
    
    def get_mems(self, mem):
        mems = []
        for i in mem:
            for j in i:
                try:
                    mems.append(j["M"])
                except TypeError as err:
                    pass
        return mems

    def flatten_maps(self, maps):
        return [i for j in maps for i in j]

    def test_step(self, inp, lab, fn):
        inp, lab = self.convert_input(inp, lab, self.data_landmarks, self.data_config["lm_count"], self.iter)
        img = inp[0,:,:1,:,:]
        # tf.print(fn)
        # print(fn)
        if not (self.data_config["kp_list_in"] is None or self.data_config["kp_list_in"] == [0]):
            given_kp = tf.expand_dims(tf.reshape(tf.transpose(tf.expand_dims(inp[0,:,1:,:,:],axis=0), [0,2,1,3,4]), [-1, self.data_config["batch_size"], self.im_size[0], self.im_size[0]]), axis=2)
        else:
            given_kp = tf.constant(0.)
        pred, states, attn_maps = self.model.pred_self(inp, training=False) if self.iter else self.model(inp, training=False)
        loss = self.compute_loss(lab, pred)
        lab, pred, kp_loss, c_dist, c_dist_mm = self.kp_loss_c_dist(lab, pred)
        closest_to_gt = self.dist_cgt(lab, pred)
        o_1, o_2, o_3 = self.dist_outliers(lab, pred, self.kp_margin)
        o_1_mm, o_2_mm, o_3_mm = self.dist_outliers_mm(lab, pred, self.kp_margin)

        if self.model.ntm_config is not None:
            mem_names = tf.py_function(func=self.store_samples, inp=[img, pred, lab, fn, False, given_kp], Tout=[tf.string])
            cnt = 0
            for i in self.get_mems(states):
                cnt += tf.py_function(func=self.store_mem, inp=[i, mem_names, cnt], Tout=tf.int32)
        
        if self.model.attn_config is not None or self.model.ntm_config is not None:
            attn_names = tf.py_function(func=self.store_samples, inp=[img, pred, lab, fn, True, given_kp], Tout=[tf.string])
            cnt = 0
            for i in self.flatten_maps(attn_maps):
                cnt += tf.py_function(func=self.store_attn, inp=[i, attn_names, cnt], Tout=tf.int32)
        return loss, kp_loss, c_dist, c_dist_mm, o_1, o_2, o_3, o_1_mm, o_2_mm, o_3_mm, closest_to_gt
    
    def min_max_scale(self, inp):
        return (inp-inp.min())/(inp.max()-inp.min())

    def store_attn(self, attn, names, i):
        img = attn.numpy().squeeze()
        if np.sum(img) == 0:
            return 0
        
        name = names.numpy().flatten()#
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        plt.savefig(name[i.numpy()].decode('UTF-8'))
        return 1

    def store_mem(self, states, names, i):
        img = states.numpy().squeeze()
        img = (img+1.)/2.
        name = names.numpy().flatten()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        plt.savefig(name[i.numpy()].decode('UTF-8'))
        return 1
        

    def store_samples(self, img, pred, lab, filenames, at, given_kp = None):
        # tf.print(filenames)
        # print(filenames)
        # if self.data_config["batch_size"] == 1:
        #     filenames = [filenames.decode('UTF-8')]
        # else:
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
        names = []
        for i in range(img.shape[0]): # per batch iteration                
            vis_points(img[i], pred_keypoints[i], 3, given_kp=given_kp[i])
            plt.savefig(self.log_path+'/samples/'+filenames[i]+'_pred.png')
            vis_points(img[i], lab_keypoints[i], 3, given_kp=given_kp[i])
            plt.savefig(self.log_path+'/samples/'+filenames[i]+'_gt.png')
            plt.imshow(cv2.cvtColor(pred_logits[i], cv2.COLOR_GRAY2BGR))
            plt.savefig(self.log_path+'/samples/'+filenames[i]+'_pred_logits.png')
            if at:
                names.append(self.get_attn_names(filenames[i], i))
            else:
                names.append(self.get_mem_names(filenames[i], i))
        return names
            
    def get_attn_names(self, fn, batch_no):
        names = []
        if self.model.attn_config is None and self.model.ntm_config is None:
            return names 
        keys = list(map(int, self.model.attn_config.keys())) if self.model.ntm_config is None else list(map(int, self.model.ntm_config.keys()))
        if not os.path.exists(self.log_path+'/samples/'+fn):
            os.makedirs(self.log_path+'/samples/'+fn)
        for i in range(self.data_landmarks//self.data_config["lm_count"]) if self.training_params["mode"] == "iter" else range(1):
            for key in keys:
                names.append(self.log_path+'/samples/'+fn+'/'+str(key)+'_'+str(i)+'attn.png')
        return names

    def get_mem_names(self, fn, batch_no):
        names = []
        if self.model.ntm_config is None:
            return names
        keys = list(map(int, self.model.ntm_config.keys()))
        if not os.path.exists(self.log_path+'/samples/'+fn):
            os.makedirs(self.log_path+'/samples/'+fn)
        for i in range(self.data_landmarks//self.data_config["lm_count"]) if self.training_params["mode"] == "iter" else range(1):
            for key in keys:
                pos = self.model.ntm_config[str(key)]['enc_dec_param']['pos']
                if pos == "b":
                    names.append(self.log_path+'/samples/'+fn+'/'+str(key)+'_l_'+str(i)+'mem.png') 
                    names.append(self.log_path+'/samples/'+fn+'/'+str(key)+'_r_'+str(i)+'mem.png') 
                else:
                    names.append(self.log_path+'/samples/'+fn+'/'+str(key)+'_'+pos+'_'+str(i)+'mem.png')
        return names

    def sred(self, x):
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=None)

    @tf.function
    def distributed_train_step(self, inp, lab):
        per_replica_loss, pr_kp_loss, pr_c_dist, pr_c_dist_mm = self.strategy.experimental_run_v2(self.train_step, args=(inp, lab,))
        return self.sred(per_replica_loss), self.sred(pr_kp_loss), self.sred(pr_c_dist), self.sred(pr_c_dist_mm) #, self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_kp_loss, axis=None), self.strategy.reduce(tf.distribute.ReduceOp.SUM, pr_c_dist, axis=None)

    @tf.function
    def distributed_train_step_sp(self, inp, lab):
        per_replica_loss, pr_kp_loss, pr_c_dist, pr_c_dist_mm = self.strategy.experimental_run_v2(self.train_step_sp, args=(inp, lab,))
        return self.sred(per_replica_loss), self.sred(pr_kp_loss), self.sred(pr_c_dist), self.sred(pr_c_dist_mm)

    @tf.function
    def distributed_val_step(self, inp, lab):
        loss, kp_loss, c_dist, c_dist_mm, o_1, o_2, o_3, o_1_mm, o_2_mm, o_3_mm, closest_to_gt = self.strategy.experimental_run_v2(self.val_step, args=(inp, lab,))
        return self.sred(loss), self.sred(kp_loss), self.sred(c_dist), self.sred(c_dist_mm), self.sred(o_1), self.sred(o_2), self.sred(o_3), self.sred(o_1_mm), self.sred(o_2_mm), self.sred(o_3_mm), self.sred(closest_to_gt)
    
    @tf.function
    def distributed_test_step(self, inp, lab, fn):
        loss, kp_loss, c_dist, c_dist_mm, o_1, o_2, o_3, o_1_mm, o_2_mm, o_3_mm, closest_to_gt = self.strategy.experimental_run_v2(self.test_step, args=(inp, lab, fn, ))
        return self.sred(loss), self.sred(kp_loss), self.sred(c_dist), self.sred(c_dist_mm), self.sred(o_1), self.sred(o_2), self.sred(o_3), self.sred(o_1_mm), self.sred(o_2_mm), self.sred(o_3_mm), self.sred(closest_to_gt)

    def val_store(self, names, metrics):
        for i,name in enumerate(names):
            with open(os.path.join(self.log_path, name), 'ab') as txt:
                np.savetxt(txt, [np.array(metrics[i][0])], fmt='%.3f', delimiter=",")
                np.savetxt(txt, [np.array(metrics[i][1])], fmt='%.3f', delimiter=",")    
                

    def iter_loop(self, train, val, test, start_steps):
        train = iter(train)
        val = iter(val)
        test = iter(test)

        train_loss = [] # TODO fix this mess at some point
        train_loss_lm = []
        val_loss = []
        val_loss_lm = []
        train_coord_dist_lm = []
        train_coord_dist_lm_mm = []
        val_coord_dist_lm = []
        val_coord_dist_lm_mm = []
        out_1_lm = []
        out_2_lm = []
        out_3_lm = []
        out_1_lm_mm = []
        out_2_lm_mm = []
        out_3_lm_mm = []
        cgt_lm = []

        tf.print("Starting train loop...")
        start_time = time.time()
        for step in range(start_steps, self.training_params["num_training_iterations"]+1):
            img, lab, _ = next(train)
            if self.predict_self:
                loss, kp_loss, c_dist, c_dist_mm = self.distributed_train_step_sp(img, lab)
            else:
                loss, kp_loss, c_dist, c_dist_mm = self.distributed_train_step(img, lab)
            
            train_loss.append(loss)
            train_loss_lm.append([kp_loss])
            train_coord_dist_lm.append([c_dist])
            train_coord_dist_lm_mm.append([c_dist_mm])

            if step % training_params["report_interval"] == 0:
                for _ in range(training_params["validation_steps"]):
                    img_v, lab_v, _ = next(val)
                    v_loss, v_kp_loss, v_c_dist, v_c_dist_mm, v_o_1, v_o_2, v_o_3, v_o_1_mm, v_o_2_mm, v_o_3_mm, v_closest_to_gt = self.distributed_val_step(img_v, lab_v)

                    out_1_lm.append(v_o_1)
                    out_2_lm.append(v_o_2)
                    out_3_lm.append(v_o_3)
                    out_1_lm_mm.append(v_o_1_mm)
                    out_2_lm_mm.append(v_o_2_mm)
                    out_3_lm_mm.append(v_o_3_mm)

                    cgt_lm.append(v_closest_to_gt)
                    val_loss.append(v_loss)
                    val_loss_lm.append([v_kp_loss])
                    val_coord_dist_lm.append([v_c_dist])
                    val_coord_dist_lm_mm.append([v_c_dist_mm])
                
                elapsed_time = int(time.time() - start_time)
                val_res = [(tf.squeeze(tf.reduce_mean(i, axis=0)), tf.squeeze(tf.math.reduce_std(i, axis=0))) for i in [train_loss, train_loss_lm, val_loss, val_loss_lm, train_coord_dist_lm, train_coord_dist_lm_mm, val_coord_dist_lm, 
                                                                                                                        val_coord_dist_lm_mm, out_1_lm, out_2_lm, out_3_lm, out_1_lm_mm, out_2_lm_mm, out_3_lm_mm, cgt_lm]]
                tf.print("Iteration", step , "(Elapsed: ", elapsed_time, "s):")
                tf.print("mean train loss:", val_res[0][0], "std:", val_res[0][1], summarize=-1)
                tf.print("mean validation loss:",  val_res[2][0], "std:", val_res[2][1], summarize=-1)
                metric_list = ['train_loss.txt', 'train_loss_kp.txt', 'val_loss.txt', 'val_loss_kp.txt', 'train_coordd.txt','train_coordd_mm.txt', 'val_coordd.txt','val_coordd_mm.txt',
                                'val_outliers_1.txt', 'val_outliers_2.txt', 'val_outliers_3.txt','val_outliers_1_mm.txt', 'val_outliers_2_mm.txt', 'val_outliers_3_mm.txt', 'val_closest_gt.txt']
                self.val_store(metric_list, val_res)
                
                train_loss = []
                train_loss_lm = []
                val_loss = []
                val_loss_lm = []
                train_coord_dist_lm = []
                train_coord_dist_lm_mm = []
                val_coord_dist_lm = []
                val_coord_dist_lm_mm = []
                out_1_lm = []
                out_2_lm = []
                out_3_lm = []
                out_1_lm_mm = []
                out_2_lm_mm = []
                out_3_lm_mm = []
                cgt_lm = []
                       
            if step % training_params["checkpoint_interval"] == 0:
                self.model.save_weights(self.cp_path.format(step=step))
                print("saved cp-{:04d}".format(step))
        
        test_loss = [] # TODO also fix this mess
        test_kp_loss = []
        test_c_dist = []
        test_c_dist_mm = []
        test_o_1 = []
        test_o_2 = []
        test_o_3 = []
        test_o_1_mm = []
        test_o_2_mm = []
        test_o_3_mm = []
        test_cgt = []
        sample_count = 0#self.training_params["store_samples"] if start_steps == self.training_params["num_training_iterations"]+1 else 0 # TODO can edit this if we ever want to be able to plot more samples
        for _ in range(self.dataset_test_size // self.strategy.num_replicas_in_sync):
            img_t, lab_t, fn = next(test)
            if sample_count < self.training_params["store_samples"]:
                t_loss, t_kp_loss, t_c_dist, t_c_dist_mm, t_o_1,t_o_2,t_o_3, t_o_1_mm, t_o_2_mm, t_o_3_mm, t_closest_to_gt = self.distributed_test_step(img_t, lab_t, fn)
            else:
                t_loss, t_kp_loss, t_c_dist, t_c_dist_mm, t_o_1,t_o_2,t_o_3, t_o_1_mm, t_o_2_mm, t_o_3_mm, t_closest_to_gt  = self.distributed_val_step(img_t, lab_t)
            test_loss.append(t_loss)
            test_kp_loss.append([t_kp_loss])
            test_c_dist.append([t_c_dist])
            test_c_dist_mm.append([t_c_dist_mm])
            test_o_1.append(t_o_1)
            test_o_2.append(t_o_2)
            test_o_3.append(t_o_3)
            test_o_1_mm.append(t_o_1_mm)
            test_o_2_mm.append(t_o_2_mm)
            test_o_3_mm.append(t_o_3_mm)
            test_cgt.append(t_closest_to_gt)
            sample_count +=1
        
        test_res = [np.array(tf.squeeze(tf.reduce_mean(i, axis=0))) for i in [test_loss, test_kp_loss, test_c_dist, test_c_dist_mm, test_o_1, test_o_1_mm, test_o_2, test_o_2_mm, test_o_3, test_o_3_mm, test_cgt]]
        test_std = [np.array(tf.squeeze(tf.math.reduce_std(i, axis=0))) for i in [test_loss, test_kp_loss, test_c_dist, test_c_dist_mm, test_o_1, test_o_1_mm, test_o_2, test_o_2_mm, test_o_3, test_o_3_mm, test_cgt]]
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
def load_dir(path, run_number, step, fold):
    logdir = 'run_%02d/fold_%02d' % (run_number, fold)
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
    
    #     data_config, opti_config, unet_config, ntm_config, training_params = load_parameters(log_path) #this now always overwrites any given params, TODO
    # else:
    for fold in range(dataset.n_folds):
        if start_steps > 0:
            if start_steps % training_params["checkpoint_interval"] != 0:
                start_steps = int(np.round(float(start_steps) / training_params["checkpoint_interval"], 0) * training_params["checkpoint_interval"])
            log_path, cp_path, cp_dir = load_dir(path, run_number, start_steps, fold)
            start_steps += 1
        else:
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
    sigma: 2., float, size of gaussian blob on heatmap
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
        reg, 1, binary (bool), whether to use regularization
    
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
    self_predict, 1, binary, 1=true, 0=False, whether to use self predict @ train time
    store_samples, 10, int, number of samples to store as images x num gpus
    num_gpu, 4, int, number of gpus to use
    '''
    PATH = '/fast/AG_Kainmueller/elbauma/landmark-ntm/experiments' 
    DATA_DIR = '/fast/AG_Kainmueller/elbauma/landmark-ntm/datasets/'

    if args.conf is not None:
        PATH = args.conf

    path_list = [(dirpath,filename) for dirpath, _, filenames in os.walk(PATH) for filename in filenames if filename.endswith('config.json')] # searching for all experiments excluding stored jsons of ran experiments
    for experiment in path_list:
        data_config, opti_config, unet_config, ntm_config, attn_config, training_params = read_json(os.path.join(experiment[0], experiment[1]))
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = [str(i) for i in range(training_params["num_gpu"])]
        rn = 1 if args.load else None
        steps = training_params["num_training_iterations"] if args.load else 0
        main(experiment[0], DATA_DIR, data_config, opti_config, unet_config, ntm_config, attn_config, training_params, rn, steps)
    

    
    