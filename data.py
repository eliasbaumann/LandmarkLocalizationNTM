import pathlib
import warnings
import tensorflow as tf

import imgaug as ia
import imgaug.augmenters as iaa

import numpy as np
from heatmapgen import generate_heatmaps
import matplotlib.pyplot as plt

@tf.function
def decode_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=1)#tf.cast(, tf.float32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.subtract(tf.scalar_mul(tf.constant(2., dtype=tf.float32), img),tf.constant(1., dtype=tf.float32))
    return img

class Data_Loader():
    def __init__(self, data_path, name, batch_size, train_pct=80, test_pct=20, n_folds=3, repeat=True, prefetch=True, n_aug_rounds=5, sigma=1.):
        self.path = data_path
        self.name = name
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.n_folds = n_folds
        self.repeat = repeat
        self.prefetch = prefetch
        self.n_aug_rounds = n_aug_rounds
        self.sigma = sigma
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.n_landmarks = None
        self.ds_size = None
        self.keypoints = None
        self.orig_im_size = None
        self.im_size = None
        self.n_train_obs = None
        self.n_val_obs = None
        self.n_test_obs = None
        self.augmentations = []
        

    def __call__(self, im_size=None, keypoints=None):
        self.im_size = im_size
        print("Creating Datasets...")
        self.keypoints = keypoints
        if self.name == 'droso':
            data = self.load_droso()
        elif self.name == 'cephal':
            data = self.load_cephal()
        else:
            print("No correct dataset name given, please select from droso and cephal, defaulting to droso")
            data = self.load_droso()
        imx, imy = self.im_size
        data = self.resize_images(data, imx, self.orig_im_size[0], self.orig_im_size[1])
                
        # train test val split (take and skip)
        self.n_train_obs = int(self.ds_size * (self.train_pct/100.0) - (self.ds_size * (self.train_pct/100.0) % self.batch_size))
        self.n_val_obs = self.n_train_obs // self.n_folds if self.n_folds > 1 else self.n_train_obs // 5 # just do 20% of train for no n_folds defined
        self.n_train_obs = self.n_train_obs - self.n_val_obs
        self.n_test_obs = int(self.ds_size * (self.test_pct/100.0) - (self.ds_size * (self.test_pct/100.0) % self.batch_size))
       
        # if fold = 0
        # train_1 = take(0*n_val)
        # train_2 = skip(n_val).take(n_train)
        # val = skip(0*n_val).take(n_val)
        # if fold = 1
        # train_1 = take(n_val)
        # train_2 = skip(2*n_val).take(n_train-1*n_val)
        # val = skip(1*n_val).take(n_val)
        # if fold = 2
        # train_1 = take(2*n_val)
        # train_2 = skip(3*n_val).take(n_train-2*n_val)
        # val = skip(2*n_val).take(n_val)
        self.data_folds = []
        for i in range(self.n_folds):
            j = i+1
            train_1 = data.take(i*self.n_val_obs)
            train_2 = data.skip(j*self.n_val_obs).take(self.n_train_obs)
            train = train_1.concatenate(train_2)
            val = data.skip(i*self.n_val_obs).take(self.n_val_obs)
            self.data_folds.append([train, val])

        self.test_data = data.skip(self.n_train_obs+self.n_val_obs).take(self.n_test_obs)
        print("setup cv splits")
        
    
    def prep_fold(self, fold):
        imx, imy = self.im_size
        train, val = self.data_folds[fold]
        self.train_data = train
        self.val_data = val
        self.train_data = self.augment_data(self.train_data, imx, imy)

        if self.keypoints is not None:
            self.train_data, self.val_data, self.test_data = self.kp_to_input(self.train_data, self.val_data, self.test_data, self.keypoints)
        
        self.train_data = self.train_data.shuffle(buffer_size=self.n_train_obs, reshuffle_each_iteration=False)
        self.val_data = self.val_data.shuffle(buffer_size=self.n_val_obs, reshuffle_each_iteration=False)
        self.test_data = self.test_data.shuffle(buffer_size=self.n_test_obs, reshuffle_each_iteration=False)

        if self.repeat:
            self.train_data = self.train_data.repeat()
            self.val_data = self.val_data.repeat()
            self.test_data = self.test_data.repeat()
        
        self.train_data = self.train_data.batch(self.batch_size, drop_remainder=True)
        self.val_data = self.val_data.batch(self.batch_size, drop_remainder=True)
        self.test_data = self.test_data.batch(self.batch_size, drop_remainder=True)

        
        if self.prefetch:
            self.train_data = self.train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            self.val_data = self.val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print("prepared preprocessing")
  
    def resize_images(self, data, imx, origx, origy):   
        
        @tf.function
        def _rescale_lab(lab, imx, imy):
            h = tf.cast(lab[:,0], tf.float32) / tf.cast(origy, tf.float32) * imy
            w = tf.cast(lab[:,1], tf.float32) / tf.cast(origx, tf.float32) * imx
            return tf.stack([h,w], axis=1) 
        
        @tf.function
        def _tf_resize(img, lab, fn):
            lab = _rescale_lab(lab, imx, imx)
            keypoints = generate_heatmaps(lab, imx, self.n_landmarks, self.sigma)
            resized = tf.image.resize(img, [imx,imx]) 
            resized = tf.transpose(resized, perm=[2,0,1]) # convert to channels first
            resized.set_shape([1,imx,imx])
            keypoints.set_shape([self.n_landmarks,imx,imx])
            return resized, keypoints, fn

        def convert_all(images, keypoints, filename):
            return tf.data.Dataset.from_tensors((images, keypoints, filename)).map(_tf_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        return data.interleave(convert_all, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def augment_data(self, data, imx, imy):
        def _albu_transform(image, keypoints):
            image = np.concatenate((image,keypoints),axis=0)
            image = np.transpose(image, (1,2,0))
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            if self.name == 'droso':
                seq = iaa.Sequential([
                                    iaa.Fliplr(.3),
                                    iaa.Flipud(.3),
                                    sometimes(iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode="constant", pad_cval=1e-5)),
                                    sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                                        rotate=(-45, 45),
                                                        shear=(-16, 16),
                                                        order=[0, 1],
                                                        cval=1e-5,
                                                        mode="constant")),
                                    sometimes(iaa.ElasticTransformation(alpha=(0.0, 40.0), sigma=(4.0, 8.0), cval=1e-5, mode="constant")),
                                    ], random_order=True)
            elif self.name == 'cephal': # no up down flipping, reduced rotate, could reduce this even more, because images don't really differ here..
                seq = iaa.Sequential([
                                    sometimes(iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode="constant", pad_cval=1e-5)),
                                    sometimes(iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                                        rotate=(-5, 5),
                                                        shear=(-5, 5),
                                                        order=[0, 1],
                                                        cval=1e-5,
                                                        mode="constant")),
                                    sometimes(iaa.ElasticTransformation(alpha=(0.0, 20.0), sigma=(2.0, 4.0), cval=1e-5, mode="constant")),
                                    ], random_order=True)
            else:
                seq  =  iaa.Sequential([iaa.ElasticTransformation(alpha=(0.0, 40.0), sigma=(4.0, 8.0), cval=1e-5, mode="constant")], random_order=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning) # only doing this because imgaug warns me that my "image" is stacked with its heatmaps
                image_aug = seq.augment_image(image=image)
            image_aug = np.transpose(image_aug, (2,0,1))
            image = image_aug[:1]
            keypoints = image_aug[1:]
            return image, keypoints
          

        def _augment(img, lab, fn):
            image, keypoints = tf.numpy_function(_albu_transform, [img, lab], [tf.float32, tf.float32])
            image.set_shape([1,imx,imy])
            keypoints.set_shape([self.n_landmarks,imx,imy])
            return image, keypoints, fn


        def generate_augmentations(images, keypoints, filename):
            regular_ds = tf.data.Dataset.from_tensors((images, keypoints, filename)).map(_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            for _ in range(self.n_aug_rounds): 
                aug_ds = tf.data.Dataset.from_tensors((images, keypoints, filename)).map(_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                regular_ds.concatenate(aug_ds)
            return regular_ds

        return data.interleave(generate_augmentations, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    
    def kp_to_input(self, data, val_data, test_data , kp_list):
        """
        This function allows to input keypoints into the model 
        by selecting with a list, always starting with 0 
        i.e. [0,1,14,22] or [0,2,4,5]
        """
        inv_kp_list = np.repeat(True, self.n_landmarks+1) # all landmarks + image itself which has only one dimension in this case
        inv_kp_list[kp_list] = False
        inv_kp_ind = [i for i, j in enumerate(inv_kp_list, start=0) if j]
        kp_list = tf.constant(kp_list, dtype=tf.int32)
        inv_kp_ind = tf.constant(inv_kp_ind, dtype=tf.int32)

        @tf.function
        def resplit(images, keypoints, fn):
            params = tf.concat([images, keypoints], axis=0)
            inp = tf.gather(params, kp_list, axis=0)
            lab = tf.gather(params, inv_kp_ind, axis=0)
            return tf.data.Dataset.from_tensors((inp, lab, fn))


        return data.flat_map(resplit), val_data.flat_map(resplit), test_data.flat_map(resplit)

    def load_cephal(self):
        self.n_landmarks = 20 # Actually 19, workaround so we can easily do the iterative stuff...
        self.ds_size = 400
        self.orig_im_size = tf.constant([1935,2400])
        data_dir = pathlib.Path(self.path+self.name+'/images/')
        list_im = tf.data.Dataset.list_files(str(data_dir)+'*/*') # has default shuffle
        
        @tf.function
        def process_path(file_path):
            img = decode_image(file_path)
            img.set_shape([2400,1935,1]) # H,W,C
            file_name = tf.strings.split(tf.strings.split(file_path, sep='/')[-1], sep='.')[0]
            label = tf.strings.split(tf.io.read_file(self.path+self.name+'/raw/'+file_name+'.txt'),sep='\r\n')[:19]
            label = tf.map_fn(lambda x: tf.strings.split(x,sep=','),label)
            label = tf.strings.to_number(label, out_type=tf.dtypes.int32)
            w,h = tf.split(label, 2, axis=1)
            label = tf.concat([h,w], axis=1)
            return img, label, file_name

        return list_im.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        

    def load_droso(self):
        self.n_landmarks = 40
        self.ds_size = 712
        self.orig_im_size = tf.constant([3840,3234])
        data_dir = pathlib.Path(self.path+self.name+'/images/')
        list_im = tf.data.Dataset.list_files(str(data_dir)+'*/*')

        @tf.function
        def process_path(file_path):
            img = decode_image(file_path)
            img.set_shape([3234,3840,1]) # H, W, C
            file_name = tf.strings.split(tf.strings.split(file_path, sep='/')[-1], sep='.')[0]
            label = tf.strings.split(tf.io.read_file(self.path+self.name+'/raw/'+file_name+'.txt'), sep='\n')[:-1]
            label = tf.map_fn(lambda x: tf.strings.split(x, sep=' '), label)
            label = tf.strings.to_number(label, out_type=tf.dtypes.float32)
            w,h = tf.split(label,2, axis=1)
            w = tf.clip_by_value(w,0,3839)
            h = tf.clip_by_value(h,0,3233)
            label = tf.concat([h,w],axis=1) 
            return img, label, file_name
        
        return list_im.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)