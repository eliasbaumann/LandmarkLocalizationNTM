import pathlib
import tensorflow as tf
import albumentations as albu
import numpy as np
from heatmapgen import Heatmap_Generator
import matplotlib.pyplot as plt

PATH = 'C:/Users/Elias/Desktop/Landmark_Datasets/'

@tf.function
def decode_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, dtype= tf.dtypes.float32)
    img = tf.image.per_image_standardization(img)
    # img = tf.subtract(tf.scalar_mul(tf.constant(2., dtype=tf.float32), img),tf.constant(1., dtype=tf.float32))
    return img

class Data_Loader():
    def __init__(self, name, batch_size, train_pct=80, val_pct=10, test_pct=10, repeat=True, prefetch=True, n_aug_rounds=5):
        self.name = name
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.repeat = repeat
        self.prefetch = prefetch
        self.n_aug_rounds = n_aug_rounds
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.n_landmarks = None
        self.ds_size = None
        self.keypoints = None
        self.augmentations = []
        

    def __call__(self, im_size=None, keypoints=None):
        print("Creating Datasets...")
        self.keypoints = keypoints

        data = self.load_droso()
            
        if self.name == 'cephal':
            data = self.load_cephal()
        
        data = self.resize_images(data, im_size)
        # data = data.shuffle(buffer_size=self.ds_size) #TODO add this for actual runs, 
        
        # train test val split (take and skip)
        n_train_obs = int(self.ds_size * (self.train_pct/100.0) - (self.ds_size * (self.train_pct/100.0) % self.batch_size))
        n_val_obs = int(self.ds_size * (self.val_pct/100.0) - (self.ds_size * (self.val_pct/100.0) % self.batch_size))
        n_test_obs = int(self.ds_size * (self.test_pct/100.0) - (self.ds_size * (self.test_pct/100.0) % self.batch_size))
       
        self.train_data = data.take(n_train_obs)
        self.val_data = data.skip(n_train_obs).take(n_val_obs)
        self.test_data = data.skip(n_train_obs+n_val_obs).take(n_test_obs)
       
        self.train_data = self.augment_data(self.train_data, im_size)

        if self.keypoints is not None:
            self.train_data, self.val_data, self.test_dataself = self.kp_to_input(self.train_data, self.val_data, self.test_data, self.keypoints)
        
        if self.repeat:
            self.train_data = self.train_data.repeat()
            self.val_data = self.val_data.repeat()
            self.test_data = self.test_data.repeat()
        
        self.train_data = self.train_data.batch(self.batch_size, drop_remainder = True)
        self.val_data = self.val_data.batch(self.batch_size, drop_remainder=True)
        self.test_data = self.test_data.batch(self.batch_size, drop_remainder=True)

        
        if self.prefetch:
            self.train_data = self.train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            self.val_data = self.val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print("Datasets loaded")

    def convert_to_hm(self, keypoints, im_size):
        heatmaps = Heatmap_Generator(im_size,self.n_landmarks, 3).generate_heatmaps(keypoints) #TODO parameterize
        return heatmaps

    def resize_images(self, data, im_size):   
        def _albu_resize(image, keypoints):
            transformed = albu.Compose([albu.Resize(im_size[0], im_size[1])], 
                                       p=1, keypoint_params=albu.KeypointParams(format='xy',remove_invisible=True))(image=image, keypoints=keypoints)
            image = np.array(transformed['image'],dtype=np.float32)
            keypoints = np.array(transformed['keypoints'],dtype=np.float32)
            if(len(image.shape)<3):
                image = np.expand_dims(image,axis=0)
            elif(image.shape[0]!=1): 
                image = np.expand_dims(np.squeeze(image),axis=0) # image = np.reshape(image,(1,image.shape[0],image.shape[1]))
            return image, keypoints
        
        def _resize(img, lab, fn):
            image, keypoints = tf.numpy_function(_albu_resize, [img, lab], [tf.float32, tf.float32])
            image.set_shape([1,im_size[0],im_size[1]])
            keypoints = self.convert_to_hm(keypoints, im_size)
            keypoints.set_shape([40,im_size[0],im_size[1]])
            return image, keypoints, fn
        
        def convert_all(images, keypoints, filename):
            return tf.data.Dataset.from_tensors((images, keypoints, filename)).map(_resize)

        return data.flat_map(convert_all)

    def augment_data(self, data, im_size):
        def _albu_transform(image, keypoints):
            image = np.stack((image,keypoints),axis=0)
            transformed = albu.Compose([albu.ElasticTransform(p=.5),
                                        albu.Flip(p=.5),
                                        albu.RandomSizedCrop((int(.5*im_size[0]),int(.9*im_size[0])),im_size[0],
                                                              im_size[1],w2h_ratio=float(im_size[1])/float(im_size[0]),p=.5),
                                        albu.ShiftScaleRotate(shift_limit=.1,scale_limit=.1,rotate_limit=90,p=.5)],
                                       p=1)(image=image)
            image = np.array(transformed['image'][0:1],dtype=np.float32)
            keypoints = np.array(transformed['image'][1:],dtype=np.float32)
            #keypoints = np.array(transformed['keypoints'],dtype=np.float32)
            if(len(image.shape)<3):
                image = np.expand_dims(image,axis=0)
            elif(image.shape[0]!=1): #TODO feels wrong
                image = np.expand_dims(np.squeeze(image),axis=0)
            return image, keypoints
          

        def _augment(img, lab, fn):
            image, keypoints = tf.numpy_function(_albu_transform, [img, lab], [tf.float32, tf.float32])
            image.set_shape([1,im_size[0],im_size[1]])
            keypoints = self.convert_to_hm(keypoints, im_size)
            keypoints.set_shape([40,im_size[0],im_size[1]])
            return image, keypoints, fn


        def generate_augmentations(images, keypoints, filename):
            regular_ds = tf.data.Dataset.from_tensors((images, keypoints, filename))
            for _ in range(self.n_aug_rounds): 
                aug_ds = tf.data.Dataset.from_tensors((images, keypoints, filename)).map(_augment)
                regular_ds.concatenate(aug_ds)
            return regular_ds
        
        return data.flat_map(generate_augmentations)

    
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
        self.n_landmarks = 19
        data_dir = pathlib.Path(PATH+self.name+'/RawImage/')
        list_im = tf.data.Dataset.list_files(str(data_dir)+'*/*')
        
        @tf.function
        def process_path(file_path):
            img = decode_image(file_path)
            file_name = tf.strings.split(tf.strings.split(file_path, sep='\\')[-1], sep='.')[0]
            label = tf.strings.split(tf.io.read_file(PATH+self.name+'/400_senior/'+file_name+'.txt'),sep='\r\n')[:self.n_landmarks]
            label = tf.map_fn(lambda x: tf.strings.split(x,sep=','),label)
            label = tf.strings.to_number(label, out_type=tf.dtypes.int32)
            return img, label

        return list_im.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        

    def load_droso(self):
        self.n_landmarks = 40
        self.ds_size = 471
        data_dir = pathlib.Path(PATH+self.name+'/images/')
        list_im = tf.data.Dataset.list_files(str(data_dir)+'*.jpg')

        @tf.function
        def process_path(file_path):
            img = decode_image(file_path)
            file_name = tf.strings.split(tf.strings.split(file_path, sep='\\')[-1], sep='.')[0]
            label = tf.strings.split(tf.io.read_file(PATH+self.name+'/raw/'+file_name+'.txt'), sep='\n')[:-1]
            label = tf.map_fn(lambda x: tf.strings.split(x, sep=' '), label)
            label = tf.strings.to_number(label, out_type=tf.dtypes.float32)
            w,h = tf.split(label,2, axis=1)
            w = tf.clip_by_value(w,0,3839)
            h = tf.clip_by_value(h,0,3233)
            label = tf.concat([w,h],axis=1)
            return img, label, file_name
        
        return list_im.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        #self.im_size = decode_image(str(PATH+self.name+'/images/NG-SP196-909-0001.jpg')).shape
