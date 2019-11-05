import pathlib
import tensorflow as tf
import albumentations as albu
import numpy as np

PATH = 'C:/Users/Elias/Desktop/Landmark_Datasets/'


def decode_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

class Data_Loader():
    def __init__(self, name, batch_size, repeat=True, prefetch=True):
        self.name = name
        self.batch_size = batch_size
        self.repeat = repeat
        self.prefetch = prefetch
        self.data = None
        self.n_landmarks = None
        self.augmentations = []
        

    def __call__(self):
        im_size = [512, 512] # just define a standard value
        if self.name == 'droso':
            im_size = [512, 608] # set one value to 512 and rounded the other -> images will be minimally stretched (HW)
            self.load_droso()
            
        elif self.name == 'cephal':
            im_size = [512, 413] # HW
            self.load_cephal()

        self.pre_process(im_size)

        self.data.shuffle(buffer_size=1000)
        if self.repeat:
            self.data.repeat()
        
        self.data.batch(self.batch_size)
        
        if self.prefetch:
            self.data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def pre_process(self,im_size):
        def _albu_transform(image, keypoints):
            transformed = albu.Compose([albu.Resize(im_size[0],im_size[1],always_apply=True),
                                        albu.Flip(p=.5),
                                        albu.RandomSizedCrop((int(.3*im_size[0]),int(.9*im_size[0])),im_size[0],
                                                              im_size[1],w2h_ratio=float(im_size[1])/float(im_size[0]),p=.5),
                                        albu.ShiftScaleRotate(shift_limit=.1,scale_limit=.1,rotate_limit=90,p=.5),
                                        albu.RandomBrightnessContrast(brightness_limit=.2,contrast_limit=.2,p=.5)],
                                       p=1,
                                       keypoint_params=albu.KeypointParams(format='xy'))(image=image, keypoints=keypoints)
            return np.array(transformed['image'],dtype=np.float32), np.array(transformed['keypoints'],dtype=np.float32)
        
        def _albu_resize(image, keypoints):
            transformed = albu.Compose([albu.Resize(im_size[0], im_size[1], always_apply=True)], 
                                       p=1, keypoint_params=albu.KeypointParams(format='xy'))(image=image, keypoints=keypoints)
            return np.array(transformed['image'],dtype=np.float32), np.array(transformed['keypoints'],dtype=np.float32)

        def _augment(img, lab):
            # img_dtype = img.dtype
            # img_shape = tf.shape(img)
            images, keypoints = tf.numpy_function(_albu_transform, [img, lab], [tf.float32, tf.float32])
            return images, keypoints

        def _resize(img, lab):
            images, keypoints = tf.numpy_function(_albu_resize, [img, lab], [tf.float32, tf.float32])
            return images, keypoints

        def generate_augmentations(images, keypoints):
            regular_ds = tf.data.Dataset.from_tensors((images,keypoints)).map(_resize)
            for _ in range(3): # TODO this is how many rounds of additional images
                aug_ds = tf.data.Dataset.from_tensors((images,keypoints)).map(_augment)
                regular_ds.concatenate(aug_ds)

            return regular_ds

        self.data = self.data.flat_map(generate_augmentations)

    def load_cephal(self):
        self.n_landmarks = 19
        data_dir = pathlib.Path(PATH+self.name+'/RawImage/')
        list_im = tf.data.Dataset.list_files(str(data_dir)+'*/*')
        
        def process_path(file_path):
            img = decode_image(file_path)
            file_name = tf.strings.split(tf.strings.split(file_path, sep='\\')[-1], sep='.')[0]
            label = tf.strings.split(tf.io.read_file(PATH+self.name+'/400_senior/'+file_name+'.txt'),sep='\r\n')[:self.n_landmarks]
            label = tf.map_fn(lambda x: tf.strings.split(x,sep=','),label)
            label = tf.strings.to_number(label, out_type=tf.dtypes.int32)
            return img, label

        self.data = list_im.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        

    def load_droso(self):
        self.n_landmarks = 40
        data_dir = pathlib.Path(PATH+self.name+'/images/')
        list_im = tf.data.Dataset.list_files(str(data_dir)+'*')

        def process_path(file_path):
            img = decode_image(file_path)
            file_name = tf.strings.split(tf.strings.split(file_path, sep='\\')[-1], sep='.')[0]
            label = tf.strings.split(tf.io.read_file(PATH+self.name+'/raw/'+file_name+'.txt'), sep='\n')[:-1]
            label = tf.map_fn(lambda x: tf.strings.split(x, sep=' '), label)
            label = tf.strings.to_number(label, out_type=tf.dtypes.float32)
            return img, label

        self.data = list_im.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #self.im_size = decode_image(str(PATH+self.name+'/images/NG-SP196-909-0001.jpg')).shape

