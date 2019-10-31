import tensorflow as tf
import pathlib

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
        self.im_size = None
        

    def __call__(self):
        if self.name == 'droso':
            self.load_droso()
        elif self.name == 'cephal':
            self.load_cephal()
        
        self.data.shuffle(buffer_size=1000)
        if self.repeat:
            self.data.repeat()
        
        self.data.batch(self.batch_size)
        
        if self.prefetch:
            self.data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def pre_process(self):
        # Construct a pre_processing pipeline here
        pass

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
        self.im_size = decode_image(str(PATH+self.name+'/images/NG-SP196-909-0001.jpg')).shape
    