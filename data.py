import tensorflow as tf
import numpy as np
import pathlib

PATH = 'C:/Users/Elias/Desktop/Landmark_Datasets/'

DROSO_DEFAULTS = np.concatenate(([tf.string]*7, [tf.float32]*98), axis=0)

def decode_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img,tf.float32)
    return img

def get_label(file_path):
    names = tf.strings.split(file_path, '\\')
    return names[-1]

def process_path(file_path):
    label = get_label(file_path)
    img = decode_image(file_path)
    return img,label

class Data_Loader():
    def __init__(self, name):
        self.name = name
        

    def __call__(self):
        if self.name == 'droso':
            self.load_droso()

    def load_droso(self):
        
        data_dir = pathlib.Path(PATH+self.name+'/20X_jpg/')
        label_dir = pathlib.Path(PATH+self.name+'/Leica_2X_coords.tsv')
        
        label_data = tf.data.experimental.CsvDataset(str(label_dir),DROSO_DEFAULTS,header=True,field_delim='\t')
        lookup = tf.concat([label_data[:8]],axis=1)

        list_ds = tf.data.Dataset.list_files(str(data_dir)+'*')
        bigboy_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for img, label in bigboy_ds.take(1):
            print(img.numpy().shape)
            print(label.numpy())

        for row in lookup.take(1):
            print(row)        

if __name__ == "__main__":
    asdf = Data_Loader('droso')
    asdf()