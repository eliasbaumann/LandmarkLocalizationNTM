import tensorflow as tf
import pathlib

PATH = 'C:/Users/Elias/Desktop/Landmark_Datasets/'

def decode_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


class Data_Loader():
    def __init__(self, name):
        self.name = name
        

    def __call__(self):
        if self.name == 'droso':
            self.load_droso()
        elif self.name == 'cephal':
            self.load_cephal()

    def load_cephal(self):
        data_dir = pathlib.Path(PATH+self.name+'/RawImage/')
        list_im = tf.data.Dataset.list_files(str(data_dir)+'*/*')
        
        def process_path(file_path):
            img = decode_image(file_path)
            file_name = tf.strings.split(tf.strings.split(file_path, sep='\\')[-1], sep='.')[0]
            label = tf.strings.split(tf.io.read_file(PATH+self.name+'/400_senior/'+file_name+'.txt'),sep='\r\n')[:19]
            label = tf.map_fn(lambda x: tf.strings.split(x,sep=','),label)
            label = tf.strings.to_number(label,out_type=tf.dtypes.int32)
            return img, label

        
        self.data = list_im.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        
    def load_droso(self):
        #TODO redo when we have droso data
        pass
    
if __name__ == "__main__":
    asdf = Data_Loader('cephal')
    asdf()