import imageio as iio
import numpy as np
import os


class get_dataset:
    
    dataset = np.array([])
    directory_location="./flower/"
    file_list = np.array([])
    instances = 0
    
    def get_names_of_all_the_files(self):
        self.file_list = np.array(os.listdir(self.directory_location))
        return os.listdir(self.directory_location)
    
    def get_image_data(self, file):
        img = iio.imread(self.directory_location+ file) #temp
        return img
    
    def load_database(self):
        dataset = []
        
        for file in self.file_list:
            dataset.append(self.get_image_data(file))
        self.dataset = np.array(dataset)
        self.instances = self.dataset.shape[0]
        return np.array(dataset)