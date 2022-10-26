import os
from sklearn.model_selection import train_test_split
import pandas as pd

class Preperation:
    def __init__(self, data_path:str, save_path: str='data', split: str='train'):
        
        self.data_path = data_path
        self.split = split
        self.save_path = save_path
        
    def create_text_files(self, list_of_images:list, labels:list, split):
        """
        create text file that contains the lists 
        """
        path = os.path.join(self.save_path,f"{split}_data.list")
        label_path = os.path.join(self.save_path,f"{split}_labels.list")

        if os.path.exists(path):
            os.remove(path)
            
        if os.path.exists(label_path):
            os.remove(label_path)
            
        with open(path,'a') as f:
            f.write("\n".join(list_of_images))
        
        with open(label_path,'a') as f:
            f.write("\n".join(labels))
            
class Chinese_signs_data_preperation(Preperation):
    def __init__(self, data_path:str, save_path: str='data', split: str='train'):
        assert split in ['train','valid']
        super().__init__(data_path, save_path, split)
        self.split_text()
        
    def split_data(self, list_of_images:list, labels:list):
        """
        split data into train, test and valid if test file doesnt alread exist
        else split traininto train and valid
        """
        train_data, test_data, train_labels, test_labels = train_test_split(list_of_images, labels, test_size=0.1) 
        
        return train_data, test_data, train_labels, test_labels
                
    def split_text(self):
        """
        creates list of images and list of labels from a specififc format and
        outputs list files
        """
        with open(self.data_path) as f:
            imgs_file = f.readlines()
        data = [os.path.join(self.split,img.split(';')[0]) for img in imgs_file]
        labels = [img.split(';')[-2] for img in imgs_file]
        if self.split=='train':
            data, test_data, labels, test_labels = self.split_data(data, labels)
            self.create_text_files(test_data, test_labels,'test')
        self.create_text_files(data, labels, self.split)
        
        
class German_signs_data_preperation(Preperation):
    def __init__(self, data_path:str, save_path: str='data', split: str='train'):
        super().__init__(data_path, save_path, split)
        self.split_text()
    
    
    def split_text(self):
        names = {'train':'Train.csv',
             'test':'Meta.csv',
             'valid':'Test.csv'}
        csv = pd.read_csv(os.path.join(self.data_path,names[self.split]))
        list_images = csv['Path'].apply(lambda x: x.strip())
        labels = csv['ClassId'].apply(lambda x: str(x))
        self.create_text_files(list_images, labels, self.split)