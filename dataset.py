import torch
from torch.utils.data import Dataset
import os
import cv2
from utils.helper import add_padding
import numpy as np
def text_2_lists(data_path_file:str, labels_path_file:str)->list:
    """
    reads list of files for train,test or valid and returns 
    list of labels and data
    """
    with open(data_path_file) as d:
        data = d.readlines()
    with open(labels_path_file) as l:
        labels = l.readlines()
    return data,labels

class Signs(Dataset):
    def __init__(self, DATAROOT:str, split: str='train', transforms=None, model_name: str ='CNN'):
        assert split.lower() in ['train','test','valid'], "wrong split value, needss to be in [train,test,valid]"
        self.split = split
        self.DATAROOT = DATAROOT
        labels_path = os.path.join(DATAROOT,f'{self.split}_labels.list')
        data_path = os.path.join(self.DATAROOT,f'{self.split}_data.list')
        
        self.data, self.labels = text_2_lists(data_path, labels_path)
        self.transforms = transforms
        self.model_name=model_name
        self.reshape_sizes={'CNN':30,
                            'VGG':224}
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        img_path, label = f"{os.path.join(self.DATAROOT,self.data[index])}" , int(self.labels[index])  
        img = cv2.imread(img_path.strip())
        reshape= self.reshape_sizes[self.model_name]
        
        if img.shape[0]<reshape or img.shape[1]<reshape:
            img = add_padding(img,(reshape,reshape))
        else:
            img = cv2.resize(img,(reshape,reshape), interpolation = cv2.INTER_AREA)
        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        if self.model_name=='CNN':
            img= np.transpose(img, (2, 0, 1))
        img = img.astype('float')
        return torch.tensor(img, dtype=torch.float32) , label