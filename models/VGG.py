import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, nb_classes) -> None:
        super().__init__()
        
        self.nb_classes = nb_classes
        self.model = nn.Sequential(
            
            nn.Conv2d(224,64,3,stride=1, padding=1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,3,stride=1, padding=1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2,1),
            
            nn.Conv2d(64,128,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2,1),
            
            nn.Conv2d(128,256,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2,1),
            
            nn.Conv2d(256,512,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2,1),
            
            nn.Conv2d(512,512,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2,1),
            nn.Flatten(1,3),
            nn.Linear(8192,4096),
            nn.Linear(4096,4096),
            nn.Linear(4096,self.nb_classes),
            nn.Softmax(dim=1))
        self.init_weights()
        
        
    def init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                
    
    def forward(self, input:torch.Tensor):
        return self.model(input)
        