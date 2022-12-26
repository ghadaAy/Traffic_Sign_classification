In this repo, we present a model for the classification of traffic signs.
The CNN model is inside the folder models.

Accuray got to 96% on validation set
Link to dataset: https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification


|                Layers              | 
| -----------------------------------| 
|    Convolutional layer 1 (3,32,5)  |
|   Convolutional layer 2 (32,32,5)  |
|         Max Pooling layer (2x2)    |
|                  Dropout           |
|    Convolutional layer 3 (32,64,3) |
|    Convolutional layer 4 (64,64,3) |
|       Max Pooling layer (2x2)      |
|                  Dropout           |
|         FC layer 1 (576,576)       |
|     FC layer 2 (576,nb_classes)    |
