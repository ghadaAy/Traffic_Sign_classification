
import yaml
import argparse
import torch
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os

def read_file_output_list(path_to_file:str)->list:
    with open(path_to_file,'r') as c:
        return c.read().splitlines()
    
def read_yaml(yaml_f):
    with open(yaml_f,'r') as y:
        return yaml.safe_load(y)

def write_yaml(yaml_dict: dict, yaml_f):
    with open(yaml_f,'w') as y:
        return yaml.dump(yaml_dict, y)
    
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",type=str, required=True)
    parser.add_argument("--train_list_path", "-t",type=str, required=True)
    args = parser.parse_args()
    return args

def read_checkpoint(load_weights:str, net, optimizer):
    checkpoints = torch.load(load_weights)
    net.load_state_dict(checkpoints['model_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    start_epoch = checkpoints['epoch']
    loss = checkpoints['loss']
    return start_epoch, loss

def save_weights(epoch, model, optimizer, loss, PATH):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, PATH)
    
def calculate_nb_classes(cfg_file: str, path_to_train_list:str):
    """
    update cfg file with nb of classes
    """
    with open(path_to_train_list) as f:
        list_cats = f.readlines()
    set_cats = set([int(x) for x in list_cats])
    nb_cats = max(set_cats)+1
    yaml_cfg = read_yaml(cfg_file)
    yaml_cfg['nb_classes'] = nb_cats
    
    write_yaml(yaml_cfg, cfg_file)
    logging.info('Config file number of classes updated')
    return yaml_cfg


def plots(epoch:int, path:str, running_loss,
          running_val_loss, running_acc,running_val_acc):
    plt.figure(figsize=(12,8))
    x = range(epoch+1)
    plt.plot(x, running_loss,color='green', marker='o', label = "train loss")
    plt.plot(x, running_val_loss, color='magenta', marker='o', label = "validation loss")
    plt.plot(x, running_acc,color='blue', marker='o', label = "train accuracy")
    plt.plot(x, running_val_acc, color='red', marker='o', label = "validation accuracy")
    
    plt.legend()
    plt.savefig(path)
    plt.close()
    
def add_padding(img:np.array, new_shape=(32,32)):
    
    new_h, new_w = new_shape
    old_h, old_w, channels = img.shape
    if old_h>new_h:
        img = cv2.resize(img,(old_w,new_h), interpolation = cv2.INTER_AREA)
    elif old_w>new_w:
        img = cv2.resize(img,(new_w,old_h), interpolation = cv2.INTER_AREA)
    old_h, old_w, channels = img.shape
    color = (255,255,255)
    # compute center offset
    x_center = (new_w - old_w) 
    y_center = (new_h - old_h) 
    new_img = np.full((new_h,new_w, channels), color, dtype=np.uint8)
    new_img[:img.shape[0],:img.shape[1]] = img

    return new_img


   