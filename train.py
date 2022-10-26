import logging
import torch
from dataset import Signs
from utils.augmentation import Augmentation_albumentation
from random import randint
import logging
from utils.helper import calculate_nb_classes, save_weights, read_file_output_list, parser, read_checkpoint, plots
from importlib import import_module
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import os
import pandas as pd

def train():
    train_loss = 0
    train_correct = 0
    counter = 0
    epoch_loss = 0
    epoch_acc = 0
    net.train()
    
    for _, (input, label) in tqdm(enumerate(train_dataloader)):
        counter+=1
        input = input.cuda()
        label = label.cuda()
        pred = net(input)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        _,out_pred = torch.max(pred.data, 1)
        loss.backward()
        train_loss+= loss.item()*input.size(0)
        train_correct+= (out_pred==label).sum().item() 
        optimizer.step()
    epoch_loss = train_loss/counter
    epoch_acc = train_correct/len(train_dataloader.dataset)*100
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, epochs, loss.item()))
    print('Epoch [{}/{}], Acc: {:.4f}'.format(epoch, epochs, epoch_acc))

    if epoch%save_epoch==0 and epoch!=0:
        save_weights(epoch, net, optimizer, loss, os.path.join(log_file,f"model_{epoch}.pth"))
    return epoch_loss, epoch_acc
    
            
def valid():
    net.eval()
    epoch_val_loss=0
    epoch_val_acc=0
    val_loss = 0
    val_correct = 0
    counter = 0
    with torch.no_grad():
        for _, (input, label) in tqdm(enumerate(valid_dataloader)):
            counter+=1
            input = input.cuda()
            label = label.cuda()
            out = net(input)
            _, pred = torch.max(out.data,1)
            loss = criterion(out, label)
            val_loss+= loss.item()*input.size(0)
            val_correct += (pred==label).sum().item()

        epoch_val_loss = val_loss/counter
        epoch_val_acc = val_correct/len(valid_dataloader.dataset)*100
        print(f"epoch_loss={epoch_val_loss}, epoch_validation_acc={epoch_val_acc}, lr={lr}")

        return epoch_val_loss, epoch_val_acc
        
if __name__=="__main__":
    
    start_epoch = 0
    args= parser()
    cfg = calculate_nb_classes(args.cfg, args.t)
    
    dataroot= cfg['data_path']  
    DATA_NAME = cfg['data_name']
    NB_CLASSES = cfg['nb_classes']
    MODEL = cfg['model']
    log_file = cfg['log_file']
    load_weights = cfg['load_weights']
    batch_size = cfg['batch_size']
    lr = cfg['lr']
    epochs = cfg['epochs']
    weight_decay = cfg['weight_decay']
    save_epoch = cfg['save_epoch']
    assert NB_CLASSES>0, "Categories file is empty"
    assert MODEL!='', "model name cannot be empty"
    assert log_file!='', "please enter a log folder"
    FIG_PATH = os.path.join(log_file,"acc_loss.jpg")
    logging.info("Model used:",MODEL)
    
    M = randint(0,9)
    trans, ops = Augmentation_albumentation(2, M, .6)
    train_dataset = Signs(dataroot,transforms=trans)
    valid_dataset  = Signs(dataroot,transforms=trans,split="valid")

    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size,
                                    shuffle= True)
    valid_dataloader  = torch.utils.data.DataLoader(dataset = valid_dataset,  batch_size=batch_size,
                                    shuffle= True)
    
    net = import_module(f".{MODEL}","models").Model(NB_CLASSES)
    
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-07)
    criterion = nn.CrossEntropyLoss()
    
    if load_weights!='':
        start_epoch, loss = read_checkpoint(load_weights, net, optimizer)
        df = pd.read_csv(os.path.join(log_file,"loss_acc.csv")).head(start_epoch+1)
        running_val_loss, running_val_acc= list(df['validation_loss']),list(df['validation accuracy'])
        running_loss, running_acc= list(df['train_loss']), list(df['train_accuracy'])
        start_epoch+=1
        logging.info(f'{load_weights} loaded')
        logging.info(f'start training at {start_epoch}')
    else:
        running_val_loss, running_val_acc, running_loss, running_acc =[],[],[],[]
    
    for epoch in tqdm(range(start_epoch, epochs+1)):
        epoch_loss, epoch_acc = train()
        epoch_val_loss, epoch_val_acc = valid()
    
        running_val_loss.append(epoch_val_loss)
        running_val_acc.append(epoch_val_acc)
        running_loss.append(epoch_loss)
        running_acc.append(epoch_acc)
        plots(epoch, FIG_PATH,running_loss,
          running_val_loss, running_acc,running_val_acc)

        df = pd.DataFrame(list(zip(running_loss,running_acc,running_val_loss,running_val_acc)),
                    columns=['train_loss','train_accuracy', 'validation_loss', 'validation accuracy'])    
        df.to_csv(os.path.join(log_file, "loss_acc.csv"))