import logging
import torch
from dataset import Signs
import logging
from utils.helper import calculate_nb_classes, save_weights, read_file_output_list, parser, read_checkpoint, plots
from importlib import import_module
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import os

def test():
    train_loss = 0
    train_correct = 0
    counter = 0
    epoch_loss = 0
    epoch_acc = 0
    net.eval()
    
    for _, (path, input, label) in tqdm(enumerate(test_dataloader)):
        counter+=1
        input = input.cuda()
        label = label.cuda()
        pred = net(input)
        loss = criterion(pred, label)
        _,out_pred = torch.max(pred.data, 1)
        train_loss+= loss.item()*input.size(0)
        train_correct+= (out_pred==label).sum().item() 
        print(path,out_pred.item(),label.item())
    epoch_loss = train_loss/counter
    epoch_acc = train_correct/len(test_dataloader.dataset)*100
    
    return epoch_loss, epoch_acc

if __name__=="__main__":
    
    start_epoch = 0
    args= parser()
    cfg = calculate_nb_classes(args.cfg, args.train_list_path)
    
    dataroot= cfg['data_path']  
    DATA_NAME = cfg['data_name']
    NB_CLASSES = cfg['nb_classes']
    MODEL = cfg['model']
    log_file = cfg['log_file']
    load_weights = cfg['load_weights']
    
    lr = cfg['lr']
    epochs = cfg['epochs']
    weight_decay = cfg['weight_decay']
    save_epoch = cfg['save_epoch']
    
    batch_size = 1
    assert NB_CLASSES>0, "Categories file is empty"
    assert MODEL!='', "model name cannot be empty"
    assert log_file!='', "please enter a log folder"
    FIG_PATH = os.path.join(log_file,"acc_loss.jpg")
    logging.info("Model used:",MODEL)
    
    test_dataset = Signs(dataroot,transforms=None,split='test')

    test_dataloader  = torch.utils.data.DataLoader(dataset = test_dataset,  batch_size=batch_size,
                                    shuffle= True)
    
    net = import_module(f".{MODEL}","models").Model(NB_CLASSES)
    
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-07)

    start_epoch, loss = read_checkpoint(load_weights, net, optimizer)
    
    start_epoch+=1
    logging.info(f'{load_weights} loaded')
    
    for epoch in range(1):
        path, epoch_loss, epoch_acc = test()
        print("loss=",epoch_loss,"accuracy=", epoch_acc)
