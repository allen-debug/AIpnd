#Programmer : Allen
#Date: 2 AUG 2020

import torch
import time
import json
import argparse
import numpy as np
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='directory to the data ')
parser.add_argument('--gpu', type=bool, default=False, help='use gpu if available')
parser.add_argument('--arch', type=str, default='vgg16', help='choosing the architecture available')
parser.add_argument('--epochs', type=int, default=7, help='epoch number')
parser.add_argument('--batchsize', type=int, default=16, help='define the batch size')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Defining the learning rate')
parser.add_argument('--hidden_units', type=int, default=4000, help='Defining the hidden unit for layers')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='map json to the name of the flower')
parser.add_argument('--checkpoint' , type=str, default='checkpoint.pth', help='checkpoint to save and predit')

args = parser.parse_args()

data_dir = args.data_dir  


if args.arch=='vgg16':
    print("training using vgg")
    cll=models.vgg16()
    model = models.vgg16(pretrained=True)
    ip_layer=25088
elif args.arch=='densenet121':
    print("training using densenet")
    cll=models.densenet121()
    model = models.densenet121(pretrained=True)
    ip_layer=1024
else:
    print("training using alexnet")
    cll=models.alexnet()
    model = models.alexnet(pretrained=True)
    ip_layer=9216
    
  

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

dataaload = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
])
"""test transforms"""
testting = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
 ])

valiidation = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
])

loadt = datasets.ImageFolder(train_dir, transform=dataaload)
transt = datasets.ImageFolder(valid_dir, transform=testting)
datat  = datasets.ImageFolder(test_dir, transform=valiidation)

loaderr = torch.utils.data.DataLoader(loadt, batch_size=64, shuffle=True)
vloaderr = torch.utils.data.DataLoader(transt, batch_size=32)
tloaderr  = torch.utils.data.DataLoader(datat, batch_size=32)


with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
epochs = args.epochs
def training_model(model, vloaderr, print_every, criterion, optimizer,epochs, device='cpu'):
    start = time.time()
    epochs = epochs
    print_every = print_every
    running_loss=0
    stp = 0
    

    if device=='gpu' and torch.cuda.is_available():
        model.to('cuda')
        print('GPU is available')
    else:
        print('GPU not available')
        
    for ch in range(epochs):
        """training"""
        for ip, lb in loaderr :
            
            """training"""
            stp += 1
            if device=='gpu'and torch.cuda.is_available(): #
                ip, lb = ip.to('cuda'), lb.to('cuda')#
            optimizer.zero_grad()
        
            logps = model.forward(ip)
            loss = criterion(logps, lb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if stp % print_every == 0:
                print('Validation Completed')
                
                """training"""
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                
                    for ip, lb in vloaderr:
                        
                        """training"""
                        ip, lb = ip.to('cuda'), lb.to('cuda')
                        logps = model.forward(ip)
                        batch_loss = criterion(logps, lb)
                    
                        valid_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == lb.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {ch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Valid loss: {valid_loss/len(vloaderr):.3f}.. "
                    f"Valid accuracy: {(accuracy/len(vloaderr))*100:.2f} %")
                running_loss = 0
                model.train()    
    

    

for param in model.parameters():
    param.requires_grad = False
    
hlayer=args.hidden_units
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(ip_layer, hlayer)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hlayer, 630)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(630, 102)),                          
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),args.learning_rate)
if args.gpu==True:
    device = 'gpu'
else:
    device='cpu'
training_model(model, loaderr, 40, criterion, optimizer, epochs, device)

def testing_model(tloaderr):    
    stp = 0
    """acc"""
    acc = 0

    with torch.no_grad():
        for ip, lb in tloaderr:
        
            ip = ip.to('cuda')
            lb = lb.to('cuda')
        
            op = model.forward(ip)
                    
            ps = torch.exp(op)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == lb.view(*top_class.shape)
            acc += torch.mean(equals.type(torch.FloatTensor)).item()
                    
        print(f"Test accuracy: {(acc/len(tloaderr))*100:.2f} %")

testing_model(tloaderr)
checkpoint=args.checkpoint
model.class_to_idx = loadt.class_to_idx
checkpoint = {'input_size': [3, 224, 224],
              'batch_size': loaderr.batch_size,
              'output_size': 102,
              'arck': cll,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'optimizer_dict':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epoch': epochs
             }
torch.save(checkpoint, 'checkpoint.pth')

