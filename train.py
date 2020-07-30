import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
from torch.autograd import Variable
from collections import OrderedDict
import time
import json
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='Path to dataset ')
parser.add_argument('--gpu', type=bool, default=False, help='whether to use gpu')
parser.add_argument('--arch', type=str, default='vgg', help='architecture [available: densenet, vgg]', required=True)
parser.add_argument('--epochs', type=int, default=7, help='Number of epochs')
parser.add_argument('--batchsize', type=int, default=16, help='Size for a batch')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden_units_Layer1', type=int, default=4000, help='hidden units for fc layer 1')
parser.add_argument('--hidden_units_Layer2', type=int, default=630, help='hidden units for fc layer 2')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
parser.add_argument('--checkpoint' , type=str, default='checkpoint.pth', help='path of your saved model')
args = parser.parse_args()

data_dir = args.data_dir  
lr = args.learning_rate 
epochs = args.epochs
Layer1=args.hidden_units_Layer1
Layer2=args.hidden_units_Layer2
catfile=args.cat_to_name
checkpoint=args.checkpoint
b_size=args.batchsize

if args.arch=='vgg':
    model = models.vgg16(pretrained=True)
    inputlayer=25088
else:    
    model = models.densenet121(pretrained=True)
    inputlayer=1024
if args.gpu==True:
    device = 'gpu'
    print('GPU calculation')
else:    
    device='cpu'
    print('cpu calculation')

# Directorie
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define your transforms for the training, validation, and testing sets
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

# Label mapping
with open(catfile, 'r') as f:
    cat_to_name = json.load(f)
    
#Build and train your network

def do_deep_learning(model, vloaderr, print_every, criterion, optimizer, epochs, device='cpu'):
    start = time.time()
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    if device=='gpu' and torch.cuda.is_available():
        model.to('cuda')
        print('GPU available')
    else:
        print('GPU NOT available')
    for ch in range(epochs):
        """training"""
        for ip, lb in loaderr :
            """training"""
            stp += 1
            ip, lb = ip.to(device),lb.to(device)
            optimizer.zero_grad()
        
            logps = model.forward(ip)
            loss = criterion(logps, lb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if stp % pv == 0:
                """training"""
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                
                    for ip, lb in vloaderr:
                        """training"""
                        ip, lb = ip.to(device), lb.to(device)
                        logps = model.forward(ip)
                        batch_loss = criterion(logps, lb)
                    
                        valid_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == lb.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {ch+1}/{epochs}.. "
                    f"Train loss: {running_loss/pv:.3f}.. "
                    f"Valid loss: {valid_loss/len(vloaderr):.3f}.. "
                    f"Valid accuracy: {(accuracy/len(vloaderr))*100:.2f} %")
                running_loss = 0
                model.train()    

    
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inputlayer, Layer1)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(Layer1, Layer2)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(Layer2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
do_deep_learning(model, trainloader, 40, criterion, optimizer, epochs, device)
# validation on the test set
def check_accuracy_on_test(tloaderr):    
    stp = 0
    """acc"""
    acc = 0

    with torch.no_grad():
        for ip, lb in tloaderr:
        
            ip = ip.to(device)
            lb = lb.to(device)
        
            op = model.forward(ip)
                    
            ps = torch.exp(op)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == lb.view(*top_class.shape)
            acc += torch.mean(equals.type(torch.FloatTensor)).item()
                    
        print(f"Test accuracy: {(acc/len(tloaderr))*100:.2f} %")

check_accuracy_on_test(tloaderr)

model.class_to_idx = loadt.class_to_idx

checkpoint = {'input_size': [3, 224, 224],
              'batch_size': trainloader.batch_size,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'optimizer_dict':optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epoch': epochs
             }
torch.save(checkpoint, 'checkpoint.pth')
