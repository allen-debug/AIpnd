"""import files"""

from PIL import Image
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
import torch
import numpy as np
from torchvision import datasets, transforms, models
import argparse


"""def arc"""
alexnet = models.alexnet(pretrained=True)
"""def arc"""
vgg16 = models.vgg16(pretrained=True)
"""def arc"""
densenet = models.densenet121(pretrained=True)
"""def arc"""

models = {'alexnet': alexnet, 'vgg': vgg16,'densenet': densenet}
"""def models"""
models_nodes = {'alexnet': 9216, 'vgg': 25088,'densenet': 1024}
"""def nodes"""

"""def main.... continue tonight"""

def args_get():
    """def parser"""
     
    parser = argparse.ArgumentParser()
    
    
    """def parser"""
    parser.add_argument("dir", type = str, default = 'flowers/',
                        help = 'path to the folter of flowers')
    """def parser"""
    parser.add_argument('--arch', type = str, default = 'densenet',
                        help = 'CNN Model Architecture')
    """def parser"""
    parser.add_argument('--learning_rate', type = float, default = 0.003,
                        help = 'The learning rate of the algorithm')
    """def parser"""
    parser.add_argument('--hidden_units', type = int, default = 512,
                        help = 'Number of hidden units')
    """def parser"""                    
    parser.add_argument('--gpu', action='store_true',
                            help = 'Use GPU to train')
    """def parser"""
    parser.add_argument('--epochs', type = int, default = 3,
                        help = 'Number of epocs of the training')
    """def parser"""
    parser.add_argument('--save_dir', type = str, default = 'trained',
                        help = 'Ouput directory of save models')
    """def parser"""
    parser.add_argument('--save_file', type = str, default = 'checkpoint.pth',
                        help = 'File name of training model')
    
    return parser.parse_args()

"""def mn"""

def main():
    """def args"""
    in_arg = args_get()
    
    print(in_arg)
    
    data_dir = in_arg.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    """data transforms"""
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    """test transforms"""
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    """img db"""
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    """tst db"""
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    """train loader"""
    trainloader = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    """test loader"""
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    """def device"""
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    """def model"""
    model = models[in_arg.arch]
    """def nodes"""
    incomming_nodes = models_nodes[in_arg.arch]
    
   
    for am in model.parameters():
        """def am"""
        
        am.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    """def opt"""
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    model.to(device);        
    
    """def ophs"""
            
    epochs = in_arg.epochs
    stp = 0
    """running loss"""
    running_loss = 0
    pr_ev = 5

    for ch in range(epochs):
        """training"""
        for ip, lb in trainloader:
            
            """train loader"""
            stp += 1
        
            ip, lb = ip.to(device), lb.to(device)
        
        
            optimizer.zero_grad()
            """opt """
        
            logps = model.forward(ip)
            loss = criterion(logps, lb)
            loss.backward()
            optimizer.step()
        
            """running loss"""

            running_loss += loss.item()
        
            if stp % pr_ev == 0:
                """train loader"""
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    """train loader"""
                    for ip, lb in testloader:
                        """test loader"""
                        ip, lb = ip.to(device), lb.to(device)
                        logps = model.forward(ip)
                        batch_loss = criterion(logps,lb)
                    
                        test_loss += batch_loss.item()
                    
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == lb.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    """Phase 1 over"""
                    
                print(f"Epoch {ch+1}/{epochs}.. "
                    f"Train loss: {running_loss/pr_ev:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    
    
    dir = in_arg.save_dir
    if not os.path.exists(dir):
        """def os"""
        os.mkdir(dir)        
         
    checkpoint = {'classifier': model.classifier,
              'class_to_idx': image_datasets.class_to_idx,
              'state_dict': model.state_dict(),
              'model_name' : in_arg.arch}

    torch.save(checkpoint, in_arg.save_dir + "/" + in_arg.save_file)
    

"""def main func"""    
################################-----------------

main()
