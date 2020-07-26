import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import seaborn as sns
import argparse
import json

# Imports functions created for this program
from get_input_args_train import get_input_args

def main():

    # Function to retrieve command line arguments entered by the user
    in_arg = get_input_args()
    
    # Function that checks command line arguments using in_arg  
    #check_command_line_arguments(in_arg)
    
    ## Load data
    image_datasets, dataloaders, cat_to_name = load_data(in_arg.data_dir)
        
    # Load model
    model = load_model(in_arg.arch, in_arg.hidden_units, in_arg.learn_r)

    # need better place to store this code
    crit = nn.NLLLoss()
    opt = optim.Adam(model.classifier.parameters(), lr=in_arg.learn_r)
    
    # Train the model
    train(model, in_arg.epochs, in_arg.learn_r, crit, opt, dataloaders['train_g'])
#    train(model, in_arg.epochs, in_arg.learning_rate, criterion, optimizer, train_loader, validation_loader)

    
    # Save checkpoint
    save_checkpoint(in_arg.save_dir, model, opt, in_arg.epochs, in_arg.arch, image_datasets)

 #################################
## Load transformed data set
def load_data(data_dir):
    """Load data"""

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets    
#    for directory in [train_dir, valid_dir, test_dir]:
#        if not os.path.isdir(directory):
#            raise IOError("Directory " + directory + " does not exist")
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train_g': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ]),
        'valid_g': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test_g': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    #Load the datasets with ImageFolder
    image_datasets = {
        'train_g' : datasets.ImageFolder(train_dir, transform=data_transforms['train_g']),
        'valid_g' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid_g']),
        'test_g' : datasets.ImageFolder(test_dir, transform=data_transforms['test_g'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {    
        'train_g' : torch.utils.data.DataLoader(image_datasets['train_g'], batch_size=64, shuffle=True),
        'valid_g' : torch.utils.data.DataLoader(image_datasets['valid_g'], batch_size=32, shuffle=True),
        'test_g' : torch.utils.data.DataLoader(image_datasets['test_g'], batch_size=32, shuffle=False)
    }
    
    # Save mapping of classes to indices from the training datasets
    #class_to_idx = image_datasets['training'].class_to_idx #return class ID's present in training dataset
    #print('Steve22')
    #print(class_to_idx)
    
    ##load in a mapping from category label to category name
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return image_datasets, dataloaders, cat_to_name

###########################  
## Load model
def load_model (arch, hidden_units, learn_r):
#    if arch == 'vgg19': #setting model based on vgg19
    model = models.vgg19(pretrained=True)

        #don't update weights of the pre-trained model
    for am in model.parameters():
        am.requires_grad = False 

        # update classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units,  bias=True)), #first layer
        ('Relu1', nn.ReLU()), #apply activation function
        ('Dropout1', nn.Dropout(p = 0.3)),
        ('fc2', nn.Linear(hidden_units, 102, bias=True)), #output layer
        ('output', nn.LogSoftmax(dim=1)) #apply loss function
    ]))  
      
    # replace classifier in pre-trained model with the updated version
    model.classifier = classifier
        
    # Set criterion and optimizer
    crit = nn.NLLLoss()
    opt = optim.Adam(model.classifier.parameters(), lr=learn_r)           
    
    print(crit)
    return model
###########################
##Train Model
def train(model, epochs, learn_r, crit, opt, tr_lo):

    model.train() # Puts model into training mode
    print_every = 40
    steps = 0
    use_gpu = False
    start = time.time()
    
    # Check to see if GPU is available
    if torch.cuda.is_available():
        use_gpu = True
        model.cuda()

    else:
        model.cpu()
        
    print('Model: ', model)
    
    # Iterate through each training pass based on number of epochs & GPU availability
    for h in range(epochs):
        running_loss = 0
        for im, lb in iter(tr_lo):
            steps += 1

            if use_gpu:
                im = Variable(im.float().cuda())
                lb = Variable(lb.long().cuda()) 
            else:
                im = Variable(im)
                lb = Variable(lb) 

            opt.zero_grad() # Clear the gradients to ensure they aren't accumulated
     
            # Forward and backward passes
            output = model.forward(im) # Forward propogation 
            loss = crit(output, lb) # Calculates loss
            loss.backward() # Calculates gradient
            opt.step() # Updates weights based on gradient & learning rate

            running_loss += loss.item()

# To delete - no need to validate model as already done in pytorch version
#            if steps % print_every == 0:
#                validation_loss, accuracy = validate(model, criterion, dataloaders['validation']) # Call validate function
#
#                 print("Epoch: {}/{} ".format(epoch+1, epochs),
#                        "Training Loss: {:.3f} ".format(running_loss/print_every),
#                        "Validation Loss: {:.3f} ".format(validation_loss),
#                        "Validation Accuracy: {:.3f}".format(accuracy))
                
    # Confirm time to train model
    print(f"Time to train model: {(time.time() - start)/3:.3f} seconds")

##############################################   
def save_checkpoint(save_dir, model, opt, epochs, arch, image_datasets):
    ## Save the checkpoint
    model.cpu()

    checkpoint = {
        'epochs': epochs, #Number of epochs
        'arch': model, #architecture used
        'state_dict': model.state_dict(), #learned paramaters from training
        'class_to_idx': image_datasets['train_g'].class_to_idx, #mapping of flower class values to flower indices
        'optimizer_state_dict': opt.state_dict(),
        'classifier': model.classifier} #Latest checkpoint to continue training later    
                
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print('Checkpoint saved!')
                
    return None

###########################
    
if __name__ == "__main__":
    main()
