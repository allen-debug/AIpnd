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
from get_input_args_predict import get_input_args

def main():

    # Function to retrieve command line arguments entered by the user
    in_arg = get_input_args()
    
    #Load pre-trained model from checkpoint
    model = load_checkpoint(in_arg.checkpoint)
    #print(model)
    
    # Process Image
    processed_image = process_image(in_arg.image_path) 
    
    # Classify Prediction
    top_probs, top_labels, top_flowers = predict(in_arg.image_path, model, in_arg.category_names, in_arg.top_k)
    
###############################    
def load_checkpoint(fil_pa):   
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(fil_pa)

    model = models.vgg19(pretrained=True)
    
    for am in model.parameters():
        am.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Hyperparameters for our network
    ip_s = 25088
    hn_s = [4096, 256]
    op_s = 102
    
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                                          ('fc1', nn.Linear(ip_s, hn_s[0],  bias=True)), #first layer
                                          ('Relu1', nn.ReLU()), #apply activation function
                                          ('Dropout1', nn.Dropout(p = 0.3)),
                                          ('fc2', nn.Linear(hn_s[0], op_s, bias=True)), #output layer
                                          ('output', nn.LogSoftmax(dim=1)) #apply loss function
                                          ]))
    
    # Put the classifier on the pretrained network
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    #model.arch = checkpoint['arch'] 
            
    return model

###############################
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Use same image transformation code as earlier
    ig = Image.open(image_path)
    t_i = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    i_t = t_i(ig)
    
    return i_t

###############################

def predict(image_path, model, category_names, topk):
    #Predict the class (or classes) of an image using a trained deep learning model
    
    # TODO: Implement the code to predict the class from an image file   
    p_i = process_image(image_path)
    p_i.unsqueeze_(0)
    probs = torch.exp(model.forward(p_i))
    top_probs, top_labs = probs.topk(topk)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()

    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    print(top_flowers)
    print(top_probs)
    print(top_labels)
    print(topk)
    
    
    
    
    return top_probs, top_labels, top_flowers
###############################
    
if __name__ == "__main__":
    main()
