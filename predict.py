#Programmer : Allen
#Date: 2 AUG 2020
import torch
import json
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import argparse
from torch.autograd import Variable




parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='/home/workspace/ImageClassifier/flowers/test/34/image_06961.jpg', help='test image to predict')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint to save and predit')
parser.add_argument('--gpu', type = bool, default=False, help='use gpu if available')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='map json to the name of the flower')
parser.add_argument('--topk', type=int, default=5, help='To output the topk probability')
args = parser.parse_args()

if args.gpu==True:
    device = 'gpu'
else:    
    device='cpu'
    

    


def chk_load(filepath):
    checkpoint = torch.load(filepath)
    model= checkpoint['arck']
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False
    return model
checkpoint=args.checkpoint
modelod=chk_load(checkpoint)
 

def process_image(image):
    """img pro"""
    im = Image.open(image)
   
    img_pr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    """output"""
    output = img_pr(im)
    return output

def predict(image, model, topk=3):
    image=  Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0) 
    if device=='gpu'and torch.cuda.is_available():
        image = image.cuda()
        model.to('cuda')
    result = model(image).topk(topk)
    pr = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
    lob = result[1].data.cpu().numpy()[0]
    idx_to_class = { value : key for key,value in model.class_to_idx.items()}
    topk_class = [idx_to_class[lo] for lo in lob]  
    return pr, topk_class
image = args.image
imagess = process_image(image)
topk=args.topk
pr, clss = predict(imagess,modelod, topk)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
fnames=[cat_to_name.get(clss[i]) for i in range(len(clss))]
for i in range(4):
    print("Flower Name: {},  Class: {} , Probability: {}".format(fnames[i],clss[i],pr[i]))
    
