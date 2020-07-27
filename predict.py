"""import files"""

import numpy as np
from torch import nn

import json
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import torch

import os
import argparse


"""def arch"""
alexnet = models.alexnet(pretrained=True)
"""def arch"""
vgg16 = models.vgg16(pretrained=True)
"""def arch"""
densenet = models.densenet121(pretrained=True)
"""def models"""

models = {'alexnet': alexnet, 'vgg': vgg16,'densenet': densenet}

"""def main"""

def arg_get():
    """def parser"""
   
    
    parser = argparse.ArgumentParser()
    
    """def parser"""
    parser.add_argument("image_patch", type = str,
                        help = 'the image patch to test')
    """def parser"""                    
    parser.add_argument('--gpu', action='store_true',
                        help = 'Use GPU to train')
    """def parser"""
    parser.add_argument('--load_file', type = str, default = 'trained/checkpoint.pth',
                        help = 'Ouput directory of save models')
    """def parser"""
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'Ouput directory of save models')
    """def parser"""
    parser.add_argument('--top_k', type = int, default = 5,
                        help = 'The most K result classes')

    return parser.parse_args()


def main():
    """def args"""
    
    in_arg = arg_get()
    
    print(in_arg)
    """def model"""
    
    model_trained = load_checkpoint(in_arg.load_file)
    """def device"""
    
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")
   
    """def open"""
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    p_t,l_t,f_t = predict(in_arg.image_patch,model_trained,in_arg.top_k,cat_to_name,device)
    """def prediction"""
    print_prediction(p_t,l_t,f_t,cat_to_name,in_arg.image_patch)


def print_prediction(p_t,l_t,f_t,cat_to_name,image_path):
    flower_num = image_path.split('/')[2]
    title = cat_to_name[flower_num]
    """define print"""
    print("")
    """define print"""
    print("The flower  is {}".format(title))
    """define print"""
    out_str = ''
    """define print"""
    print("----------------------------")
    """define print"""
    for flower, probability in zip(f_t, p_t):
        """define print"""
        out_str += "flower name :  {}   -> probability  : {} \n".format(flower,probability)
        """def print func"""
        
    print(out_str)

    
"""def chk pnt"""    
def load_checkpoint(filepath):
    """def chkpnt"""
    checkpoint = torch.load(filepath)
    """def model"""
    model = models[checkpoint['model_name']]
    """def model"""
    model.classifier = checkpoint['classifier']
    """def model"""
    model.load_state_dict(checkpoint['state_dict'])
    """def model"""
    model.class_to_idx = checkpoint['class_to_idx']
    """rtn model"""
    
    return model

    
def img_pro(image):
    
    image = Image.open(image)
    
    if image.size[0] > image.size[1]:
        """img sz"""
        image.thumbnail((10000, 256))
    else:
        """img tnail"""
        
        image.thumbnail((256, 10000)) 
    """mar def"""    
    l_mar = (image.width-224)/2
    """mar def"""  
    b_mar = (image.height-224)/2
    """mar def"""  
    r_mar = l_mar + 224
    """mar def"""  
    t_mar = b_mar + 224
    """mar def"""  
    image = image.crop((l_mar, b_mar, r_mar,   
                      t_mar))
    """mar def"""  
    image = np.array(image)/255
    """mar def"""  
    mean = np.array([0.485, 0.456, 0.406]) 
    """mar def"""  
    std = np.array([0.229, 0.224, 0.225]) 
    """mar def"""  
    image = (image - mean)/std
    """mar def"""  
    
    
    image = image.transpose((2, 0, 1))
    """rtn stmt"""  
    
    return image

def predict(image_path, model, topk,cat_to_name, device):
    """img arr"""
    
    image_array = img_pro(image_path)
    """img arr"""
    image_array = torch.from_numpy(image_array).type(torch.FloatTensor)
    """img arr"""
    image_array.unsqueeze_(0)
    """img arr"""
    model.to(device)
    """img arr"""
    with torch.no_grad():
        """model"""
        model.eval()
        """model"""
        ps = model.forward(image_array.to(device))
        """ps"""

        ps = torch.exp(ps)
        """ps"""

        p_t, c_t =  ps.topk(topk, dim=1)
        """t p lb"""
        
        p_t = p_t.detach().cpu().numpy().tolist()[0]
        """tp p"""
        c_t = c_t.detach().cpu().numpy().tolist()[0]
        """tp cls"""
        
        
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        """idx"""
        
        l_t = [idx_to_class[one_top_class] for one_top_class in c_t]
        """tp lb"""
        
        f_t = [cat_to_name[idx_to_class[one_top_class]] for one_top_class in c_t]
        """rtn stmt"""
    
        
        return p_t,l_t,f_t
    
    
##########--------------
"""def main"""

main()    
