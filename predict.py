import torch
from torch import nn
from torch import optim
from torch.autograd import variable
from torchvision import datasets,transforms,models
from torchvision.datasets import ImageFolder
import argparse
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
import numpy as np
import time
import os
import random
import seaborn as sns
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06764.jpg')
    parser.add_argument('--top_k', dest='top_k', default='4')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    if image.width> image.height:
        image.thumbnail((10000000,256))
    else:
        image.thumbnail((256,10000000))
        
        
    l=(image.width-224)/2
    b=(image.height-224)/2
    r=(l+224)/2
    t=(b+224)/2
    
    image=image.crop((l,b,r,t))
    
    image=np.array(image)
    image=image/255
    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229, 0.224, 0.225])
    image = ((image - mean) / std)
    image = image.transpose(2, 0, 1)
    return torch.tensor(image)

def predict(image_path, model, topk=5):
    model.eval()
    model.cpu()
    image = Image.open(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    image = image.unsqueeze(0)
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    index_for_class = {model.class_to_idx[k]: key for key in model.class_to_idx}
    top_class_labs = []
    for label in top_labs.numpy()[0]:
        labs.append(index_for_class[label])
    return top_probs, top_class_labs

def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    image_path = args.filepath
    probs, classes = predict(iamge_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File  for selection is: ' + image_path)
    print(labels)
    print(probability)
    n=0 
    while n < len(labels):
        print("{} with a probability of {}".format(labels[n], probability[n]))
        n += 1 # cycle through

if __name__ == "__main__":
    main()