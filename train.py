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

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "inlinebackend.figure_format = 'retina'")
from workspace_utils import active_session
torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model,epochs_number,criterion,optimizer,training_loader,validation_loader,current_device):
    model.to(current_device)
    epochs= epochs_number
    print_ev=1
    running_loss= 0 
    steps= 0
    
    for epoch in range(epochs):
        model.train()
        
        for inputs,labels in training_loader:
            steps=steps+1
            
            inputs,labels=inputs.to(current_device),labels.to(current_device)
            optimizer.zero_grad()
            
            log_ps=model(inputs)
            loss=criterion (log_ps,labels)
            loss.backward()
            optimizer.step()
            
            running_loss=running_loss+loss.item()
            
        if steps % print_ev==0:
            model.eval()
            with torch.no_grad():
                test_loss,accuracy=testClassfier(model,criterion,validation_loader,current_device)
                
            train_loss=running_loss/print_ev
            valid_loss=test_loss/len(validation_loader)
            valid_accuracy=accuracy/len(validation_loader)
            
            running_loss=0
            model.train()
            
    return train_loss,valid_loss   

def main():
    args = parse_args()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.RandomRotation(45),transforms.RandomVerticalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    testing_transforms = transforms.Compose([transforms.CenterCrop(224),transforms.Resize(255),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    data_transforms = {"training": training_transforms,"validation": validation_transforms,"testing": testing_transforms}
    
    train_data = datasets.ImageFolder(train_dir,transform=training_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=testing_tranforms)
    image_datasets = {"training": train_data,"validation": valid_data,"testing": test_data}

                                                                
    training_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_data,batch_size=64,shuffle=True)
    testingloader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)

    dataloaders = {"training": training_loader,"validation": validation_loader, "testing": testingloader}
    
    model = models.vgg16(pretrained=True)
    model.name= 'vgg16'
    
    for param in model.parameters():
    param.requires_grad = False
    
class Classfier(nn.Module):
    
    def __init__(self,input_size,output_size,hidden_layers,drop_out=0.2):
        super().__init__()
        self.hidden_layers= nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
        hlayers = zip(hidden_layers[:-1],hidden_layers[1:])
        self.hidden_layers.extends([nn.Linear(hinput,houtput) for hinput,houtput in hlayers])
        self.output = nn.Linear(hidden_layers[-1],output_size)
        self.dropout = nn.Dropout(p=drop_out)
        
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        
        for layer in self.hidden_layers:
            x= self.dropout(F.relu(layer(x)))
            
        x= F.log_softmax(self.output(x),dim=1)
        return x
    
    
hidden_layers= [4096,1024]
input_size= 25088
output_size= 102
drop_out = 0.2

model.classfier = Classfier(input_size,output_size,hidden_layers,drop_out)



def saveCheckPoint(model):
    model.class_to_idx=train_data.class_to_idx
    checkpoint = {'input_size':1024,
                  'output_size':102,
                  'name': model.name,
                  'class_to_idx': model.class_to_idx,
                  'classfier': model.classfier,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint,'checkpoint.pth')
    
    
model.classfier = Classfier(input_size,output_size,hidden_layers,drop_out)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
model.to(current_device)
hidden_layers= [4096,1024]
input_size= 25088
output_size= 102
drop_out = 0.2
optimizer=optim.Adam(model.classifier,parameters(),lr=learning_rate)

with active_session():
    train_loss,valid_loss,valid_accuracy= trainClassfier(model,epochs_number,criterion,optimizer,training_loader,validation_loader,current_device)
    
print("Finalized result \n",
      f"Training loss: {train_loss:.3f}.. \n",
      f"Testing loss: {valid_loss:.3f}.. \n",
      f"Testing accuracy: {valid_accuracy:.3f}")

filename=saveCheckpoint(model)

if __name__ == "__main__":
    main()