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

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "inlinebackend.figure_format = 'retina'")
from workspace_utils import active_session
torch.cuda.is_available()

def parse_args():
    print("arguments being parsed")
    parser = argparse.ArgumentParser(description="Program Training")
    parser.add_argument('--data_dir', action='store', default='./flowers/')
    parser.add_argument('--arch', dest='arch', default='vgg16')
    parser.add_argument('--learning_rate', dest='learning_rate',type=float, default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units',nargs='+',type=list, default='512')
    parser.add_argument('--epochs', dest='epochs',type=float,default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model,epochs_number,criterion,optimizer,training_loader,validation_loader,current_device):
    model.to(current_device)
    epochs= epochs_number
    print_ev=1
    running_loss= 0 
    steps= 0
    print("inside train")
    for epoch in range(epochs):
        model.train()
        
        for inputs,labels in training_loader:
            steps=steps+1
            print(steps)
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
                    test_loss,accuracy=testClassifier(model,criterion,validation_loader,current_device)
                
                train_loss=running_loss/print_ev
                valid_loss=test_loss/len(validation_loader)
                valid_accuracy=accuracy/len(validation_loader)            
                running_loss=0
                model.train()            
    return train_loss,valid_loss,valid_accuracy


def testClassifier(model, criterion, validation_loader, current_device):
    model.to(current_device)
    accuracy = 0
    test_loss = 0
    print("inside testClassifier")
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(current_device), labels.to(current_device)
        log_ps = model.forward(inputs)
        batch_loss = criterion(log_ps, labels)
        test_loss += batch_loss.item()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
   
    return test_loss, accuracy

def main():
    args = parse_args()
    #data_dir = 'flowers'
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    user_gpu = args.gpu  # user input via cli

    if user_gpu == 'gpu' and torch.cuda.is_available():
        current_device = 'cuda:0'
    else:
        current_device = 'cpu'
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.RandomRotation(45),transforms.RandomVerticalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    testing_transforms = transforms.Compose([transforms.CenterCrop(224),transforms.Resize(255),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    data_transforms = {"training": training_transforms,"validation": validation_transforms,"testing": testing_transforms}
    
    train_data = datasets.ImageFolder(train_dir,transform=training_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=testing_transforms)
    image_datasets = {"training": train_data,"validation": valid_data,"testing": test_data}

                                                                
    training_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_data,batch_size=64,shuffle=True)
    testingloader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=True)

    dataloaders = {"training": training_loader,"validation": validation_loader, "testing": testingloader}
    

    #model = models.vgg16(pretrained=True)
    #model.name= 'vgg16'
    #hidden_layers= [4096,1024]
    hidden_layers= args.hidden_units
    #input_size= 25088
    #learning_rate = 0.001
    learning_rate = args.learning_rate
    output_size= 102
    drop_out = 0.2
    #epochs_number = 5
    epochs_number = args.epochs
    arch = args.arch
    if arch.lower() == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
        print(model)
        print("Pre-trained model is set to: Alexnet!")
    elif arch.lower() == 'resnet':
        model = models.resnet18(pretrained=True)
        print(model)
        input_size = 512       
        print("Pre-trained model is set to: Resnet18!")
    elif arch.lower() == 'vgg':
        model = models.vgg19(pretrained=True)
        print(model)
        input_size = 25088 
        print("Pre-trained model is set to: Vgg19!")
    elif arch.lower() == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024
        print("Pre-trained model is set to: Densenet121!")
    else:
        print("Correct choice for architecture is: alexnet or resnet or vgg or densenet...exiting!")
        Return    
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = Classifier(input_size,output_size,hidden_layers,drop_out)
    model.class_to_idx = train_data.class_to_idx
    #model.class_to_idx = training_loader.dataset.class_to_idx
    #model.classifier = Classifier(input_size,output_size,hidden_layers,drop_out)

    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
    model.to(current_device)
    optimizer=optim.Adam(model.classifier.parameters(),lr=learning_rate)

    with active_session():
        train_loss,valid_loss,valid_accuracy= train(model,epochs_number,criterion,optimizer,training_loader,validation_loader,current_device)
    
    print("Finalized result \n",
      f"Training loss: {train_loss:.3f}.. \n",
      f"Testing loss: {valid_loss:.3f}.. \n",
      f"Testing accuracy: {valid_accuracy:.3f}")

    filename=saveCheckPoint(model)

    
    
class Classifier(nn.Module):
    
    def __init__(self,input_size,output_size,hidden_layers,drop_out=0.2):
        super().__init__()
        #self.hidden_layers= nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
        #self.hidden_layers= nn.ModuleList([nn.Linear(int(input_size),int(hidden_layers))])
        self.hidden_layers= nn.ModuleList([nn.Linear(int(input_size),int(hidden_layers[0]))])
        hlayers=zip(hidden_layers[:-1],hidden_layers[1:])
       
       # replaced self.hidden_layers= nn.ModuleList([nn.Linear(int(input_size),int(hidden_layers[0]))])
        #hlayers = zip(int(hidden_layers),int(hidden_layers))
        zip(itertools.repeat(int(hidden_layers),int(hidden_layers)))
        #hlayers = zip(int(hidden_layers[:-1]),int(hidden_layers[1:]))
        self.hidden_layers.extend([nn.Linear(int(hinput),int(houtput)) for hinput,houtput in hlayers])
        self.output = nn.Linear(hidden_layers[-1],output_size)
        self.dropout = nn.Dropout(p=drop_out)
        
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        
        for layer in self.hidden_layers:
            x= self.dropout(F.relu(layer(x)))
            
        x= F.log_softmax(self.output(x),dim=1)
        return x
    
    
#hidden_layers= [4096,1024]
#input_size= 25088
output_size= 102
drop_out = 0.2





def saveCheckPoint(model):
    checkpoint = {'input_size':1024,
                  'output_size':102,
                  'name': model.name,
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}
    print("inside savecheckpoint")
    #model.class_to_idx=train_data.class_to_idx
    
    save_dir = args.save_dir
    torch.save(checkpoint,save_dir)
    
    
if __name__ == "__main__":
    main()
