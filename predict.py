import torch
from torch import nn
import numpy as np
from torch import optim
from PIL import Image
from collections import OrderedDict
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import time
import os 
import copy 
import json
data_dir = 'flowers'
train_dir = data_dir + '/train'
val_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
cat_to_name = "cat_to_name.json"
FILE = "checkpoint.pth"
image_path ='flowers/test/100/image_07902.jpg'
topk = 5

#device parameter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#image_transform
image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#model = models.densenet121(pretrained=True)
#for param in model.parameters():
    #param.requires_grad = False  
#loadchecpoint
checkpoint = torch.load(FILE)
model = checkpoint['model']
classifier =checkpoint['classifier'] 
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_dict'])
criterion = checkpoint['criterion']
#
model.eval()
model.class_to_idx = train_data.class_to_idx

image = Image.open(image_path)
image = image_transform(image)
image = image.unsqueeze_(0)
image=model.forward(image.to(device))
ps = F.softmax(image.data,dim=1)
print(ps)      
with torch.no_grad():
    probs = np.array(ps.topk(topk)[0][0])
print('this is top 5 prob ' , probs)  
idx_to_class = {val:key for key, val in train_data.class_to_idx.items()}
#print(idx_to_class)
top_classes = [np.int(idx_to_class[i]) for i in np.array(ps.topk(topk)[1][0])]
print(top_classes)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
labels = [cat_to_name[str(i)] for i in top_classes]
print(labels)

for i in range(5):
    print("\n the flower name : {} ==> class :  {} ==> probability :  {}".format(labels[i],top_classes[i], probs[i]))
