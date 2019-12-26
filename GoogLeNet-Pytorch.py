#!/usr/bin/env python
# coding: utf-8

# # GoogLeNet

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as vdatasets
import torchvision.utils as vutils
import random
import os, pickle
# from tensorboardX import SummaryWriter
torch.manual_seed(1)

# DATA_PATH = os.environ['/data/']
USE_CUDA = torch.cuda.is_available()
print("CUDA: {}".format(USE_CUDA))


# In[2]:


batch_size = 5
learning_rate = 0.0002
epoch = 10


# ## Data

# In[3]:

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

img_dir = "./data"


# In[4]:


img_data = vdatasets.CIFAR10(
    img_dir,
    train=True,
    transform=transform,
    download=True,
)
img_data


# In[5]:


print("""
CIFAR10数据集中的图片分为{}类: {}
""".format(
    len(img_data.classes),
    ', '.join(img_data.classes),
))


# In[6]:


my_img_index = 0

my_img = img_data.data[my_img_index]
# plt.imshow(my_img)
print("第{}张图片：".format(my_img_index))
print("size: {}".format(my_img.shape))
print("transform 后 size: {}".format(img_data[my_img_index][0].size()))

print("它所属类别为：{} {}".format(
    img_data.targets[my_img_index],
    img_data.classes[img_data.targets[my_img_index]]
))


# In[7]:


img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)


# ## Model

# In[8]:


class Inception(nn.Module):
    
    def __init__(self,in_ch,out_ch1,mid_ch13,out_ch13,mid_ch15,out_ch15,out_ch_pool_conv,auxiliary=False):
        super(Inception,self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch1,kernel_size=1,stride=1),
            nn.ReLU())
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_ch,mid_ch13,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch13,out_ch13,kernel_size=3,stride=1,padding=1),
            nn.ReLU())
        
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_ch,mid_ch15,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch15,out_ch15,kernel_size=5,stride=1,padding=2),
            nn.ReLU())
        
        self.pool_conv1 = nn.Sequential(
            nn.MaxPool2d(3,stride=1,padding=1),
            nn.Conv2d(in_ch,out_ch_pool_conv,kernel_size=1,stride=1),
            nn.ReLU())
        
        self.auxiliary = auxiliary
        
        if auxiliary:
            self.auxiliary_layer = nn.Sequential(
                nn.AvgPool2d(5,3),
                nn.Conv2d(in_ch,128,1),
                nn.ReLU())
        
    def forward(self,inputs,train=False):
        conv1_out = self.conv1(inputs)
        conv13_out = self.conv13(inputs)
        conv15_out = self.conv15(inputs)
        pool_conv_out = self.pool_conv1(inputs)
        outputs = torch.cat([conv1_out,conv13_out,conv15_out,pool_conv_out],1) # depth-wise concat
        
        if self.auxiliary:
            if train:
                outputs2 = self.auxiliary_layer(inputs)
            else:
                outputs2 = None
            return outputs, outputs2
        else:
            return outputs


# In[9]:


class GoogLeNet(nn.Module):
    
    def __init__(self,num_output=1000):
        super(GoogLeNet,self).__init__()
        
        self.stem_layer = nn.Sequential(
                                                        nn.Conv2d(3,64,7,2,3),
                                                        nn.ReLU(),
                                                        nn.MaxPool2d(3,2,1),
                                                        nn.Conv2d(64,64,1),
                                                        nn.ReLU(),
                                                        nn.Conv2d(64,192,3,1,1),
                                                        nn.ReLU(),
                                                        nn.MaxPool2d(3,2,1)
                                                        )
        
        #in_ch,out_ch_1,mid_ch_13,out_ch_13,mid_ch_15,out_ch_15,out_ch_pool_conv
        self.inception_layer1 = nn.Sequential(
                                                                Inception(192,64,96,128,16,32,32),
                                                                Inception(256,128,128,192,32,96,64),
                                                                nn.MaxPool2d(3,2,1)
                                                               )
        
        self.inception_layer2 = nn.Sequential(
                                                                Inception(480,192,96,208,16,48,64),
                                                                Inception(512,160,112,224,24,64,64),
                                                                Inception(512,128,128,256,24,64,64),
                                                                Inception(512,112,144,288,32,64,64),
                                                                Inception(528,256,160,320,32,128,128),
                                                                nn.MaxPool2d(3,2,1)
                                                               )
        
        #self.inception_layer3 = Inception(528,256,160,320,32,128,128,True) # auxiliary classifier
        #self.auxiliary_layer = nn.Linear(128*4*4,num_output)
        
        self.inception_layer3 = nn.Sequential(
                                                                #nn.MaxPool2d(3,2,1),
                                                                Inception(832,256,160,320,32,128,128),
                                                                Inception(832,384,192,384,48,128,128),
                                                                nn.AvgPool2d(7,1)
                                                               )
        
        self.dropout = nn.Dropout2d(0.4)
        self.output_layer = nn.Linear(1024,num_output)
        
    def forward(self,inputs,train=False):
        outputs = self.stem_layer(inputs)
        outputs = self.inception_layer1(outputs)
        outputs = self.inception_layer2(outputs)
        #outputs,outputs2 = self.inception_layer3(outputs)
        #if train:
            # B,128,4,4 => B,128*4*4
        #    outputs2 = self.auxiliary_layer(outputs2.view(inputs.size(0),-1))
        outputs = self.inception_layer3(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.view(outputs.size(0),-1) # 동일 : outputs = outputs.view(batch_size,-1)
        outputs = self.output_layer(outputs)
        
        #if train:
        #   return outputs, outputs2
        return outputs


# In[10]:


inception = Inception(64, 32, 32, 64, 16, 32, 64)
model = GoogLeNet(10)


# In[11]:


loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ## Train

# In[12]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inception.to(device)
model = nn.DataParallel(model)
model.to(device)

print(device)


# In[ ]:


for epoch_index in range(epoch):
    running_loss = 0.0
    for i, (img, label) in enumerate(img_batch, 0):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(img)
        
        #print(output.size())
        loss = loss_func(output,label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch_index + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


# In[ ]:

MODEL_PATH = "googlenet-model"

# Save:
torch.save(model.state_dict(), MODEL_PATH)

# Load:

# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# model

# In[ ]:




