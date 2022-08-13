#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import optim
import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F


# In[3]:


train_images = np.load("../tc/train-images.npy")
train_labels = np.load("../tc/train-labels.npy")
test_images = np.load("../tc/test-images.npy")
test_labels = np.load("../tc/test-labels.npy")


# In[3]:


#对1维图像还原成2维图像
train_images=train_images.reshape(-1,1,28,28)
test_images=test_images.reshape(-1,1,28,28)
# 对图像修改维度 从num*height*width*channel 到 num*channel*height*width
#train_images=np.transpose(train_images,[0,3,1,2])
#test_images=np.transpose(test_images,[0,3,1,2])
# 对数据进行分batch
batch_size=16
train_dataset=TensorDataset(torch.tensor(train_images),torch.tensor(train_labels))
test_dataset=TensorDataset(torch.tensor(test_images),torch.tensor(test_labels))
train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True)


# In[4]:


#定义网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #输入[batchsize，1,28,28]
        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.pooling1=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu1=torch.nn.ReLU()
        self.conv2=torch.nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.pooling2=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu2=torch.nn.ReLU()
        #self.conv3=torch.nn.Conv2d(in_channels=3,out_channels=120,kernel_size=5)
        
       # self.linear1=torch.nn.Linear(256,120)
        self.linear1=torch.nn.Linear(6*12*12,120)
        self.relu3=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(120,84)
        self.relu4=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(84,10)
       # self.softmax=F.softmax(dim=1)
        
    def forward(self,x):
        # x batchsize,1,28,28
        
        out=self.conv1(x) # out batchsize, 6,24,24
        out=self.relu1(out) 
        out=self.pooling1(out) # batchsize,6,12,12

        
#         out2=self.conv2(out) # batchsize,16,8,8
#         out2=self.relu2(out2)
#         out2=self.pooling2(out2) #batchsize,16,4,4
        
        #print(out.shape)
        out2=out
        out3=out2.view(out2.shape[0],-1)
        #print(out3.shape)
        out4=self.linear1(out3)
        out4=self.relu3(out4)
        out4=self.linear2(out4)
        out4=self.relu4(out4)
        
        result=self.linear3(out4)
        return result


# In[5]:


device = torch.device('cuda:0')
model = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-4,momentum=0.9)


# In[6]:


def train():
    train_loss = 0
    train_acc = 0

    for item in (train_loader):
        data, label = item[0].float().to(device), item[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,label.long()).sum()  
        loss.backward()  
        optimizer.step()  
        train_loss += loss.item()
        train_acc += (torch.argmax(output, dim=1) == label).sum().item()
    return train_loss / train_images.shape[0], train_acc / train_images.shape[0]


# In[7]:


def test():
    test_loss = 0
    test_acc = 0

    for item in (test_loader):
        data, label = item[0].float().to(device), item[1].to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output,label.long()).sum()  
            test_loss += loss.item()
            test_acc += (torch.argmax(output, 1) == label).sum().item()

    return test_loss / test_images.shape[0], test_acc / test_images.shape[0]


# In[8]:


epochs=50
train_loss_list=[]
train_acc_list=[]
test_loss_list=[]
test_acc_list=[]

for i in range(epochs):
    loss1,acc1=train()
    train_loss_list.append(loss1)
    train_acc_list.append(acc1)
    loss2,acc2=test()
    test_loss_list.append(loss2)
    test_acc_list.append(acc2)
    
    print("epoch:",i,loss1,acc1,loss2,acc2)


# In[ ]:


for i in tqdm.tqdm_notebook(range(epochs), desc='1st loop'):
    loss1,acc1=train()
    train_loss_list.append(loss1)
    train_acc_list.append(acc1)
    loss2,acc2=test()
    test_loss_list.append(loss2)
    test_acc_list.append(acc2)
    
    print("epoch:",i,loss1,acc1,loss2,acc2)


# In[35]:


f = open("test_result.txt", "a")
for i in range(len(train_acc_list)):
    f.write("%.6f " %train_acc_list[i])
    f.write("%.6f \n" %train_loss_list[i])
    f.write("%.6f " %test_acc_list[i])
    f.write("%.6f \n" %test_loss_list[i])

f.close()


# In[33]:


count=[i for i in range(len(train_acc_list))]
plt.plot(count,train_acc_list,label='train')
plt.plot(count,test_acc_list,label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend() # 显示图例
plt.savefig('acc.jpg')


# In[34]:


count=[i for i in range(len(train_loss_list))]
plt.plot(count,train_loss_list,label='train')
plt.plot(count,test_loss_list,label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend() # 显示图例
plt.savefig('loss5.jpg')


# In[ ]:




