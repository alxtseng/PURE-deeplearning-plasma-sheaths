#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:01:38 2019

@author: alextseng
"""

import torch
#import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as Variable
import numpy as np

from dataloader import load_data
from classes import Net
import imageio

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 1e-4
momentum = 0.5
log_interval = 10



para_list, iead_list = load_data()
para_tensor = torch.FloatTensor(para_list)
iead_tensor = torch.FloatTensor(iead_list)

net = Net()

net.train()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

print(iead_list.shape)
print(iead_list[0])


def train(num_epochs):
    print('Training Network')
    #losses= []
    for epoch in range(num_epochs):
        running_loss = 0.0
        #train_acc = 0.0
        for i, data in enumerate(para_tensor, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs = data
            labels = iead_tensor[i]

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = net(inputs)
        #print(outputs.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
            
            running_loss += loss.item()
            #losses.append(loss.data)
            
            #_, prediction = torch.max(outputs, 0)
            
            
            #train_acc += torch.sum(prediction == labels.data)
            
            
            #if i % 2000 == 1999:    # print every 2000 mini-batches
                #print('[%d, %5d] loss: %.3f' %
                 #     (epoch + 1, i + 1, running_loss / 2000))
                #running_loss = 0.0
                
        #train_acc = train_acc / 90000
        #train_loss = running_loss / 90000        
        running_loss /= para_list.shape[0]
        #print(losses)
        print(running_loss)
    
    #print("Epoch {}, Train Accuracy: {} , TrainLoss: {} ".format(epoch, train_acc, train_loss))
    print('Finished Training')

train(100)

torch.save(net, "model.pth")

#print(predictions.shape)

"""
print('Training Network')
for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(para_list, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data
        labels = iead_list[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(outputs.shape)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    print(running_loss)

print('Finished Training')

print(outputs.shape)


picture = np.squeeze(iead_list)
"""

#Produce figures 
print('Producing figures...')
for i, para in enumerate(para_list):
    fig = plt.figure(i)
    #fig.suptitle('Ti/Te: '+str(simulation_list[-i].Ti_Te) +\
    #    ' B: '+str(round(simulation_list[-i].B,2)) +\
   #     ' psi: '+str(round(simulation_list[-i].psi,2)))
   
    outputs = net(torch.FloatTensor(para))
    ax = fig.subplots(1,2,False,True,True)
    ax[0].imshow(outputs.data.numpy().reshape((500,180)))
    ax[0].set_title('NN')
    ax[0].xlabel("incident angle [deg]")
    ax[0].ylabel("incident energy [eV]")
    ax[1].imshow(iead_list[i].reshape((500,180)))
    ax[1].set_title('hPIC')
    ax[1].xlabel("incident angle [deg]")
    ax[1].ylabel("incident energy [eV]")
    #error = iead_list[-i,:]/np.max(iead_list[-i,:]) - predictions[-i,:]/np.max(predictions[-i,:])
    #ax[2].imshow(error.reshape((240,90))[:50,:])
    #ax[2].set_title('error')
    plt.savefig(str(i)+'.png')
    plt.close()
#end for

#Produce animated figure
print('Producing animated figure...')
images = []
for filename in [str(i)+'.png' for i in range(num_figures)]:
    images.append(imageio.imread(filename))
#end for
imageio.mimsave('movie.gif',images,duration=0.5)

