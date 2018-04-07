# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 09:38:20 2018

@author: HP
"""
# import the necessary packages
import numpy as np
import os
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop
# loading all image path
data_files=glob.iglob("data/**/*.nii",recursive=True) # load all nii files

# find patient id 002_S_0413 like that pat_id=413
pat_id=[]
image_url=[]
image_class=[]  
for data in data_files:
   #print(data)
    image_url.insert(len(image_url),data)
    data1=data.split(os.sep)
   # print(data1[1])
    data2=data1[1].split("_")
    pat_id.insert(len(pat_id),int(data2[2]))
 
 # this is for preprocessing    
#f= open("new.bat","w+")
#for i in range(len(image_url)):
     #f.write("bse -i "+image_url[i]+" -o "+image_url[i]+"\n")
#f.close()
     
fr = pd.read_csv("DXSUM_PDXCONV_ADNIALL.csv")
for j in range(len(pat_id)):
    val=0
    for i in range(len(fr)):
        if(fr['RID'][i]==pat_id[j]):
            if(fr['DXNORM'][i]==1):
                val=0
            if(fr['DXMCI'][i]==1):
                val=0
            if(fr['DXAD'][i]==1):
                val=1
    image_class.insert(j,val)

count=0;
final_label=[]
image_data1=np.zeros((22420,)+(170,256)+(1,),dtype=np.uint8)
for i in range(len(image_class)):
    img=nb.load(image_url[i])
    image_data=img.get_data()
    get_z=image_data.shape[2]
    for j in range(30,220):
        slice_1=image_data[:,:,j]
        slice_2=np.resize(slice_1,(170,256,1))
        image_data1[count]=slice_2
        final_label.insert(len(final_label),image_class[i])
        count=count+1

#===========================================================================
# data splitting 
(trainData, testData, trainLabels, testLabels) = train_test_split(image_data1,final_label,test_size=0.25)
trainLabels=to_categorical(trainLabels, num_classes=2)
testLabels=to_categorical(testLabels, num_classes=2)

# =============================================================================


model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (5,5),strides=(1,1),padding = 'Same', activation ='relu', input_shape = (170,256,1)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),strides=(1,1), padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# =============================================================================
model.add(Flatten())
model.add(Dense(250, activation = "relu"))
model.add(Dense(2, activation = "softmax"))

# Define the optimizer
#optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = "adam" , loss = "mean_squared_error", metrics=["accuracy"])


# Without data augmentation i obtained an accuracy of 0.98114
#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,

    
hist=model.fit(trainData, trainLabels,validation_data = (testData, testLabels),epochs=5,verbose = True,batch_size=16)

    
    
    
