
# coding: utf-8

# In[2]:

from __future__ import print_function

import numpy as np
import cPickle
import matplotlib.pyplot as plt
import caffe


# In[3]:

def collage(data):
    images = [img for img in data.transpose(0, 2, 3, 1)]
    side = int(np.ceil(len(images)**0.5))
    for i in range(side**2 - len(images)):
        images.append(images[-1])
    collage = [np.concatenate(images[i::side], axis=0)
               for i in range(side)]
    collage = np.concatenate(collage, axis=1)
    #collage -= collage.min()
    #collage = collage / np.absolute(collage).max() * 256
    return collage
    


# ## Read data from CIFAR 10

# In[4]:

trnData = []
trnLabels = []
tstData = []
tstLabels = []
for i in range(1,6):
    with open('../data_batch_{}'.format(i)) as f:
        data = cPickle.load(f)
    if i == 5:
        tstData = data['data']
        tstLabels = data['labels']
    else:
        trnData.append(data['data'])
        trnLabels.append(data['labels'])
trnData = np.concatenate(trnData).reshape(-1, 3, 32, 32)
trnData = np.concatenate([trnData[:,:,:,::-1], trnData[:,:,:,:]])
trnLabels = np.concatenate(trnLabels)
trnLabels = np.concatenate([trnLabels, trnLabels])
tstData = tstData.reshape(-1, 3, 32, 32)
tstData = np.concatenate([tstData[:,:,:,::-1], tstData[:,:,:,:]])
tstLabels = np.concatenate([tstLabels, tstLabels])

print('Trn data shape:', trnData.shape)
print('Tst data shape:', tstData.shape)
plt.subplot(1, 2, 1)
img = collage(trnData[:16])
plt.imshow(img)
plt.subplot(1, 2, 2)
img = collage(tstData[:16])
plt.imshow(img)
plt.show()

print('Trn labels shape: ', trnLabels.shape)
print('Tst labels shape: ', tstLabels.shape)
print(trnLabels[:20])
print(tstLabels[:20])


# ## Normalize data

# In[5]:

trnData = trnData.astype(np.float32) / 255 - 0.5
tstData = tstData.astype(np.float32) / 255 - 0.5
'''def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return ZCAMatrix
np.dot(ZCAMatrix, inputs)
'''


# ## Load solver

# In[6]:

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.get_solver('net_solver.prototxt')


# In[ ]:

for i in range(10):
    solver.net.blobs['data'].data[...] = trnData[:64,:, :, :]
    solver.net.blobs['labels'].data[...] = trnLabels[:64].reshape(-1,1,1,1)
    solver.step(1)
    import ipdb; ipdb.set_trace()

    


# ## Prepare generator and show data from the generator

# In[27]:

border = 32 - dataSize
#central crop validation data
valData = tstData[:,:,border:border+dataSize, border:border+dataSize]
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 2)
img = collage(valData[:16] + 0.5)
plt.imshow(img)


gen1 = generator(data=trnData, labels=trnLabels, targetSize=dataSize, batchSize=512)

import itertools
for batchData in itertools.islice(gen1, 1):
    img = collage(batchData[0][:16])
    plt.subplot(1, 2, 1)
    plt.imshow(img + 0.5)
plt.show()
# it(
#    {'data':trnData}, trnLabels,
#    validation_data=(tstData, tstLabels),
#    batch_size=512, nb_epoch=20)
print('DONE')



# In[ ]:

model.optimizer.lr = 0.001
while True:
    print('LERANING RATE', model.optimizer.lr)
    model.fit_generator(
        gen1, 
        samples_per_epoch = 40000, 
        nb_epoch = 30,  validation_data=(valData, tstLabels), nb_worker=1)
    model.optimizer.lr *= 0.5

