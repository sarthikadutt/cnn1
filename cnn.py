import matplotlib

from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Flatten, Dense
from tensorflow.contrib.distributions.python.ops.bijectors import inline

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
import matplotlib.pyplot as plt

print(type(train_X))
print(len(train_X))
print(len(test_X))
import numpy as np
classes = np.unique(train_Y)
no_of_cls = len(classes)

print('Total Number of Output Classes',no_of_cls)
print('Total  Classes',classes)

labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

from keras.utils import to_categorical
import matplotlib.pyplot as plt

print('Training data shape : ',train_X.shape,train_Y.shape)

print('Testing data shape : ',test_X.shape,test_Y.shape)

classes = np.unique(train_Y)
nclasses = len(classes)
print('Total number of output',nclasses)
print('Output Classes',classes)

labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

plt.figure(figsize=[5,5])

train_img = 15
test_img = 50
plt.subplot(1,2,1)
plt.imshow(train_X[train_img,:,:],cmap='gray')
plt.title('This is a :{}'.format(labels[train_Y[train_img]]))

plt.subplot(1,2,2)
plt.imshow(test_X[test_img,:,:],cmap='gray')
plt.title('This is a :{}'.format(labels[test_Y[test_img]]))
plt.show()

classes = np.unique(train_Y)
nclasses =  len(classes)
print('total number of output',nclasses)
print('output class', classes)
plt.figure(figsize=[5,5])
plt.subplot(1,1,1)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.show()

# data preprocessing

train_X = train_X.reshape(-1,28,28,1)
test_X = test_X.reshape(-1,28,28,1)

train_X.shape , test_X.shape

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X = train_X/255

test_X = test_X/255

train_Y[:10]

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print('Orignal Label : ',train_Y[1])
print('After conversion to one - hot : ',train_Y_one_hot[1])

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X,train_Y_one_hot,test_size=0.2,random_state=101)

train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense ,Dropout,Flatten
from keras.layers import  Conv2D,MaxPool2D
from keras.layers.normalization  import  BatchNormalization
from keras.layers.advanced_activations import  ReLU

batch_size = 64
epochs = 5
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(ReLU())
fashion_model.add(MaxPool2D((2,2),padding='same'))
fashion_model.add(Conv2D(64,kernel_size=(3,3),activation='linear',padding='same'))
fashion_model.add(ReLU())
fashion_model.add(MaxPool2D((2,2),padding='same'))
fashion_model.add(Conv2D(128,kernel_size=(3,3),activation='linear',padding='same'))
fashion_model.add(ReLU())
fashion_model.add(MaxPool2D((2,2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128,activation='linear'))
fashion_model.add(ReLU())
fashion_model.add(Dense(num_classes,activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['acc'])

fashion_model.summary()

print(train_X.shape)

print(train_label.shape)

print(valid_X.shape)

print(valid_label.shape)

fashion_train = fashion_model.fit(train_X,train_label,batch_size=batch_size,epochs=epochs,validation_data=(valid_X,valid_label))
fashion_model.evaluate(test_X,test_Y_one_hot)

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss =  fashion_train.history['val_loss']
epochs =range(len(accuracy))
plt.plot(epochs,accuracy,'bo',label='Training accuracy')
plt.plot(epochs,val_accuracy,'b',label='Validation Accuracy')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.show()

fashion_model = Sequential()
fashion_model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(ReLU())
fashion_model.add(MaxPool2D((2,2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64,kernel_size=(3,3),activation='linear',padding='same'))
fashion_model.add(ReLU())
fashion_model.add(MaxPool2D((2,2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128,kernel_size=(3,3),activation='linear',padding='same'))
fashion_model.add(ReLU())
fashion_model.add(MaxPool2D((2,2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128,activation='linear'))
fashion_model.add(ReLU())
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes,activation='softmax'))

fashion_model.summary()
epochs = 5

fashion_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
fashion_train_dropout = fashion_model.fit(train_X,train_label,batch_size=batch_size,epochs=epochs,validation_data=(valid_X,valid_label))
fashion_model.save('fashion_model_droput.h5py')
test_eval = fashion_model.evaluate(test_X,test_Y_one_hot)

accuracy = fashion_train_dropout.history['acc']
val_accuracy = fashion_train_dropout.history['val_acc']
loss = fashion_train_dropout.history['loss']
val_loss =  fashion_train_dropout.history['val_loss']
epochs =range(len(accuracy))
plt.plot(epochs,accuracy,'bo',label='Training accuracy')
plt.plot(epochs,val_accuracy,'b',label='Validation Accuracy')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.show()
predicted_classes = fashion_model.predict(test_X)
predicted_classes[0]
np.round(predicted_classes[0])