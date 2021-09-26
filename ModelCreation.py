#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Importing the necessary libraries:
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# Resizing all the images, because ResNet works with a specific size of images:
image_size = [224, 224]

train_path = 'Datasets/Train/'
valid_path = 'Datasets/Test/'


# In[3]:


# Importing the ResNet 50 library and added pre-processing layer to the front of VGG:
resnet = ResNet50(input_shape=image_size + [3], weights='imagenet', include_top=False)


# In[4]:


# Not training the existing weights:
for layer in resnet.layers:
    layer.trainable = False


# In[5]:


# Getting number of output classes:
folders = glob('Datasets/Train/*')


# In[6]:


folders


# In[7]:


x = Flatten()(resnet.output)


# In[8]:


prediction = Dense(len(folders), activation='softmax')(x)

# Creating model object:
model = Model(inputs = resnet.input, outputs=prediction)


# In[9]:


model.summary()


# In[10]:


# Optimizations for the model:
model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[11]:


# Use ImageDataGenerator to augment our datasets:
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2, 
                                   zoom_range=0.2,
                                   horizontal_flip=True)
# No augmentation in test data:
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[12]:


training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[13]:


test_set = test_datagen.flow_from_directory('Datasets/Test/',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[14]:


#Fitting the model:
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=30,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)


# ### Visualizing losses:

# In[15]:


plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'], label='Validation loss')
plt.legend()


# ### Visualizing accuracies:

# In[17]:


plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='validation accuracy')
plt.legend()


# ### Saving our model as a h5 file:

# In[18]:


from tensorflow.keras.models import load_model

model.save('model_ResNet50.h5')


# ### Making predictions:

# In[19]:


pred = model.predict(test_set)


# In[22]:


pred


# In[23]:


# Making our predictions easier to understand:
pred = np.argmax(pred, axis=1)


# In[24]:


pred


# ### Making real-time predictions:

# In[25]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[26]:


model = load_model('model_ResNet50.h5')


# In[42]:


# Reading our input image:
img = image.load_img('Datasets/Test/lamborghini/14.jpg', target_size=(224, 224))


# In[43]:


x = image.img_to_array(img)


# In[44]:


x


# In[45]:


# Confirming the shape:
x.shape


# In[46]:


x = x/255 # Did this in accordance to test_datagen(cell 11)


# In[47]:


x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[48]:


model.predict(img_data)


# In[49]:


finalOutput = np.argmax(model.predict(img_data), axis=1)


# In[50]:


finalOutput


# ### Hence, it is successfully predicting that the input image is a Lamborghini.
