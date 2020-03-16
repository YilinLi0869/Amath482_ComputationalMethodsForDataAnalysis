#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


# In[16]:


# Load the Fashion_MNIST data
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full),(X_test,y_test) = fashion_mnist.load_data()


# In[17]:


X_train_full.shape


# In[18]:


# Plot the first images of fashion items in row k (Not necessary)
plt.figure()
for k in range(10):
    plt.subplot(2,5,k+1)
    plt.imshow(X_train_full[k], cmap='gray')
    plt.axis('off')
plt.show()


# In[19]:


# Try
y_train_full[:10]


# In[20]:


# Preprocess the data

# Convert X_train, X_valid, and X_test to floating point numbers between 0 and 1 by dividing each by 255.0
# Remove 5,000 images from the training data to use as validation data
X_valid = X_train_full[:5000] / 255.0
# End up with 55,000 training examples in an array X_train
X_train = X_train_full[5000:] / 255.0
X_test = X_test / 255.0

y_valid = y_train_full[:5000]
y_train = y_train_full[5000:]


# In[21]:


# Part I

from functools import partial

my_dense_layer = partial(tf.keras.layers.Dense, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    my_dense_layer(300),
    my_dense_layer(200),
    my_dense_layer(100),
    my_dense_layer(10, activation="softmax")
])


# In[30]:


model.compile(loss="sparse_categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
             metrics=["accuracy"])


# In[31]:


history = model.fit(X_train, y_train, epochs=15, validation_data=(X_valid,y_valid))


# In[32]:


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[33]:


y_pred = model.predict_classes(X_train)
conf_train = confusion_matrix(y_train, y_pred)
print(conf_train)


# In[34]:


model.evaluate(X_test,y_test)


# In[35]:


y_pred = model.predict_classes(X_test)
conf_test = confusion_matrix(y_test, y_pred)
print(conf_test)


# In[37]:


fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

# create table and save to file
df = pd.DataFrame(conf_test)
ax.table(cellText=df.values, rowLabels=np.arange(10), colLabels=np.arange(10), loc='center', cellLoc='center')
fig.tight_layout()
plt.savefig('conf_mat2.pdf')


# In[ ]:




