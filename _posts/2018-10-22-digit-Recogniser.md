---
title: "Machine Learning Project: Digit Recogniser (Kaggle)"
date: 2018-10-24
tags: [machine learning, data science, kaggle, MINST]
header:
  image: "/images/digit-recogniser/header.jpg"
excerpt: "Machine Learning, Data Science, MINST"
mathjax: "true"
---

# Tensorflow implementation of digit recognition

Data from Kaggle Digit Recognition competition  
https://www.kaggle.com/c/digit-recognizer/data


```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
```

    Using TensorFlow backend.
    

## Load data


```python
train_file = 'train.csv'
test_file = 'test.csv'

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
```


```python
train_df.head() # labels and pixels
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>




```python
test_df.head() # no labels here
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 784 columns</p>
</div>



So each row of the dataset is a sequence of ink intensity for the 784 pixels that represents square image with the size 28x28.


```python
train_labels = train_df.label
train_images = train_df.iloc[:,1:]
test_images = test_df
```

## Show some digits from the input dataset


```python
plt.figure(figsize=(12,6))
for i in range(0,9):
    plt.subplot(250 + (i+1))
    img = train_images.iloc[i,:].values.reshape(28, 28)
    plt.imshow(img, cmap='Blues')
    plt.title(train_labels[i])
```


![png](./images/digit-recogniser/output_10_0.png)


## Preprocess data for NN


```python
train_images = (train_images/train_images.max()).fillna(0) # normalize values
test_images = (test_images/test_images.max()).fillna(0) # normalize values
train_labels = pd.get_dummies(train_labels) # one-hot encoding of the label
```


```python
train_labels.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Build simple 1-hidden layer dense NN
Some more information about Keras' Sequential models  
https://keras.io/getting-started/sequential-model-guide/


```python
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# Multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 32)                25120     
    _________________________________________________________________
    activation_1 (Activation)    (None, 32)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                330       
    _________________________________________________________________
    activation_2 (Activation)    (None, 10)                0         
    =================================================================
    Total params: 25,450
    Trainable params: 25,450
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# import pydot
# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png')
```

## Train the model


```python
print(train_images.values.shape)
print(train_labels.values.shape)
```

    (42000, 784)
    (42000, 10)
    

Let's fit the weights of our NN. To estimate overfitting, let's use validation dataset which we set up as 5% of the training dataset - validation_split = 0.05.


```python
history=model.fit(train_images.values, train_labels.values, validation_split = 0.05, 
            nb_epoch=25, batch_size=64)
```

    E:\IDE\WPy-3662\python-3.6.6.amd64\lib\site-packages\ipykernel_launcher.py:2: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      
    

    Train on 39900 samples, validate on 2100 samples
    Epoch 1/25
    39900/39900 [==============================] - 3s 81us/step - loss: 0.4539 - acc: 0.8785 - val_loss: 0.3141 - val_acc: 0.9143
    Epoch 2/25
    39900/39900 [==============================] - 3s 64us/step - loss: 0.2465 - acc: 0.9286 - val_loss: 0.2520 - val_acc: 0.9348
    Epoch 3/25
    39900/39900 [==============================] - 3s 70us/step - loss: 0.2031 - acc: 0.9421 - val_loss: 0.2269 - val_acc: 0.9424
    Epoch 4/25
    39900/39900 [==============================] - 3s 65us/step - loss: 0.1752 - acc: 0.9490 - val_loss: 0.2058 - val_acc: 0.9429
    Epoch 5/25
    39900/39900 [==============================] - 3s 67us/step - loss: 0.1562 - acc: 0.9545 - val_loss: 0.1911 - val_acc: 0.9481
    Epoch 6/25
    39900/39900 [==============================] - 3s 66us/step - loss: 0.1414 - acc: 0.9586 - val_loss: 0.1873 - val_acc: 0.9486
    Epoch 7/25
    39900/39900 [==============================] - 3s 71us/step - loss: 0.1306 - acc: 0.9613 - val_loss: 0.1816 - val_acc: 0.9533
    Epoch 8/25
    39900/39900 [==============================] - 3s 63us/step - loss: 0.1202 - acc: 0.9647 - val_loss: 0.1698 - val_acc: 0.9552
    Epoch 9/25
    39900/39900 [==============================] - 3s 67us/step - loss: 0.1114 - acc: 0.9671 - val_loss: 0.1710 - val_acc: 0.9543
    Epoch 10/25
    39900/39900 [==============================] - 2s 61us/step - loss: 0.1052 - acc: 0.9692 - val_loss: 0.1703 - val_acc: 0.9581
    Epoch 11/25
    39900/39900 [==============================] - 3s 66us/step - loss: 0.0993 - acc: 0.9710 - val_loss: 0.1605 - val_acc: 0.9590
    Epoch 12/25
    39900/39900 [==============================] - 3s 70us/step - loss: 0.0937 - acc: 0.9724 - val_loss: 0.1620 - val_acc: 0.9576
    Epoch 13/25
    39900/39900 [==============================] - 3s 78us/step - loss: 0.0884 - acc: 0.9736 - val_loss: 0.1523 - val_acc: 0.9624
    Epoch 14/25
    39900/39900 [==============================] - 3s 71us/step - loss: 0.0840 - acc: 0.9757 - val_loss: 0.1536 - val_acc: 0.9610
    Epoch 15/25
    39900/39900 [==============================] - 3s 78us/step - loss: 0.0802 - acc: 0.9763 - val_loss: 0.1642 - val_acc: 0.9557
    Epoch 16/25
    39900/39900 [==============================] - 3s 80us/step - loss: 0.0766 - acc: 0.9780 - val_loss: 0.1610 - val_acc: 0.9562
    Epoch 17/25
    39900/39900 [==============================] - 3s 70us/step - loss: 0.0731 - acc: 0.9785 - val_loss: 0.1608 - val_acc: 0.9581
    Epoch 18/25
    39900/39900 [==============================] - 3s 75us/step - loss: 0.0703 - acc: 0.9791 - val_loss: 0.1611 - val_acc: 0.9614
    Epoch 19/25
    39900/39900 [==============================] - 3s 71us/step - loss: 0.0680 - acc: 0.9797 - val_loss: 0.1630 - val_acc: 0.9633
    Epoch 20/25
    39900/39900 [==============================] - 3s 76us/step - loss: 0.0647 - acc: 0.9809 - val_loss: 0.1575 - val_acc: 0.9624
    Epoch 21/25
    39900/39900 [==============================] - 3s 63us/step - loss: 0.0631 - acc: 0.9816 - val_loss: 0.1624 - val_acc: 0.9600
    Epoch 22/25
    39900/39900 [==============================] - 2s 63us/step - loss: 0.0599 - acc: 0.9826 - val_loss: 0.1659 - val_acc: 0.9590
    Epoch 23/25
    39900/39900 [==============================] - 3s 63us/step - loss: 0.0579 - acc: 0.9824 - val_loss: 0.1617 - val_acc: 0.9619
    Epoch 24/25
    39900/39900 [==============================] - 2s 60us/step - loss: 0.0558 - acc: 0.9833 - val_loss: 0.1745 - val_acc: 0.9586
    Epoch 25/25
    39900/39900 [==============================] - 2s 59us/step - loss: 0.0535 - acc: 0.9839 - val_loss: 0.1684 - val_acc: 0.9595
    


```python
hist_df = pd.DataFrame(history.history)
```


```python
hist_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>val_loss</th>
      <th>val_acc</th>
      <th>loss</th>
      <th>acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.314067</td>
      <td>0.914286</td>
      <td>0.453909</td>
      <td>0.878546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.251983</td>
      <td>0.934762</td>
      <td>0.246468</td>
      <td>0.928571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.226936</td>
      <td>0.942381</td>
      <td>0.203141</td>
      <td>0.942080</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.205757</td>
      <td>0.942857</td>
      <td>0.175246</td>
      <td>0.948972</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.191095</td>
      <td>0.948095</td>
      <td>0.156158</td>
      <td>0.954486</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize=(14,6))
plt.style.use('bmh')
params_dict = dict(linestyle='solid', linewidth=0.25, marker='o', markersize=6)

plt.subplot(121)
plt.plot(hist_df.loss, label='Training loss', **params_dict)
plt.plot(hist_df.val_loss, label='Validation loss', **params_dict)
plt.title('Loss for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(hist_df.acc, label='Training accuracy', **params_dict)
plt.plot(hist_df.val_acc, label='Validation accuracy', **params_dict)
plt.title('Accuracy for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
```




    <matplotlib.legend.Legend at 0xe4d7fd0>




![png](./images/recogniser/output_24_1.png)


Some conclusions here: 
1. At some number of the epochs (8-10 in this case) the validation loss stops decreasing, i.e. the network begin to overfit. The training loss still decreases.
2. The same is true for the training/validation accuracy - validation accuracy stops increasing after some number of epochs.
3. Since validation data was not used for training, measuring accuracy on the the validation dataset gives us an estimation of the ability of the model to generalize.

Basically we tuned the only one parameter - number of epochs using validation.
Now I will use epochs=10.


```python
# Predict on the whole dataset now, 10 epochs
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# Multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images.values, train_labels.values, nb_epoch=10, batch_size=64)
```

    E:\IDE\WPy-3662\python-3.6.6.amd64\lib\site-packages\ipykernel_launcher.py:14: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      
    

    Epoch 1/10
    42000/42000 [==============================] - 3s 69us/step - loss: 0.4778 - acc: 0.8690
    Epoch 2/10
    42000/42000 [==============================] - 3s 71us/step - loss: 0.2596 - acc: 0.9256
    Epoch 3/10
    42000/42000 [==============================] - 3s 67us/step - loss: 0.2147 - acc: 0.9383
    Epoch 4/10
    42000/42000 [==============================] - 3s 73us/step - loss: 0.1843 - acc: 0.9471
    Epoch 5/10
    42000/42000 [==============================] - 3s 77us/step - loss: 0.1621 - acc: 0.9524
    Epoch 6/10
    42000/42000 [==============================] - 3s 66us/step - loss: 0.1459 - acc: 0.9574
    Epoch 7/10
    42000/42000 [==============================] - 3s 65us/step - loss: 0.1321 - acc: 0.9615
    Epoch 8/10
    42000/42000 [==============================] - 3s 62us/step - loss: 0.1219 - acc: 0.9647
    Epoch 9/10
    42000/42000 [==============================] - 3s 66us/step - loss: 0.1127 - acc: 0.9678
    Epoch 10/10
    42000/42000 [==============================] - 3s 67us/step - loss: 0.1060 - acc: 0.9693
    




    <keras.callbacks.History at 0xe779390>



## Predict test labels


```python
pred_classes = model.predict_classes(test_images.values)
```


```python
pred_classes
```




    array([2, 0, 9, ..., 3, 9, 2], dtype=int64)




```python
pred = pd.DataFrame({'ImageId': range(1, len(pred_classes)+1), 'Label': pred_classes})
```


```python
pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27995</th>
      <td>27996</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>27997</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>27998</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>27999</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>28000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred.to_csv('subm06.csv', index=False)
```
