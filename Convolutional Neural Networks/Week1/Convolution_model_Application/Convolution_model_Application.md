# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Important Note on Submission to the AutoGrader

Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:

1. You have not added any _extra_ `print` statement(s) in the assignment.
2. You have not added any _extra_ code cell(s) in the assignment.
3. You have not changed any of the function parameters.
4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
5. You are not changing the assignment code where it is not required, like creating _extra_ variables.

If you do any of the following, you will get something like, `Grader not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/convolutional-neural-networks/supplement/DS4yP/h-ow-to-refresh-your-workspace).

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 124
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_7_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            tf.keras.Input(shape=(64 , 64 ,3)),
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tfl.ZeroPadding2D(padding=3), # As import tensorflow.keras.layers as tfl
            ## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(filters=32,kernel_size=7,strides=1),
            ## BatchNormalization for axis 3
            tfl.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001),
            ## ReLU
            tfl.ReLU(),
            ## Max Pooling 2D with default parameters
            tfl.MaxPool2D(),
            ## Flatten layer
            tfl.Flatten(),
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(1,activation='sigmoid')
        ])
    
    return model
```


```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d (ZeroPadding2 (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu (ReLU)                 (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 32768)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 102ms/step - loss: 1.1256 - accuracy: 0.7250
    Epoch 2/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.2101 - accuracy: 0.9033
    Epoch 3/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1681 - accuracy: 0.9317
    Epoch 4/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1323 - accuracy: 0.9483
    Epoch 5/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0832 - accuracy: 0.9700
    Epoch 6/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.0795 - accuracy: 0.9767
    Epoch 7/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0714 - accuracy: 0.9733
    Epoch 8/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.2123 - accuracy: 0.9250
    Epoch 9/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.2147 - accuracy: 0.9183
    Epoch 10/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.0814 - accuracy: 0.9750





    <tensorflow.python.keras.callbacks.History at 0x7f4d28d0b310>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 31ms/step - loss: 0.1529 - accuracy: 0.9467





    [0.15293952822685242, 0.9466666579246521]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 4



![png](output_28_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(filters= 8. , kernel_size=4 , padding='same',strides=1)(input_img)
    ## RELU
    A1 = tfl.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tfl.MaxPool2D(pool_size=8, strides=8, padding='SAME')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tfl.Conv2D(filters= 16. , kernel_size=2 , padding='same',strides=1)(P1)
    ## RELU
    A2 =  tfl.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tfl.MaxPool2D(pool_size=4, strides=4, padding='SAME')(A2)
    ## FLATTEN
    F = tfl.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tfl.Dense(units= 6 , activation='softmax')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 64, 8)         392       
    _________________________________________________________________
    re_lu_1 (ReLU)               (None, 64, 64, 8)         0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 8, 8, 16)          528       
    _________________________________________________________________
    re_lu_2 (ReLU)               (None, 8, 8, 16)          0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 2, 2, 16)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 64)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3428 - accuracy: 0.9009 - val_loss: 0.5218 - val_accuracy: 0.8000
    Epoch 2/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3401 - accuracy: 0.9028 - val_loss: 0.5197 - val_accuracy: 0.8000
    Epoch 3/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3372 - accuracy: 0.9009 - val_loss: 0.5142 - val_accuracy: 0.8000
    Epoch 4/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3347 - accuracy: 0.9019 - val_loss: 0.5121 - val_accuracy: 0.8083
    Epoch 5/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3317 - accuracy: 0.9037 - val_loss: 0.5072 - val_accuracy: 0.8083
    Epoch 6/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.3290 - accuracy: 0.9065 - val_loss: 0.5052 - val_accuracy: 0.8083
    Epoch 7/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3267 - accuracy: 0.9046 - val_loss: 0.5017 - val_accuracy: 0.8000
    Epoch 8/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3241 - accuracy: 0.9065 - val_loss: 0.4997 - val_accuracy: 0.8000
    Epoch 9/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3213 - accuracy: 0.9065 - val_loss: 0.4964 - val_accuracy: 0.8083
    Epoch 10/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3187 - accuracy: 0.9046 - val_loss: 0.4940 - val_accuracy: 0.8083
    Epoch 11/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3163 - accuracy: 0.9046 - val_loss: 0.4914 - val_accuracy: 0.8083
    Epoch 12/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.3137 - accuracy: 0.9046 - val_loss: 0.4905 - val_accuracy: 0.8167
    Epoch 13/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3107 - accuracy: 0.9065 - val_loss: 0.4877 - val_accuracy: 0.8167
    Epoch 14/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3075 - accuracy: 0.9093 - val_loss: 0.4864 - val_accuracy: 0.8167
    Epoch 15/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.3050 - accuracy: 0.9102 - val_loss: 0.4834 - val_accuracy: 0.8167
    Epoch 16/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.3019 - accuracy: 0.9120 - val_loss: 0.4818 - val_accuracy: 0.8250
    Epoch 17/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2994 - accuracy: 0.9130 - val_loss: 0.4795 - val_accuracy: 0.8167
    Epoch 18/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2968 - accuracy: 0.9120 - val_loss: 0.4782 - val_accuracy: 0.8250
    Epoch 19/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.2943 - accuracy: 0.9130 - val_loss: 0.4748 - val_accuracy: 0.8333
    Epoch 20/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2913 - accuracy: 0.9130 - val_loss: 0.4729 - val_accuracy: 0.8333
    Epoch 21/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2884 - accuracy: 0.9130 - val_loss: 0.4703 - val_accuracy: 0.8333
    Epoch 22/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2856 - accuracy: 0.9130 - val_loss: 0.4690 - val_accuracy: 0.8333
    Epoch 23/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.2832 - accuracy: 0.9130 - val_loss: 0.4660 - val_accuracy: 0.8333
    Epoch 24/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2802 - accuracy: 0.9139 - val_loss: 0.4640 - val_accuracy: 0.8333
    Epoch 25/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2774 - accuracy: 0.9148 - val_loss: 0.4610 - val_accuracy: 0.8333
    Epoch 26/100
    17/17 [==============================] - 2s 108ms/step - loss: 0.2749 - accuracy: 0.9157 - val_loss: 0.4595 - val_accuracy: 0.8333
    Epoch 27/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2723 - accuracy: 0.9157 - val_loss: 0.4560 - val_accuracy: 0.8333
    Epoch 28/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2695 - accuracy: 0.9167 - val_loss: 0.4543 - val_accuracy: 0.8417
    Epoch 29/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2667 - accuracy: 0.9176 - val_loss: 0.4514 - val_accuracy: 0.8417
    Epoch 30/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2642 - accuracy: 0.9167 - val_loss: 0.4496 - val_accuracy: 0.8417
    Epoch 31/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.2615 - accuracy: 0.9194 - val_loss: 0.4466 - val_accuracy: 0.8417
    Epoch 32/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.2589 - accuracy: 0.9194 - val_loss: 0.4438 - val_accuracy: 0.8417
    Epoch 33/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2561 - accuracy: 0.9213 - val_loss: 0.4413 - val_accuracy: 0.8500
    Epoch 34/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2537 - accuracy: 0.9213 - val_loss: 0.4390 - val_accuracy: 0.8500
    Epoch 35/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2516 - accuracy: 0.9222 - val_loss: 0.4374 - val_accuracy: 0.8583
    Epoch 36/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2492 - accuracy: 0.9231 - val_loss: 0.4347 - val_accuracy: 0.8583
    Epoch 37/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2471 - accuracy: 0.9231 - val_loss: 0.4340 - val_accuracy: 0.8583
    Epoch 38/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.2450 - accuracy: 0.9231 - val_loss: 0.4314 - val_accuracy: 0.8583
    Epoch 39/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2426 - accuracy: 0.9250 - val_loss: 0.4298 - val_accuracy: 0.8583
    Epoch 40/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2405 - accuracy: 0.9250 - val_loss: 0.4281 - val_accuracy: 0.8583
    Epoch 41/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2385 - accuracy: 0.9250 - val_loss: 0.4268 - val_accuracy: 0.8583
    Epoch 42/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2364 - accuracy: 0.9259 - val_loss: 0.4246 - val_accuracy: 0.8583
    Epoch 43/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2343 - accuracy: 0.9278 - val_loss: 0.4230 - val_accuracy: 0.8583
    Epoch 44/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2322 - accuracy: 0.9259 - val_loss: 0.4214 - val_accuracy: 0.8667
    Epoch 45/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2305 - accuracy: 0.9296 - val_loss: 0.4202 - val_accuracy: 0.8667
    Epoch 46/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2284 - accuracy: 0.9296 - val_loss: 0.4181 - val_accuracy: 0.8667
    Epoch 47/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2264 - accuracy: 0.9333 - val_loss: 0.4158 - val_accuracy: 0.8667
    Epoch 48/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2244 - accuracy: 0.9361 - val_loss: 0.4147 - val_accuracy: 0.8667
    Epoch 49/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2223 - accuracy: 0.9352 - val_loss: 0.4123 - val_accuracy: 0.8667
    Epoch 50/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.2207 - accuracy: 0.9361 - val_loss: 0.4111 - val_accuracy: 0.8750
    Epoch 51/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2187 - accuracy: 0.9370 - val_loss: 0.4100 - val_accuracy: 0.8750
    Epoch 52/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.2166 - accuracy: 0.9370 - val_loss: 0.4075 - val_accuracy: 0.8750
    Epoch 53/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2148 - accuracy: 0.9380 - val_loss: 0.4060 - val_accuracy: 0.8750
    Epoch 54/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2127 - accuracy: 0.9389 - val_loss: 0.4039 - val_accuracy: 0.8917
    Epoch 55/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2110 - accuracy: 0.9398 - val_loss: 0.4037 - val_accuracy: 0.8917
    Epoch 56/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2092 - accuracy: 0.9389 - val_loss: 0.4013 - val_accuracy: 0.9000
    Epoch 57/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2073 - accuracy: 0.9426 - val_loss: 0.4001 - val_accuracy: 0.8917
    Epoch 58/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2060 - accuracy: 0.9417 - val_loss: 0.4001 - val_accuracy: 0.8917
    Epoch 59/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2039 - accuracy: 0.9417 - val_loss: 0.3973 - val_accuracy: 0.8917
    Epoch 60/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.2024 - accuracy: 0.9435 - val_loss: 0.3967 - val_accuracy: 0.8917
    Epoch 61/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.2003 - accuracy: 0.9435 - val_loss: 0.3940 - val_accuracy: 0.8917
    Epoch 62/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1991 - accuracy: 0.9444 - val_loss: 0.3942 - val_accuracy: 0.8917
    Epoch 63/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1974 - accuracy: 0.9463 - val_loss: 0.3921 - val_accuracy: 0.8917
    Epoch 64/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1960 - accuracy: 0.9444 - val_loss: 0.3923 - val_accuracy: 0.8917
    Epoch 65/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1940 - accuracy: 0.9481 - val_loss: 0.3902 - val_accuracy: 0.8917
    Epoch 66/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1925 - accuracy: 0.9472 - val_loss: 0.3892 - val_accuracy: 0.8917
    Epoch 67/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.1910 - accuracy: 0.9481 - val_loss: 0.3882 - val_accuracy: 0.8917
    Epoch 68/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1894 - accuracy: 0.9481 - val_loss: 0.3876 - val_accuracy: 0.8917
    Epoch 69/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.1879 - accuracy: 0.9491 - val_loss: 0.3863 - val_accuracy: 0.8917
    Epoch 70/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1864 - accuracy: 0.9509 - val_loss: 0.3856 - val_accuracy: 0.8917
    Epoch 71/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.1848 - accuracy: 0.9519 - val_loss: 0.3849 - val_accuracy: 0.8917
    Epoch 72/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1838 - accuracy: 0.9500 - val_loss: 0.3854 - val_accuracy: 0.8917
    Epoch 73/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1820 - accuracy: 0.9519 - val_loss: 0.3828 - val_accuracy: 0.8917
    Epoch 74/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1807 - accuracy: 0.9519 - val_loss: 0.3830 - val_accuracy: 0.8917
    Epoch 75/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.1791 - accuracy: 0.9537 - val_loss: 0.3820 - val_accuracy: 0.8917
    Epoch 76/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1775 - accuracy: 0.9528 - val_loss: 0.3807 - val_accuracy: 0.8917
    Epoch 77/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1761 - accuracy: 0.9537 - val_loss: 0.3803 - val_accuracy: 0.8917
    Epoch 78/100
    17/17 [==============================] - 2s 108ms/step - loss: 0.1747 - accuracy: 0.9556 - val_loss: 0.3803 - val_accuracy: 0.8917
    Epoch 79/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1734 - accuracy: 0.9546 - val_loss: 0.3803 - val_accuracy: 0.8917
    Epoch 80/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1721 - accuracy: 0.9556 - val_loss: 0.3800 - val_accuracy: 0.8917
    Epoch 81/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1708 - accuracy: 0.9574 - val_loss: 0.3792 - val_accuracy: 0.8917
    Epoch 82/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.1695 - accuracy: 0.9583 - val_loss: 0.3785 - val_accuracy: 0.8917
    Epoch 83/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.1682 - accuracy: 0.9583 - val_loss: 0.3777 - val_accuracy: 0.8917
    Epoch 84/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1667 - accuracy: 0.9583 - val_loss: 0.3773 - val_accuracy: 0.8917
    Epoch 85/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1653 - accuracy: 0.9583 - val_loss: 0.3765 - val_accuracy: 0.8917
    Epoch 86/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1639 - accuracy: 0.9593 - val_loss: 0.3754 - val_accuracy: 0.8917
    Epoch 87/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.1626 - accuracy: 0.9593 - val_loss: 0.3744 - val_accuracy: 0.8917
    Epoch 88/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1615 - accuracy: 0.9583 - val_loss: 0.3737 - val_accuracy: 0.8917
    Epoch 89/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.1600 - accuracy: 0.9593 - val_loss: 0.3730 - val_accuracy: 0.8917
    Epoch 90/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.1589 - accuracy: 0.9593 - val_loss: 0.3734 - val_accuracy: 0.8917
    Epoch 91/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1580 - accuracy: 0.9593 - val_loss: 0.3722 - val_accuracy: 0.8917
    Epoch 92/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1569 - accuracy: 0.9583 - val_loss: 0.3723 - val_accuracy: 0.8917
    Epoch 93/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.1554 - accuracy: 0.9611 - val_loss: 0.3703 - val_accuracy: 0.8917
    Epoch 94/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.1544 - accuracy: 0.9630 - val_loss: 0.3711 - val_accuracy: 0.8917
    Epoch 95/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1530 - accuracy: 0.9611 - val_loss: 0.3690 - val_accuracy: 0.8917
    Epoch 96/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1520 - accuracy: 0.9620 - val_loss: 0.3691 - val_accuracy: 0.8917
    Epoch 97/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1507 - accuracy: 0.9593 - val_loss: 0.3681 - val_accuracy: 0.8917
    Epoch 98/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1494 - accuracy: 0.9630 - val_loss: 0.3674 - val_accuracy: 0.8917
    Epoch 99/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1483 - accuracy: 0.9630 - val_loss: 0.3673 - val_accuracy: 0.8917
    Epoch 100/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.1471 - accuracy: 0.9620 - val_loss: 0.3676 - val_accuracy: 0.8917


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [0.3427838385105133,
      0.34007129073143005,
      0.33718952536582947,
      0.33468377590179443,
      0.33167046308517456,
      0.32898184657096863,
      0.326732873916626,
      0.3240598142147064,
      0.3212774097919464,
      0.3187410831451416,
      0.3163006901741028,
      0.31366947293281555,
      0.3106764256954193,
      0.30752065777778625,
      0.3049520254135132,
      0.3019322156906128,
      0.2994450628757477,
      0.29675915837287903,
      0.29429054260253906,
      0.2913159132003784,
      0.288369745016098,
      0.28558096289634705,
      0.28315016627311707,
      0.28017017245292664,
      0.27741342782974243,
      0.27488547563552856,
      0.27230989933013916,
      0.2695438265800476,
      0.26665255427360535,
      0.2642272412776947,
      0.2614727020263672,
      0.25892573595046997,
      0.25605881214141846,
      0.2537212669849396,
      0.25155043601989746,
      0.24919931590557098,
      0.24712002277374268,
      0.24496740102767944,
      0.2426254004240036,
      0.24050553143024445,
      0.23845811188220978,
      0.23639777302742004,
      0.2342599630355835,
      0.2322351038455963,
      0.23049090802669525,
      0.2284138798713684,
      0.22640739381313324,
      0.22440776228904724,
      0.2223416417837143,
      0.2207280993461609,
      0.2187095731496811,
      0.2166401892900467,
      0.2147550731897354,
      0.2126837521791458,
      0.21103887259960175,
      0.20917299389839172,
      0.20733021199703217,
      0.20603393018245697,
      0.20386338233947754,
      0.2024434208869934,
      0.20032241940498352,
      0.19911201298236847,
      0.19744542241096497,
      0.19597794115543365,
      0.19398413598537445,
      0.1925187110900879,
      0.19100205600261688,
      0.18938232958316803,
      0.1878572255373001,
      0.18642066419124603,
      0.1848319172859192,
      0.18376228213310242,
      0.18204407393932343,
      0.1807437688112259,
      0.1791195124387741,
      0.17754508554935455,
      0.17610204219818115,
      0.1746722161769867,
      0.1734238862991333,
      0.17212994396686554,
      0.17080771923065186,
      0.16953356564044952,
      0.16819432377815247,
      0.16671398282051086,
      0.16526946425437927,
      0.16394448280334473,
      0.16257375478744507,
      0.16147327423095703,
      0.16001436114311218,
      0.15889208018779755,
      0.15798534452915192,
      0.15685969591140747,
      0.15537787973880768,
      0.15436694025993347,
      0.15301112830638885,
      0.15197470784187317,
      0.1506846845149994,
      0.14943554997444153,
      0.14827412366867065,
      0.14713162183761597],
     'accuracy': [0.9009259343147278,
      0.9027777910232544,
      0.9009259343147278,
      0.9018518328666687,
      0.9037036895751953,
      0.9064815044403076,
      0.904629647731781,
      0.9064815044403076,
      0.9064815044403076,
      0.904629647731781,
      0.904629647731781,
      0.904629647731781,
      0.9064815044403076,
      0.9092592597007751,
      0.9101851582527161,
      0.9120370149612427,
      0.9129629731178284,
      0.9120370149612427,
      0.9129629731178284,
      0.9129629731178284,
      0.9129629731178284,
      0.9129629731178284,
      0.9129629731178284,
      0.9138888716697693,
      0.914814829826355,
      0.9157407283782959,
      0.9157407283782959,
      0.9166666865348816,
      0.9175925850868225,
      0.9166666865348816,
      0.9194444417953491,
      0.9194444417953491,
      0.9212962985038757,
      0.9212962985038757,
      0.9222221970558167,
      0.9231481552124023,
      0.9231481552124023,
      0.9231481552124023,
      0.925000011920929,
      0.925000011920929,
      0.925000011920929,
      0.9259259104728699,
      0.9277777671813965,
      0.9259259104728699,
      0.9296296238899231,
      0.9296296238899231,
      0.9333333373069763,
      0.9361110925674438,
      0.9351851940155029,
      0.9361110925674438,
      0.9370370507240295,
      0.9370370507240295,
      0.9379629492759705,
      0.9388889074325562,
      0.9398148059844971,
      0.9388889074325562,
      0.9425926208496094,
      0.9416666626930237,
      0.9416666626930237,
      0.9435185194015503,
      0.9435185194015503,
      0.9444444179534912,
      0.9462962746620178,
      0.9444444179534912,
      0.9481481313705444,
      0.9472222328186035,
      0.9481481313705444,
      0.9481481313705444,
      0.9490740895271301,
      0.9509259462356567,
      0.9518518447875977,
      0.949999988079071,
      0.9518518447875977,
      0.9518518447875977,
      0.9537037014961243,
      0.9527778029441833,
      0.9537037014961243,
      0.9555555582046509,
      0.9546296000480652,
      0.9555555582046509,
      0.9574074149131775,
      0.9583333134651184,
      0.9583333134651184,
      0.9583333134651184,
      0.9583333134651184,
      0.9592592716217041,
      0.9592592716217041,
      0.9583333134651184,
      0.9592592716217041,
      0.9592592716217041,
      0.9592592716217041,
      0.9583333134651184,
      0.9611111283302307,
      0.9629629850387573,
      0.9611111283302307,
      0.9620370268821716,
      0.9592592716217041,
      0.9629629850387573,
      0.9629629850387573,
      0.9620370268821716],
     'val_loss': [0.5217766165733337,
      0.5196832418441772,
      0.5142131447792053,
      0.512135922908783,
      0.507163941860199,
      0.5052352547645569,
      0.5017001032829285,
      0.49974605441093445,
      0.4964495599269867,
      0.49404391646385193,
      0.4914250373840332,
      0.49050047993659973,
      0.4876975417137146,
      0.48639315366744995,
      0.48340904712677,
      0.48183801770210266,
      0.47951024770736694,
      0.47823619842529297,
      0.47480201721191406,
      0.4728512465953827,
      0.4703201949596405,
      0.4689841568470001,
      0.46597322821617126,
      0.4640311598777771,
      0.46101850271224976,
      0.45948168635368347,
      0.4559946060180664,
      0.4543200731277466,
      0.45142078399658203,
      0.4495662748813629,
      0.44657230377197266,
      0.44377997517585754,
      0.4413449466228485,
      0.4390482008457184,
      0.4373966157436371,
      0.4346586763858795,
      0.4340110719203949,
      0.43135884404182434,
      0.429792582988739,
      0.4280565679073334,
      0.4267653226852417,
      0.42455095052719116,
      0.4229772388935089,
      0.421378493309021,
      0.42018845677375793,
      0.4180513918399811,
      0.41577354073524475,
      0.41469866037368774,
      0.4123111069202423,
      0.411091685295105,
      0.40998518466949463,
      0.4075230360031128,
      0.406004399061203,
      0.40394821763038635,
      0.4037364721298218,
      0.4013144075870514,
      0.4000915586948395,
      0.4001079201698303,
      0.3973498046398163,
      0.39668089151382446,
      0.39395856857299805,
      0.3942229449748993,
      0.39205819368362427,
      0.39230072498321533,
      0.39015433192253113,
      0.38919562101364136,
      0.38821956515312195,
      0.3876464068889618,
      0.38634589314460754,
      0.3856070935726166,
      0.3848569989204407,
      0.38537535071372986,
      0.382759153842926,
      0.3830350339412689,
      0.38198843598365784,
      0.38071349263191223,
      0.3803069293498993,
      0.38029879331588745,
      0.3802728056907654,
      0.37999168038368225,
      0.3792051672935486,
      0.37845584750175476,
      0.37773317098617554,
      0.3772912323474884,
      0.37649980187416077,
      0.3753819763660431,
      0.37437692284584045,
      0.37368425726890564,
      0.3729625642299652,
      0.37339121103286743,
      0.3721717298030853,
      0.3723289370536804,
      0.3702546954154968,
      0.37109750509262085,
      0.3689957559108734,
      0.3690964877605438,
      0.36808714270591736,
      0.36741960048675537,
      0.36732831597328186,
      0.367613285779953],
     'val_accuracy': [0.800000011920929,
      0.800000011920929,
      0.800000011920929,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.800000011920929,
      0.800000011920929,
      0.8083333373069763,
      0.8083333373069763,
      0.8083333373069763,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.8166666626930237,
      0.824999988079071,
      0.8166666626930237,
      0.824999988079071,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184,
      0.8333333134651184,
      0.8416666388511658,
      0.8416666388511658,
      0.8416666388511658,
      0.8416666388511658,
      0.8416666388511658,
      0.8500000238418579,
      0.8500000238418579,
      0.8583333492279053,
      0.8583333492279053,
      0.8583333492279053,
      0.8583333492279053,
      0.8583333492279053,
      0.8583333492279053,
      0.8583333492279053,
      0.8583333492279053,
      0.8583333492279053,
      0.8666666746139526,
      0.8666666746139526,
      0.8666666746139526,
      0.8666666746139526,
      0.8666666746139526,
      0.8666666746139526,
      0.875,
      0.875,
      0.875,
      0.875,
      0.8916666507720947,
      0.8916666507720947,
      0.8999999761581421,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947,
      0.8916666507720947]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![png](output_41_1.png)



![png](output_41_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
