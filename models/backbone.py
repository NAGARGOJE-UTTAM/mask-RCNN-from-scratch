import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Input,ReLU,MaxPooling2D
from tensorflow.keras.models import Model



# input layer so the image can move through the input like a entry point of neural network
inputs =Input( shape=(224,224,3)) # every img enters will have the shapes like height, width, channel(R G B)

# output= conv1(image)

x=Conv2D(
    kernel_size=(3,3),
    filters=64,
    strides=1,
    padding="same"
)(inputs)

# use the reLu (rectified linear unit ) it is used to remove the neagtive values from the input of the conv2d
x = ReLU()(x)

# maxpolling 

x= MaxPooling2D(pool_size=(2,2))(x)

model = Model(inputs=inputs, outputs=x)

model.summary()
