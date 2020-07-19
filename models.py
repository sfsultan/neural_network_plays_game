# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow import keras



def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
  # define a CONV => BN => RELU pattern
  x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
  x = BatchNormalization(axis=chanDim)(x)
  x = Activation("relu")(x)
  # return the block
  return x

def downsample_module(x, K, chanDim):
  # define the CONV module and POOL, then concatenate
  # across the channel dimensions
  conv_3x3 = conv_module(x, K, 3, 3, (2, 2), chanDim,
    padding="valid")
  pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = concatenate([conv_3x3, pool], axis=chanDim)
  # return the block
  return x


def inception_module(x, numK1x1, numK3x3, chanDim):
  # define two CONV modules, then concatenate across the
  # channel dimension
  conv_1x1 = conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
  conv_3x3 = conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
  x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
  # return the block
  return x


def minigooglenet_functional(width, height, depth, classes):

  # initialize the input shape to be "channels last" and the
  # channels dimension itself
  inputShape = (width, height, depth)
  chanDim = -1
  # define the model input and first CONV module
  inputs = Input(shape=inputShape)
  x = conv_module(inputs, 96, 3, 3, (1, 1), chanDim)
  # two Inception modules followed by a downsample module
  # x = inception_module(x, 32, 32, chanDim)
  # x = inception_module(x, 32, 48, chanDim)
  # x = downsample_module(x, 80, chanDim)
  # four Inception modules followed by a downsample module
  x = inception_module(x, 64, 32, chanDim)
  # x = inception_module(x, 32, 64, chanDim)
  # x = inception_module(x, 32, 16, chanDim)
  # x = inception_module(x, 16, 32, chanDim)
  x = downsample_module(x, 32, chanDim)
  # two Inception modules followed by global POOL and dropout
  x = inception_module(x, 128, 64, chanDim)
  # x = inception_module(x, 64, 128, chanDim)
  x = AveragePooling2D((7, 7))(x)
  # softmax classifier
  x = Flatten()(x)
  x = Dense(classes)(x)
  x = Activation("softmax")(x)
  # create the model
  model = Model(inputs, x, name="minigooglenet")
  # return the constructed network architecture
  model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
  return model








def keras_functional(width, height, depth, classes):
  inputs = Input(shape=(width, height, depth))
  dense = Dense(64, activation="relu")
  x = dense(inputs)

  x = Dense(64, activation="relu")(x)
  outputs = Dense(classes)(x)

  model = Model(inputs=inputs, outputs=outputs, name="mnist_model")

  return model




def keras_squential(width, height, depth, classes):
  model = Sequential([
      Flatten(input_shape=(width, height, depth), name='input'),
      Dense(64, activation='relu', name='dense1'),
      Dense(256, activation='relu', name='dense2'),
      Dense(128, activation='relu', name='dense3'),

  ])

  return model



def nNet(width, height, depth, classes):
  model = Sequential()
  model.add(Input(shape=(width, height, depth)))  # 250x250 RGB images
  model.add(Conv2D(16, 5, strides=2, activation="softmax"))
  model.add(MaxPooling2D(5,5))
  model.add(Conv2D(32, 3, activation="softmax"))
  model.add(MaxPooling2D(2,2))
  model.add(Conv2D(64, 3, activation="softmax"))
  model.add(MaxPooling2D(3,3))
  model.add(Dropout(.5))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(classes, name='output', activation='sigmoid'))

  # model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

  return model


