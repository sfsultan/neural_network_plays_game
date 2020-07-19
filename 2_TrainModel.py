import tensorflow as tf
from tensorflow import keras
import numpy as np
from random import shuffle
import os
import threading
import datetime
from models import minigooglenet_functional, keras_functional, keras_squential, nNet

FILE_I_END = 1

WIDTH = 270
HEIGHT = 480

MODEL_NAME = 'my_model'
LOAD_PREV_MODEL = False

EPOCHS = 3

if LOAD_PREV_MODEL and os.path.isdir(MODEL_NAME):
  print("Loaded a previous model named: ", MODEL_NAME)
  model = keras.models.load_model(MODEL_NAME)

else:
  print("Created a new model.")

  model = nNet(WIDTH, HEIGHT, 3, 1)
  # model = minigooglenet_functional(WIDTH, HEIGHT, 3, 1)

  # model = keras_functional(WIDTH, HEIGHT, 3, 1)
  # model.compile(
  #   loss=keras.losses.MeanSquaredError(),
  #   optimizer=keras.optimizers.RMSprop(),
  #   metrics=["accuracy"],
  # )





model.summary()

current_dir = os.path.dirname(os.path.realpath(__file__))

for epoch in range(EPOCHS):
  data_order = [i for i in range(1,FILE_I_END+1)]
  shuffle(data_order)
  for count,i in enumerate(data_order):
    try:
      file_name = current_dir + '\\training-data\\training_data-{}.npy'.format(i)
      # full file info
      train_data = np.load(file_name)
      # print("=========File Fetched=============")
      # print('training_data-{}.npy'.format(i),len(train_data))
      # print(train_data.shape)
      # print(train_data[0,1])

      train = train_data[:-300]
      test = train_data[-300:]
      # print("========train/test===========")
      # print(train.shape)
      # print(test.shape)

      X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
      Y = np.array([i[1] for i in train])
      # print("========X/Y===========")
      # print(X.shape)
      # print(Y.shape)

      test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
      test_y = np.array([i[1] for i in test])
      # print("========test_x/test_y===========")
      # print(test_x.shape)
      # print(test_y.shape)

      # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

      model.fit(X, Y, epochs=5, validation_data=(test_x, test_y))

      test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

      print('\nTest accuracy: ', test_acc)
      print('Test loss: ', test_loss)
      print('EPOCH: ', epoch)

      if count%10 == 0:
        print('SAVING MODEL!', '\n\n')
        model.save(MODEL_NAME)

    except Exception as e:
        print(str(e))
