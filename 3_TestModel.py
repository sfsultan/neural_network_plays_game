import tensorflow as tf
from tensorflow import keras
import numpy as np
from random import shuffle
import os
from grabscreen import grab_screen
import cv2
from models import minigooglenet_functional, keras_functional, keras_squential
import numpy as np

import win32api, win32con


WIDTH = 270
HEIGHT = 480

MODEL_NAME = 'my_model'

CLICK_THRESHOLD = 0.7

try:
  model = keras.models.load_model(MODEL_NAME)
except:
  print("Unable to load the model")
  exit()


model.summary()


def main():

    print('STARTING!!!')
    prev_prediction = 0
    max_prediction = 0
    min_prediction = 0
    print("Prediction :: ")

    while(True):

      screen = grab_screen(region=(0,210, 550, 980))
      # resize to something a bit more acceptable for a CNN
      screen = cv2.resize(screen, (270,480))
      # run a color convert:
      screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)


      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

      # print(screen.shape)
      # print(screen.reshape(1,HEIGHT, WIDTH, 3).shape)
      prediction = model.predict(screen.reshape(1, WIDTH, HEIGHT, 3))
      pscalar = prediction[0].item()

      if pscalar > CLICK_THRESHOLD:
          win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,100,500,0,0)
          win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,100,500,0,0)


      if pscalar != prev_prediction:
        prev_prediction = pscalar

        if pscalar > max_prediction:
          max_prediction = pscalar

        if pscalar < min_prediction:
          min_prediction = pscalar

        print(pscalar, ' - ' , min_prediction , ' - ' , max_prediction, end="", flush=True)
        print('\r', end='')


      cv2.imshow('window',cv2.resize(screen,(550,870)))


main()
