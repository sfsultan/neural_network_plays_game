import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
import win32api
import threading

starting_value = 1

current_dir = os.path.dirname(os.path.realpath(__file__))

while True:
    file_name = current_dir + '\\training-data\\training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)

        break

def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    stream = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = True
    print('STARTING!!!')
    print('Pausing... Press "T" to pause and unpause.')

    while(True):

        if not paused:
            screen = grab_screen(region=(0,210, 550, 980))
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (270,480))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            keys = key_check()

            output = win32api.GetKeyState(0x01)
            training_data.append([screen,output])


            print("TData Len: ", len(training_data), end='')
            # print("TData Len: ", len(training_data), " Stream Len: ", len(stream), end='')
            print('\r', end='')

            if output < 0:
              output = 1
              time.sleep(0.1)
              # training_data.extend(stream[-100:][:])
              # stream = []
              # print("Left Mouse Button Pressed.")
            else:
              output = 0
              # print("Mouse not clicked")



            # last_time = time.time()
            # cv2.imshow('window',cv2.resize(screen,(550,980)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


            if len(training_data) == 2000:

                file_save_thread = threading.Thread(target=save_file, args=(training_data, file_name))
                file_save_thread.start()

                training_data = []
                file_name = current_dir + '\\training-data\\training_data-{}.npy'.format(starting_value)
                starting_value += 1


        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Unpaused... Press "T" to pause and unpause.')
                time.sleep(1)
            else:
                print('Pausing... Press "T" to pause and unpause.')
                paused = True
                time.sleep(1)


def save_file(training_data, filename):
    np.save(filename, training_data)
    print('Training file saved.')
    print(file_name)



main(file_name, starting_value)
