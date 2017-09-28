import csv
import cv2
import numpy as np

lines = []

datafolder = 'testdata'
imgfolder = 'IMG'

with open('{}/driving_log.csv'.format(datafolder)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '{}/{}/'.format(datafolder, imgfolder) + filename
    image = cv2.imread(current_path)
    images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)
print(y_train.shape)

# img = X_train[10]
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Cropping2D, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D

input_shape = X_train.shape[1:]
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
exit()
