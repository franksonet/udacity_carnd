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
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '{}/{}/'.format(datafolder, imgfolder) + filename
        image = cv2.imread(current_path)
        images.append(image)

        measurement = float(line[3])
        if i == 1:
            measurement += 0.2
        if i == 2:
            measurement -= 0.2
        measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print(X_train.shape)
print(y_train.shape)

# img = X_train[10]
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

input_shape = X_train.shape[1:]
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
exit()
