import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

samples = []

# Where to find the camera images
datafolder = 'data_1'
imgfolder = 'IMG'

with open('{}/driving_log.csv'.format(datafolder)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

angles_hist = [float(sample[3]) for sample in samples]
plt.hist(angles_hist)
# plt.show()
plt.savefig('angels_hist.jpg')
plt.clf()

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# The function to get a generator object


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = '{}/{}/'.format(datafolder, imgfolder) + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    angle = float(batch_sample[3])
                    # When on the left camera image, steering a lit bit more to the right
                    if i == 1:
                        angle += 0.2
                    # When on the right camera image, steering a lit bit more to the left
                    if i == 2:
                        angle -= 0.2
                    angles.append(angle)
            # Flip the images and angels to create more input samples to help to generalize the model
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda
from keras.layers.convolutional import Conv2D

# from keras.layers.pooling import MaxPooling2D

# Considering the total number of smaples will be timed by 6,
# I make the batch sise as 32, which means 32 * 6 = 192 image inputs for each batch
batch_size = 32

# Create two generator objects for inputs of training and validation
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

input_shape = (160, 320, 3)

# Here I implemented the nVidia pipeline to train my network
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples) / batch_size,
                                     validation_data=validation_generator,
                                     epochs=5,
                                     validation_steps=len(validation_samples) / batch_size)

model.save('model.h5')

print(history_object.history.keys())
print(history_object.history['loss'])
print(history_object.history['val_loss'])
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_graph-2.jpg')


exit()
