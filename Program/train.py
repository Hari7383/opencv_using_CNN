import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        if os.path.isfile(img_path):
            img = load_img(img_path, target_size=(224, 224))  # Use TensorFlow's load_img
            images.append(img_to_array(img))  # Use TensorFlow's img_to_array
            labels.append(label)
        
    return images, labels

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)

x_train = []
y_train = []
x_test = []
y_test = []

# Assuming 3 classes, use class indices 0, 1, and 2
images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Train/TATA_MANDO', 0)
x_train += images
y_train += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Train/TVS-Girling', 1)
x_train += images
y_train += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Train/TVS-Girling_1', 2)
x_train += images
y_train += labels

images, labels =  load_images_from_path('RANE-Dataset/RANE/Train/KIT_Green', 3)
x_train += images
y_train += labels

images, labels = load_images_from_path('RANE-Dataset/RANE/Train/RANE_BLACK', 4)
x_train += images
y_train += labels

images, labels = load_images_from_path('RANE-Dataset/RANE/Train/RANE_Green', 5)
x_train += images
y_train += labels

images, labels = load_images_from_path('RANE-Dataset/RANE/Train/RANE_R3', 6)
x_train += images
y_train += labels

images, labels = load_images_from_path('RANE-Dataset/RANE/Train/RANE_R809', 7)
x_train += images
y_train += labels

images, labels = load_images_from_path('RANE-Dataset/RANE/Train/SUZUKI_Black', 8)
x_train += images
y_train += labels

images, labels = load_images_from_path('RANE-Dataset/RANE/Train/SUZUKI_saga', 9)
x_train += images
y_train += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/TATA_MANDO', 0)
x_test += images
y_test += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/TVS-Girling', 1)
x_test += images
y_test += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/TVS-Girling_1', 2)
x_test += images
y_test += labels

images, labels =  load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/KIT_Green', 3)
x_test += images
y_test += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/RANE_BLACK', 4)
x_test += images
y_test += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/RANE_Green', 5)
x_test += images
y_test += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/RANE_R3', 6)
x_test += images
y_test += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/RANE_R809', 7)
x_test += images
y_test += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/SUZUKI_Black', 8)
x_test += images
y_test += labels

images, labels = load_images_from_path('D:/RANE/RANE-Dataset/RANE/Test/SUZUKI_saga', 9)
x_test += images
y_test += labels

x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# Check shapes
print("x_train shape:", x_train.shape)
print("y_train_encoded shape:", y_train_encoded.shape)
print("x_test shape:", x_test.shape)
print("y_test_encoded shape:", y_test_encoded.shape)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=10, epochs=100)

model.save('model.h5')
print("Model succfully saved")

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()