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

def load_dataset(base_path, folder_names, x_data, y_data):
    for idx, folder in enumerate(folder_names):
        images, labels = load_images_from_path(f"{base_path}/{folder}", idx)
        x_data += images
        y_data += labels

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)

# Define paths and folder names
train_base_path = 'D:/RANE/RANE-Dataset/RANE/Train'
test_base_path = 'D:/RANE/RANE-Dataset/RANE/Test'

folders = [
    'TATA_MANDO', 'TVS-Girling', 'TVS-Girling_1', 'KIT_Green', 'RANE_BLACK', 
    'RANE_Green', 'RANE_R3', 'RANE_R809', 'SUZUKI_Black', 'SUZUKI_saga'
]

# Initialize datasets
x_train, y_train = [], []
x_test, y_test = [], []

# Load datasets
load_dataset(train_base_path, folders, x_train, y_train)
load_dataset(test_base_path, folders, x_test, y_test)

# Normalize and encode data
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# Check shapes
print("x_train shape:", x_train.shape)
print("y_train_encoded shape:", y_train_encoded.shape)
print("x_test shape:", x_test.shape)
print("y_test_encoded shape:", y_test_encoded.shape)

# Build the model
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

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
hist = model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=10, epochs=100)

# Save the model
model.save('model.h5')
print("Model successfully saved")

# Plot training and validation accuracy
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