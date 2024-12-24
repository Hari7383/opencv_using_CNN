import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load the saved model
model = load_model('model.h5')

# Path to the new image
new_image_path = 'RANE-Dataset/RANE/Test/SUZUKI_Black/SUZUKI_black.jpg'

# Preprocess the new image
image = load_img(new_image_path, target_size=(224, 224)) 
image_array = img_to_array(image)  
image_array = image_array / 255.0 
image_array = np.expand_dims(image_array, axis=0)  

# Make predictions
predictions = model.predict(image_array)
predicted_class = np.argmax(predictions, axis=1) 

# Output the results
print(f"Predicted probabilities: {predictions}")
print(f"Predicted class: {predicted_class[0]}")
class_labels = open('labels1.txt', 'r').readline()
print(f"Predicted label: {class_labels[predicted_class[0]]}")