import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

model = load_model('keras_model.h5')
class_labels = open("labels.txt", "r").readlines()

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = img_to_array(frame_resized)
    frame_array = frame_array / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()


bgfg = cv2.createBackgroundSubtractorMOG2()

print("Press 'q' to quit the application.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    fgmask = bgfg.apply(frame)
    
    _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    
    if cv2.countNonZero(thresh) > 1500:  
        processed_frame = preprocess_frame(frame)
        predictions = model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]
        predicted_confidence = predictions[0][predicted_class]
        threshold = 0.7
        if predicted_confidence < threshold:
            predicted_label = "No object here"

    else:
        predicted_label = "No object here"
    cv2.putText(frame, f"Prediction: {predicted_label} ({predicted_confidence:.2f})", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Product Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
