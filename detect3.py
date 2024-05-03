from flask import Flask, render_template, Response
import cv2
import threading
import numpy as np
import pickle
from keras.models import load_model
from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pandas as pd
import tensorflow as tf

app = Flask(__name__)


cap=None


import pickle
#Function to start the camera
    
def start_cameraa():
    global cap
    frameWidth= 1280         # CAMERA RESOLUTION
    frameHeight = 720
    brightness = 180
    threshold = 0.75         # PROBABLITY THRESHOLD
    font = cv2.FONT_HERSHEY_SIMPLEX


    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)



    # IMPORT THE TRANNIED MODE
    model = pickle.load(open("model_trained.p", "rb"))
 ## rb = READ BYTE

    def grayscale(img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img

    def equalize(img):
        img =cv2.equalizeHist(img)
        return img
    

    def preprocessing(img):
        img = grayscale(img)
        img = equalize(img)
        img = img/255
        return img

    def getClassName(classNo):
        if   classNo == 0: return 'Speed Limit 20 km/h'
        elif classNo == 1: return 'Speed Limit 30 km/h'
        elif classNo == 2: return 'Speed Limit 50 km/h'
        elif classNo == 3: return 'Speed Limit 60 km/h'
        elif classNo == 4: return 'Speed Limit 70 km/h'
        elif classNo == 5: return 'Speed Limit 80 km/h'
        elif classNo == 6: return 'End of Speed Limit 80 km/h'
        elif classNo == 7: return 'Speed Limit 100 km/h'
        elif classNo == 8: return 'Speed Limit 120 km/h'
        elif classNo == 9: return 'No passing'
        elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
        elif classNo == 11: return 'Right-of-way at the next intersection'
        elif classNo == 12: return 'Priority road'
        elif classNo == 13: return 'Yield'
        elif classNo == 14: return 'Stop'
        elif classNo == 15: return 'No vechiles'
        elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
        elif classNo == 17: return 'No entry'
        elif classNo == 18: return 'General caution'
        elif classNo == 19: return 'Dangerous curve to the left'
        elif classNo == 20: return 'Dangerous curve to the right'
        elif classNo == 21: return 'Double curve'
        elif classNo == 22: return 'Bumpy road'
        elif classNo == 23: return 'Slippery road'
        elif classNo == 24: return 'Road narrows on the right'
        elif classNo == 25: return 'Road work'
        elif classNo == 26: return 'Traffic signals'
        elif classNo == 27: return 'Pedestrians'
        elif classNo == 28: return 'Children crossing'
        elif classNo == 29: return 'Bicycles crossing'
        elif classNo == 30: return 'Beware of ice/snow'
        elif classNo == 31: return 'Wild animals crossing'
        elif classNo == 32: return 'End of all speed and passing limits'
        elif classNo == 33: return 'Turn right ahead'
        elif classNo == 34: return 'Turn left ahead'
        elif classNo == 35: return 'Ahead only'
        elif classNo == 36: return 'Go straight or right'
        elif classNo == 37: return 'Go straight or left'
        elif classNo == 38: return 'Keep right'
        elif classNo == 39: return 'Keep left'
        elif classNo == 40: return 'Roundabout mandatory'
        elif classNo == 41: return 'End of no passing'
        elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

    while True:
        success, imgOrignal = cap.read()
        img = np.asarray(imgOrignal)
        img = cv2.resize(img, (32, 32))
        
        # Assuming preprocessing is a function you have defined
        img = preprocessing(img)
        cv2.imshow("Processed Image", img)
        
        # Ensure img is converted to grayscale if needed and has the correct shape
        img = img.reshape(1, 32, 32, 1)  # Assuming one channel (grayscale)

        cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.max(predictions)
        
        if probabilityValue > threshold:
            # Assuming getClassName is a function that returns the class name based on the class index
            class_name = getClassName(classIndex)
            cv2.putText(imgOrignal, str(classIndex) + " " + str(class_name), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        
        cv2.imshow("Result", imgOrignal)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Release the video capture object and close all windows


# Function to stop the camera
def stop_cameraa():
    global cap
    if cap:
        cap.release()

# Function to capture frames from the camera

# Function to process the captured stream

@app.route('/')
def index():
    return render_template('sign.html')

@app.route('/start_cameraa')
def start_cameraa_route():
    camera_thread = threading.Thread(target=start_cameraa)
    start_cameraa()
    return render_template('sign.html')

@app.route('/stop_cameraa')
def stop_cameraa_route():
    stop_cameraa()
    return render_template('sign.html')

if __name__ == '__main__':
    app.run(debug=True)
