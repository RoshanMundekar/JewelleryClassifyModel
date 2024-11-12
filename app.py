
from flask import Flask, render_template, request, session, url_for, redirect,flash,jsonify
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
from tensorflow import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model1 = load_model('model/model/VGGSKin.hp5')



UPLOAD_FOLDER = 'static/upload/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'random string'




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model',methods=['POST','GET'])
def model():
    if request.method == "POST":
       print("===============================================")
       file = request.files['file']
       filename = secure_filename(file.filename)
       print(filename)
       file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
       img = cv2.imread("static/upload/"+str(filename))
       image_size=224
       path="static/upload/"+"//"+str(filename)
       img = image.load_img(path, target_size=(image_size, image_size))
       x = image.img_to_array(img)
       print(type(x))
       img_4d=x.reshape(1,224,224,3)
       predictions = model1.predict(img_4d)
       print(predictions)
       pred=np.argmax(predictions[0])
       print("===============================================")
       print(pred)
       print("===============================================")
       # Dictionary for predictions
       dict1 = {0: 'BRACELET', 1: 'EARRINGS', 2: 'NECKLACE', 3: 'RINGS', 4: 'WRISTWATCH'}
       op=dict1[pred]
       print(op)
       
       if op == "BRACELET":
            final_prediction = "BRACELET"
       elif op == "EARRINGS":
            final_prediction = "EARRINGS"
       elif op == "NECKLACE":
            final_prediction = "NECKLACE"
       elif op == "NECKLACE":
            final_prediction = "NECKLACE"
       elif op == "RINGS":
            final_prediction = "RINGS"
       elif op == "WRISTWATCH":
            final_prediction = "WRISTWATCH"
       else:
            final_prediction = "Unknown"
        
       message = "The given jewelry is classified as " + final_prediction
        
       user_dict = {
            "message": message,
            "final_prediction": final_prediction,
            "Image_path": path
        }
        
       return jsonify(user_dict)
 
    return render_template('model.html')



if __name__ == '__main__':
    app.run(debug=True)
    # app.run('0.0.0.0')