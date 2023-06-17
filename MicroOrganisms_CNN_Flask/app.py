import re
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import base64
from flask import Flask, render_template, url_for, request
# from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import Concat
from tensorflow.keras.models import load_model


#Loading the model
model=load_model("model2.h5",compile=False)
app=Flask(__name__)

#default home page or route
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/Prediction')
def Prediction():
    return render_template('Prediction.html')

# @app.route('/index.html')
# def home1():
#    return render_template("index.html")

# @app.route('/logout.html')
# def logout():
#     return render_template('logout.html')

@app.route('/PredictionMicrobe', methods=["GET","POST"])
def upload():
    
    if request.method=="POST":
        file=request.files['image']
        basepath=os.path.dirname(__file__) #getting the current path i.e where app.py is present
        #print("current path",basepath)
        filepath=os.path.join(basepath,'uploads',file.filename) #from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        #print("upload folder is",filepath)
        file.save(filepath)
        img = tf.keras.utils.load_img(filepath,target_size=(256,256)) # Reading image
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0) # expanding Dimensions
        print(model.predict(x))
        pred = np.argmax(model.predict(x)) # Predicting the higher probablity index
        op = ['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 'Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria', 'Yeast'] # Creating list
        #text="CNN Classified Microbe is : " +str(op[pred[0]])

        op[pred]
        result = op[pred]
        #result=str(op[ pred[0].tolist().op(1)])

        with open(filepath, 'rb') as uploadedfile:
             img_base64 = base64.b64encode(uploadedfile.read()).decode()

        return render_template('Prediction.html',prediction=result,image=img_base64)

    #return render_template("text")

        
""" Running our application """
if __name__ == '__main__':
   app.run(debug=True)
