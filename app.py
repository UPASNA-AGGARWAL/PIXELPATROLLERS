import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pickle
from PIL import Image
import io
import easyocr
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import cv2
import os

app = Flask(__name__)
model_cb = pickle.load(open('decision_tree_model.pkl','rb'))

with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

vectorizer = pickle.load(open("tfidfvectorizer.pkl", "rb"))

reader = easyocr.Reader(['en','hi'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cb')
def cb_home():
    return render_template('cb_index.html', prediction = None, image_prediction = None)



@app.route('/cb/predict', methods=['GET','POST'])
def predict():
    prediction = None
    if (request.method == 'POST'):
        user_input = request.form['text']
        transformed_input = vectorizer.transform([user_input])
        prediction = model_cb.predict(transformed_input)[0]

    return render_template('cb_index.html', prediction = prediction, image_prediction = None)

@app.route('/cb/predict_text_api', methods=['POST'])
def predict_api():
    data = request.get_json()  # Get JSON data from request
    user_input = data.get('text', '')  # Extract 'text' field
    
    if user_input:
        transformed_input = vectorizer.transform([user_input])  # Use transform, NOT fit_transform
        prediction = model_cb.predict(transformed_input)[0]
    else:
        prediction = "No input provided"

    output = "No Input Provided"
    if (prediction == 1):
        output = "1"
    else:
        output = "0"
    return jsonify(output)
    
@app.route('/cb/predict_image', methods = ['GET','POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))  
        image = np.array(image)
        
        extracted_text = reader.readtext(image, detail=0)
        extracted_text = ' '.join(extracted_text)
        response = requests.post('http://127.0.0.1:5000/cb/predict_text_api', json={'text': extracted_text})
        return render_template('cb_index.html', image_prediction = response.json(), prediction = None)  # Pass text to template

image_dimensions = {'height':256, 'width':256, 'channels':3}

class Classifier:
    def __init__():
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['accuracy'])

    def init_model(self):
        x = Input(shape = (image_dimensions['height'],
                           image_dimensions['width'],
                           image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)
    
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 
    return img

model_df = Meso4()
model_df.load('model/Meso4_DF.h5')

@app.route('/df')
def dfhome():
    return render_template('df_index.html')

@app.route('/df/predict_df',methods=['GET','POST'])
def dfpredict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)

    image = preprocess_image(file_path)
    prediction = model_df.predict(image)[0][0]  # Adjust based on your model's output

    rounded_pred = round(prediction)  # Convert probability to binary classification
    if rounded_pred == 1:
        result = "Real"
    else:
        result = "Deepfake"

    return jsonify({'filename': file.filename, 'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)