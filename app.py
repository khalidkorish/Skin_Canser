from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
from io import BytesIO


app = Flask(__name__)

# Load the ResNet50 model
with open("resnet50.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights("resnet50.h5")

# Define the classes
classes = ['benign', 'malignant']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the image file from the request
    img_file = request.files['image']
    
    # Read the contents of the image file
    img_bytes = img_file.read()
    
    # Convert the bytes to an in-memory file-like object
    img_io = BytesIO(img_bytes)
    
    # Load the image from the in-memory file-like object
    img = image.load_img(img_io, target_size=(224, 224))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    img_array /= 255.

    # Make predictions
    prediction = loaded_model.predict(img_array)

    # Get the predicted class
    predicted_class = classes[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2110)
