from flask import Flask, request, render_template, url_for, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 224, 224, 3)
    return image_arr

classes = [0 ,1]
model=load_model("covid19_kh_abdo.h5")

@app.route('/')
def index():

    return render_template('index.html', appName="Intel Image Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        if prediction==1:
            prediction2="NORMAL"
        else:
            prediction2="COVID"
        print(prediction)
        return jsonify({'prediction': prediction2})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(image_arr)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]
        if prediction==1:
            prediction1="NORMAL"
        else:
            prediction1="COVID"
        print(prediction1)

        return render_template('index.html', prediction=prediction1, image='static/IMG/', appName="Intel Image Classification")
    else:
        return render_template('index.html',appName="Intel Image Classification")


if __name__ == '__main__':
    app.run(debug=True)