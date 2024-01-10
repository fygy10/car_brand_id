from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import openai
import os
import cv2
import json
import numpy as np


#SET IMPORT INSTANCE, FLASK, AND OPENAI REQUIREMENTS
from car_id_cnn import CarIDModel
run_model = CarIDModel()

app = Flask(__name__)
#openai.api_key = 'OPENAI_API_KEY'

#INTERACTION WITH HTML
@app.route('/')
def index():
    return render_template('index.html')


#CHECK FOR FILE TYPES FOR VALID INPUTS (IMAGES)
extensions = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


#CONNECT TO OPENAI
#contains the prompt, and settings for response
def openai_connection(prompt):
    try:
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt= f'Tell me some information about this {prompt}',
            max_tokens=100,
            temperature=0.5,
            n=1,
            stop=None
        )

        # #check on response
        # print('OpenAI Response:', response)

        #indexes first choice
        if response.choices:
            return response.choices[0].text.strip()

    except Exception as e:
        print('An error occurred during the API request:', e)
        return None



#RUNS THE CNN INSTANCE ON THE IMAGE
def run_model_evaluate(img_data):
    classification_result = run_model.evaluate(img_data)

    return classification_result



#INTERACTION WITH FLASK FOR INPUT AND OUTPUT
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        img_data = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
        result = run_model_evaluate(img_data)

        openai_response = openai_connection(result)

        formatted_response = json.dumps(openai_response, indent=2)


        return render_template('index.html', result=formatted_response)

    else:
            return 'Invalid image file'




if __name__ == '__main__':
    app.run()