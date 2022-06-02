from crypt import methods
from datetime import datetime

import logging
import os
import json
import random
import urllib.request

import tensorflow as tf
import numpy as np
from keras.models import load_model

from flask import Flask, request

app = Flask(__name__)

# saving links for affiliation & article
link_affiliation = 'https://insul-in-default-rtdb.firebaseio.com/affiliation_product.json'
link_article = 'https://insul-in-default-rtdb.firebaseio.com/article.json'

@app.route('/', methods=['GET', 'POST'])
def homepage():

    # if request POST
    if request.method == 'POST':
    
        # parse variables & inputing it to an object called userData
        userData ={
            "error": bool(0),
            "message": "success", 
            }

        # recieve the data from app
        age = request.value.get('age')
        gender = int(request.value.get('gender') == bool(1))
        polyuria = int(request.value.get('polyuria') == bool(1))
        polydipsia = int(request.value.get('polydipsia') == bool(1))
        weight_loss = int(request.value.get('weight_loss') == bool(1))
        weakness = int(request.value.get('weakness') == bool(1))
        polyphagia = int(request.value.get('polyphagia') == bool(1))
        genital_thrush = int(request.value.get('genital_thrush') == bool(1))
        visual_blurring = int(request.value.get('visual_blurring') == bool(1))
        itching = int(request.value.get('itching') == bool(1))
        irritability = int(request.value.get('irritability') == bool(1))
        delayed_healing = int(request.value.get('delayed_healing') == bool(1))
        partial_paresis = int(request.value.get('partial_paresis') == bool(1))
        muscle_stiffness = int(request.value.get('muscle_stiffness') == bool(1))
        alopecia = int(request.value.get('alopecia') == bool(1))
        obesity = int(request.value.get('obesity') == bool(1))

        # dummy data 
        # age = 40
        # gender = int(bool(0) == bool(1))
        # polyuria = int(bool(0) == bool(1))
        # polydipsia = int(bool(0) == bool(1))
        # weight_loss = int(bool(0) == bool(1))
        # weakness = int(bool(0) == bool(1))
        # polyphagia = int(bool(0) == bool(1))
        # genital_thrush = int(bool(0) == bool(1))
        # visual_blurring = int(bool(0) == bool(1))
        # itching = int(bool(0) == bool(1))
        # irritability = int(bool(0) == bool(1))
        # delayed_healing = int(bool(0) == bool(1))
        # partial_paresis = int(bool(0) == bool(1))
        # muscle_stiffness = int(bool(0) == bool(1))
        # alopecia = int(bool(1) == bool(1))
        # obesity = int(bool(1) == bool(1))

        # making an array
        arr_pred = [age, gender, polyuria, polydipsia, weight_loss, weakness, polyphagia, genital_thrush, visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity]

        # load the model
        Data = np.array(arr_pred)
        Data = Data.reshape(1, -1)

        model = tf.keras.models.load_model("mymodel.h5")
        model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        # make prediction
        pred = model.predict(Data)      
        for value in pred :
            if value > 0.5:
                value = 1
                userData["result_diagnose"] = bool(value)
            else:
                value = 0
                userData["result_diagnose"] = bool(value)

        json_user = json.dumps(userData)
        json_loadUser= json.loads(json_user)

        if json_loadUser["result_diagnose"] == False:

            # reading data from url
            with urllib.request.urlopen(link_article) as url:
                article = json.loads(url.read().decode())
                i = list(range(3))
                x = {}
                x["article"] = random.sample(article, len(i))
                
                # merge
                merged_result = { **json_loadUser, **x}
                result= json.dumps(merged_result)
                final_result= json.loads(result)
                return final_result

        elif json_loadUser["result_diagnose"] == True:

            # reading data from url
            with urllib.request.urlopen(link_affiliation) as url:
                affiliation = json.loads(url.read().decode())
                i = list(range(3))
                x = {}
                x["affiliation_product"] = random.sample(affiliation, len(i))
                
                # merge
                merged_result = { **json_loadUser, **x}
                result= json.dumps(merged_result)
                final_result= json.loads(result)
                return final_result

    # if not POST
    elif request.method == 'GET':
        # parse variables & inputing it to an object called userData
        userData ={
            "result_diagnose": bool(0),
            "error": bool(1),
            "message": "method error"
            }
        return userData

# If server error
@app.errorhandler(500)
def server_error(e):
    logging.exception("An error occurred during a request.")
    return (
        """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(
            e
        ),
        500,
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
