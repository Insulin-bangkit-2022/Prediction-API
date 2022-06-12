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

    # parse variables & inputing it to an object called userData
    userData ={
        "error": bool(0),
        "message": "success", 
        }

    # making an array
    arr_pred = np.array([request.args.get('age'), request.args.get('gender'), request.args.get('polyuria'), request.args.get('polydipsia'), request.args.get('weightLoss'), request.args.get('weakness'), request.args.get('polyphagia'), request.args.get('genital_thrus'), request.args.get('visual_blurring'), request.args.get('itching'), request.args.get('irritability'), request.args.get('delayed_healing'), request.args.get('partial_paresis'), request.args.get('muscle_stiffness'), request.args.get('alopecia'), request.args.get('obesity')])

    arr_pred2 = arr_pred.astype(np.int32)

    # load the model
    Data = np.array(arr_pred2)
    Data = Data.reshape(1, -1)

    model = tf.keras.models.load_model("new_model.h5")
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
