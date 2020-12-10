# Server Process for Alpha Zero.

# Handles HTTP requests for:
#  - GET /weights Latest model weights
#  - POST /train New training data

# Communicates with training process (train.py) via files:
#  - latest_model.h5 : Latest model weights
#  - training_data.log : Training data sent by clients

import flask

app = flask.Flask(__name__)

@app.route("/hi")
def say_hi():
    return "Hello!"

app.run(port=5000)