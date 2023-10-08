# Server Process for Alpha Zero.

# Handles HTTP requests for:
#  - GET /weights Latest model weights
#  - POST /train New training data

# Communicates with training process (train.py) via files:
#  - latest_weights.pth : Latest model weights
#  - training_data.log : Training data sent by clients

import config

import flask
from filelock import FileLock

app = flask.Flask(__name__)

@app.route("/weights", methods=("GET",))
def get_weights():
    # send file 'latest_weights.pth'
    with FileLock("latest_weights.pth.lock"):
        # return flask.send_file(io.BytesIO(open("latest_weights.pth", "rb").read()),
        #                         attachment_filename="latest_weights.pth",
        #                         mimetype="application/octet-stream")
        return open("latest_weights.pth", "rb").read()

@app.route("/train", methods=("POST",))
def post_training_data():
    # get training data from client and append to file 'training_data.log' atomically
    # data = request.something ... see flask docs
    with FileLock("training_data.log.lock"):
        with open("training_data.log", "ab") as file:
            file.write(flask.request.data)
    return flask.jsonify("OK")

app.run(host=config.server_host, port=config.server_port)