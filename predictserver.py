# Prediction Server Process for Alpha Zero.

# Handles HTTP requests for:
#  - GET /weights Latest model weights
#  - POST /train New training data
#  - GET /prediction Position evaluateion (policy and value)

# Communicates with training process (train.py) via files:
#  - latest_weights.pth : Latest model weights
#  - training_data.log : Training data sent by clients

import model
import config

import flask
import logging
import threading
from filelock import FileLock
from datetime import datetime, timedelta
 
hits = []
hits_lock = threading.Lock()

m = model.Model('cuda')
m.load()
app = flask.Flask(__name__)

logging.basicConfig(filename='app.log', level=logging.DEBUG)

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

@app.route("/predict", methods=("GET",))
def predict():
    with hits_lock:
        hits.append(datetime.utcnow())
    position = flask.request.json
    policy, value = m.predict_image(position)
    return flask.jsonify([policy.tolist(), value.tolist()])

# =============================================================================
# Througput computation background task
import threading

def compute_throughput():
    global hits
    last_computation_time = datetime.utcnow()
    while True:
        # Sleep for 1 second
        threading.Event().wait(1)

        current_time = datetime.utcnow()
        with hits_lock:
            # Count hits since the last computation
            recent_hits = [hit for hit in hits if last_computation_time <= hit < current_time]
            
            # Update hits list and last computation time
            hits[:] = [hit for hit in hits if hit >= current_time]
            last_computation_time = current_time

        # Compute and print throughput
        throughput = len(recent_hits)
        print(f"Throughput: {throughput} req/s")

# Start the throughput computation thread
thread = threading.Thread(target=compute_throughput)
thread.daemon = True  # Daemonize the thread to ensure it exits when main program does
thread.start()
# =============================================================================

app.run(host=config.server_host, port=config.server_port)