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
from threading import Thread, Lock, Event
from filelock import FileLock
from datetime import datetime, timedelta
from queue import Queue
import time
import uuid
 
hits = []
hits_lock = Lock()

m = model.Model('cuda')
m.load()
app = flask.Flask(__name__)

logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Constants
MAX_WAIT_MS = 1  # max wait time in milliseconds
MAX_BATCH_SIZE = 1024

# Global Variables
input_queue = Queue()
output_dict = {}
event_dict = {}
lock = Lock()

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

    # This is the input game position
    position = flask.request.json

    # Generate a unique ID using the uuid module
    unique_id = str(uuid.uuid4())
    input_queue.put({"id": unique_id, "data": position})

    # Create a new event for this request and store it in event_dict
    with lock:
        event_dict[unique_id] = Event()

    # Wait for the specific event or the maximum waiting time
    event_dict[unique_id].wait(1)  # 1 second

    with lock:
        # Once the prediction is done or max time is reached, try to get the prediction
        prediction = output_dict.pop(unique_id, None)
        # Clean up the event associated with this request
        event = event_dict.pop(unique_id, None)
        if event:
            event.clear()
    
    if prediction is None:
        return flask.jsonify({"error": "Prediction timed out."}), 500
    else:
        return flask.jsonify({"prediction": prediction})

# =============================================================================
# Prediction thread

def batch_predict():
    while True:
        start_time = time.time()
        inputs = []
        ids = []

        # Collect requests
        while time.time() - start_time < MAX_WAIT_MS / 1000 and len(inputs) < MAX_BATCH_SIZE:
            if not input_queue.empty():
                item = input_queue.get()
                ids.append(item["id"])
                inputs.append(item["data"])

        if inputs:
            print(len(inputs))
            policy, value = m.predict_image(inputs)
            with lock:
                # Assign predictions to outputs
                for id_, policy, value in zip(ids, policy, value):
                    output_dict[id_] = [policy.tolist(), value.tolist()]
                    # Notify the specific request that its prediction is done
                    if id_ in event_dict:
                        event_dict[id_].set()

# Start the background predict thread
thread = Thread(target=batch_predict)
thread.start()

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