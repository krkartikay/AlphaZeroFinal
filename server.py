# Server Process for Alpha Zero.

# Handles HTTP requests for:
#  - GET Latest model weights
#  - POST New training data

# Communicates with training process (train.py) via files:
#  - latest_model.h5 : Latest model weights
#  - training_data.log : Training data sent by clients

