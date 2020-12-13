# Configuration

# Defines various parameters for the training and self play processes.

server_host = "127.0.0.1"
server_port = 5000

server_address = f"http://{server_host}:{server_port}"

learning_rate = 0.1

num_actions = 9
num_evaluate = 100
num_simulate = 5

pb_c_init = 1.25
pb_c_base = 20000