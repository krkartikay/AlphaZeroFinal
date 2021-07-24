# Configuration

# Defines various parameters for the training and self play processes.

server_host = "127.0.0.1"
server_port = 5000

server_address = f"http://{server_host}:{server_port}"

learning_rate = 0.1
decay = 1e-4
train_epochs = 1000
batch_size = 5000

num_actions = 9
num_evaluate = 1000
num_simulate = 20

pb_c_init = 1.25
pb_c_base = 20000

train_after_games = 500
last_N_games = 50000
client_processes_num = 4
client_play_games_num = 10