# Configuration

# Defines various parameters for the training and self play processes.

server_host = "127.0.0.1"
server_port = 5000

server_address = f"http://{server_host}:{server_port}"

learning_rate = 3e-5
decay = 1e-4
train_epochs = 100
batch_size = 5000

num_actions = 64*64
num_evaluate = 10
num_simulate = 11

eval_verbose = True
eval_player = 'mcts'

pb_c_init = 1.25
pb_c_base = 20000

train_after_games = 1
last_N_games = 50000
client_processes_num = 8
client_play_games_num = 10

temperature = 1.0