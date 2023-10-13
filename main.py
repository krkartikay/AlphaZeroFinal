# Main process for Alpha Zero

# Runs the server in the background,
# then runs N client processes to keep generating data,
# and then runs the training process when enough data has been generated

from os import system
import pandas as pd
import streamlit as st
import subprocess
import time

from collections import Counter

import config
import train

"# Alpha Zero Dashboard"

# "0. Resetting everything!"

# with open("training_data.log",'w') as f:
#     f.write('')

# system('rm latest_weights.h5')

# "1. Starting server... "

# subprocess.Popen(["python3", "server.py"])
# time.sleep(0.1)

# f"2. Starting {config.client_processes_num} client processes..."

# for i in range(config.client_processes_num):
#     subprocess.Popen(["python3", "client.py"])
#     time.sleep(0.1)

# "3. Starting Evaluator..."

# subprocess.Popen(["python3", "evaluate.py"])
# time.sleep(0.1)

# "4. Training! Sit back and relax..... "

# subprocess.Popen(["python3", "train.py"])
# time.sleep(0.1)

num = st.empty()
training = st.empty()
loss_graph = st.empty()
eval_graph = st.empty()
# training_games_graph = st.empty()

while True:
    lines = open("training_data.log").readlines()
    num.write(f'Length of training data file: `{len(lines)}`')
    eval_data = pd.read_csv(open("evaluate.tsv"),sep="\t")
    eval_graph.line_chart(eval_data)
    loss_data = pd.read_csv(open("loss.tsv"),sep="\t")
    loss_graph.line_chart(loss_data)
    # game_results = [line.split('\t')[1] for line in lines if line.split('\t')[0]]
    # game_results_condensed = [Counter(game_results[i:i+config.train_after_games])
    #                             for i in range(0, len(game_results), config.train_after_games)]
    # game_results_graph = {'First wins': [x['1'] for x in game_results_condensed],
    #                         'Second wins': [x['-1'] for x in game_results_condensed],
    #                         'Draw': [x['0'] for x in game_results_condensed]}
    # training_games_graph.line_chart(game_results_graph)
    time.sleep(1)