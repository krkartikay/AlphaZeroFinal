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

"0. Resetting everything!"

with open("training_data.log",'w') as f:
    f.write('')

system('rm latest_weights.h5')

"1. Starting server... "

subprocess.Popen(["python3", "server.py"])

f"2. Starting {config.client_processes_num} client processes..."

for i in range(config.client_processes_num):
    subprocess.Popen(["python3", "client.py"])

"3. Starting Evaluator..."

subprocess.Popen(["python3", "evaluate.py"])

"4. Training! Sit back and relax..... "

num = st.empty()
training = st.empty()
loss_graph = st.empty()
eval_graph = st.empty()
training_games_graph = st.empty()

loss = []
last_len = 0

while True:
    lines = open("training_data.log").readlines()
    num.write(f'Length of training data file: `{len(lines)}`')
    time.sleep(1)
    training.write('')
    if len(lines) > config.train_after_games + last_len:
        training.write(f'Training now ...')
        losses = train.train()
        loss += [x[0] for x in losses]
        loss_graph.line_chart(pd.DataFrame(loss, columns=["loss"]))
        last_len = len(lines)
    eval_data = pd.read_csv(open("evaluate.tsv"),sep="\t")
    eval_graph.line_chart(eval_data)
    game_results = [line.split('\t')[1] for line in lines if line.split('\t')[0]]
    game_results_condensed = [Counter(game_results[i:i+config.train_after_games])
                                for i in range(0, len(game_results), config.train_after_games)]
    game_results_graph = {'First wins': [x['1'] for x in game_results_condensed],
                            'Second wins': [x['-1'] for x in game_results_condensed],
                            'Draw': [x['0'] for x in game_results_condensed]}
    training_games_graph.line_chart(game_results_graph)