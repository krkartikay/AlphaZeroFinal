# Main process for Alpha Zero

# Runs the server in the background,
# then runs N client processes to keep generating data,
# and then runs the training process when enough data has been generated

from os import system
import pandas as pd
import streamlit as st
import subprocess
import time

import config
import train

"# Alpha Zero Dashboard"

"0. Resetting everything!"

with open("training_data.log",'w') as f:
    f.write('')

system('rm latest_weights.h5')

"1. Starting server... "

subprocess.Popen(["python", "server.py"])

f"2. Starting {config.client_processes_num} client processes..."

for i in range(config.client_processes_num):
    subprocess.Popen(["python", "client.py"])

"3. Starting Evaluator..."

subprocess.Popen(["python", "evaluate.py"])

"4. Training! Sit back and relax..... "

num = st.empty()
training = st.empty()
loss_graph = st.empty()
eval_graph = st.empty()

loss = []
last_len = 0

while True:
    lines = open("training_data.log").readlines()
    num.write(f'Length of training data file: `{len(lines)}`')
    time.sleep(1)
    if len(lines) > config.train_after_games + last_len:
        training.write(f'Training now ...')
        losses = train.train()
        training.empty()
        loss += losses
        loss_graph.line_chart(pd.DataFrame(loss, columns=["loss"]))
        last_len = len(lines)
        eval_data = pd.read_csv(open("evaluate.tsv"),sep="\t")
        eval_graph.line_chart(eval_data)