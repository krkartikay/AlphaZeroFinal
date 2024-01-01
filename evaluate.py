# Neural network evaluation process

# Gets latest neural net from server and evaluates it by playing against
# an opponent that always plays a random legal move. Reports win/loss rate
# to a file 'evaluate.tsv'

import game
import model
import config
import selfplay

import time
import random
import requests
import tempfile


def load_model_weights():
    data = requests.get(config.server_address + "/weights")
    tfile, tname = tempfile.mkstemp(".h5")
    open(tname, "wb").write(data.content)
    net.load(tname)

def evaluate_net(net: model.Model, eval_player = 'nnet', exclude_illegal=False):
    players = ["random", eval_player]
    random.shuffle(players)
    g = game.GameState()
    i = 0
    t1 = time.time()
    while not g.terminated():
        actions = g.legal_actions()
        s = sum(actions)
        turn = i % 2
        if config.eval_verbose:
            print('player: ', players[turn])
        if players[turn] == "nnet":
            probs = net.predict(g)[0][0]
            if exclude_illegal:
                probs = [probs[i] * actions[i] for i in range(len(actions))]
                s = sum(probs)
                probs = [p/s for p in probs]
        elif players[turn] == 'mcts':
            m = selfplay.MCTS(net)
            probs = m.get_probs(g)
        else:
            probs = [x/s for x in actions]
        action = random.choices(list(range(len(actions))), probs)[0]
        i += 1
        if actions[action] == False:
            t2 = time.time()
            return i, "illegal", (t2-t1)/i
        g = g.next_state(action)
        if config.eval_verbose:
            print(g.board)
    winner = g.winner()
    t2 = time.time()
    if winner == 1:
        return i, players[0], (t2-t1)/i
    elif winner == -1:
        return i, players[1], (t2-t1)/i
    else:
        return i, "draw", (t2-t1)/i

def evaluate_model(model: model.Model, verbose=False):
    d = {}
    eval_player = config.eval_player
    all_moves = []
    for i in range(config.num_evaluate):
        results = evaluate_net(model, eval_player, exclude_illegal=False)
        num_moves, winner, time_taken = results
        all_moves.append(num_moves)
        d[winner] = d.get(winner, 0) + 1
        d['moves'] = d.get('moves', 0) + num_moves
        if verbose:
            print(f"{results}\t{d}")
    win     = d.get(eval_player, 0)
    draw    = d.get('draw', 0)
    loss    = d.get('random', 0)
    illegal = d.get('illegal', 0)
    moves = d.get('moves', 0) / config.num_evaluate
    return (win, draw, loss, illegal, moves, time_taken, all_moves)

if __name__ == "__main__":
    try:
        log = open('evaluate.tsv', 'r')
        log.close()
    except FileNotFoundError:
        with open("evaluate.tsv", "w") as log:
            log.write("win\tdraw\tloss\tillegal\tmoves\ttime_taken_ms\n")

    net = model.Model()
    load_model_weights()

    win, draw, loss, illegal, moves, time_taken, all_moves = evaluate_model(net, verbose=True)

    with open("evaluate.tsv", "a") as log:
        log.write(f"{win}\t{draw}\t{loss}\t{illegal}\t{moves}\t{time_taken*1000:0.4f}\n")
    # print(f"Win: {win:3}\tDraw: {draw:3}\tLose: {loss:3}\tIllegal: {illegal:4}\t\t" +
    #       f"Time taken: {(t2-t1)*1000/config.num_evaluate:0.1f} ms per game")
