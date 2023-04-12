import json
import gnugoPlayer
import Goban
from copy import deepcopy
from numpy import random

path = "./IA_Nolan/go-package/training_data"


def export_training_data(name, data):
    with open(f'{path}/{name}.json', 'ab+') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(json.dumps([data]).encode())
        else:
            f.seek(-1, 2)
            f.truncate()
            f.write(" ,\n".encode())
            f.write(json.dumps(data).encode())
            f.write(']'.encode())


def import_training_data(name):
    with open(f'{path}/{name}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def run_round(b, players, nextplayer, nextplayercolor, training_move=''):
    # legal moves are given as internal (flat) coordinates, not A1, A2, ...
    legals = b.legal_moves()
    # I have to use this wrapper if I want to print them
    otherplayer = (nextplayer + 1) % 2
    othercolor = Goban.Board.flip(nextplayercolor)

    # The move must be given by "A1", ... "J8" string coordinates (not as an internal move)
    if len(training_move) != 0:
        move = players[nextplayer].getPlayerMove(my_move=training_move)
    else:
        move = players[nextplayer].getPlayerMove(alea=True)
    b.push(Goban.Board.name_to_flat(move))

    players[otherplayer].playOpponentMove(move)
    return otherplayer, othercolor


def initiate_training_data():
    game = {
        "depth": int(random.choice(range(1, 10))),
        "list_of_moves": [],
        "black_stones": [],
        "white_stones": [],
        "rollouts": 0,
        "black_wins": 0,
        "black_points": 0,
        "white_wins": 0,
        "white_points": 0,
    }
    board = Goban.Board()

    players = []
    player1 = gnugoPlayer.myPlayer(0)
    player1.newGame(Goban.Board._BLACK)
    players.append(player1)

    player2 = gnugoPlayer.myPlayer(0)
    player2.newGame(Goban.Board._WHITE)
    players.append(player2)

    nextplayer = 0
    nextplayercolor = Goban.Board._BLACK

    nb_moves = game["depth"]
    while not board.is_game_over() and nb_moves != 0:
        nextplayer, nextplayercolor = run_round(
            board, players, nextplayer, nextplayercolor)
        nb_moves -= 1
    game["black_stones"] = [(board.move_to_str(i)) for i in range(len(board.get_board()))
                            if (board.get_board()[i] == Goban.Board._BLACK)]
    game["white_stones"] = [(board.move_to_str(i)) for i in range(len(board.get_board()))
                            if (board.get_board()[i] == Goban.Board._WHITE)]
    game["list_of_moves"] = game["black_stones"] + game["white_stones"]
    game["depth"] -= nb_moves
    return game


def rollout_game(game):
    board = Goban.Board()

    players = []
    player1 = gnugoPlayer.myPlayer(0)
    player1.newGame(Goban.Board._BLACK)
    players.append(player1)

    player2 = gnugoPlayer.myPlayer(0)
    player2.newGame(Goban.Board._WHITE)
    players.append(player2)

    nextplayer = 0
    nextplayercolor = Goban.Board._BLACK

    for move in game["list_of_moves"]:
        nextplayer, nextplayercolor = run_round(
            board, players, nextplayer, nextplayercolor, training_move=move)
    depth = 0
    while not board.is_game_over():
        depth += 1
        nextplayer, nextplayercolor = run_round(
            board, players, nextplayer, nextplayercolor)
    if board.result() == "1-0":
        game["white_wins"] += 1
    elif board.result() == "0-1":
        game["black_wins"] += 1
    elif board.result() == "1/2-1/2":
        game["black_wins"] += 0.5
        game["white_wins"] += 0.5
    game["black_points"] += board.compute_score()[0]
    game["white_points"] += board.compute_score()[1]
    game["rollouts"] += 1
    return game


def generate_training_data(game, rollout=100):
    for i in range(rollout):
        game = rollout_game(game)
    return game


if __name__ == "__main__":
    len_data = 100
    games = [initiate_training_data() for i in range(100)]
    name = "gnugo10-VS-gnugo10"
    print(f"Generating {name} data... \t0/{len_data}", end="\r")
    for i in range(len(games)):
        games[i] = generate_training_data(games[i], 100)
        print(f"Generating {name} data... \t{i+1}/{len_data}", end="\r")
        export_training_data(name, games[i])
    print(f"Full training data {name} available in {path}/{name}.json")
