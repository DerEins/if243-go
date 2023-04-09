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


def generate_training_data(player1, player2):
    game = {
        "depth": int(random.choice(range(1, 50))),
        "list_of_moves": [],
        "black_stones": [],
        "white_stones": [],
        "rollouts": 0,
        "black_wins": 0,
        "black_points": 0,
        "white_wins": 0,
        "white_points": 0,
    }
    training_board = Goban.Board()

    players = []
    player1.newGame(Goban.Board._BLACK)
    players.append(player1)

    player2.newGame(Goban.Board._WHITE)
    players.append(player2)

    training_nextplayer = 0
    training_nextplayercolor = Goban.Board._BLACK

    def run_game(b, nextplayer, nextplayercolor, training_board=False, training_move=''):
        # legal moves are given as internal (flat) coordinates, not A1, A2, ...
        legals = b.legal_moves()
        # I have to use this wrapper if I want to print them
        otherplayer = (nextplayer + 1) % 2
        othercolor = Goban.Board.flip(nextplayercolor)

        # The move must be given by "A1", ... "J8" string coordinates (not as an internal move)
        assert (not training_board or training_move != 0)
        if len(training_move) != 0:
            move = players[nextplayer].getPlayerMove(my_move=training_move)
        else:
            move = players[nextplayer].getPlayerMove(alea=True)
        assert Goban.Board.name_to_flat(
            move) in legals, f'Illegal moves from {nextplayer}'

        b.push(Goban.Board.name_to_flat(move))
        if training_board:
            game["list_of_moves"].append(move)
            if (nextplayercolor == Goban.Board._BLACK):
                game["black_stones"].append(move)
            else:
                game["white_stones"].append(move)
        players[otherplayer].playOpponentMove(move)
        return otherplayer, othercolor

    nb_moves = game["depth"]
    while not training_board.is_game_over() and nb_moves != 0:
        training_nextplayer, training_nextplayercolor = run_game(
            training_board, training_nextplayer, training_nextplayercolor, True)
        nb_moves -= 1

    for i in range(100):
        b = Goban.Board()
        players[0].newGame(Goban.Board._BLACK)
        players[1].newGame(Goban.Board._WHITE)
        b_nextplayer = training_nextplayer
        b_nextplayercolor = training_nextplayercolor
        for move in game["list_of_moves"]:
            run_game(b, b_nextplayer, b_nextplayercolor, False, move)

        while not b.is_game_over():
            b_nextplayer, b_nextplayercolor = run_game(
                b, b_nextplayer, b_nextplayercolor, training_board=False)
        if b.result() == "1-0":
            game["white_wins"] += 1
        elif b.result() == "0-1":
            game["black_wins"] += 1
        game["black_points"] += b.compute_score()[0]
        game["white_points"] += b.compute_score()[1]
        game["rollouts"] += 1
    return game


if __name__ == "__main__":
    len_data = 100
    name = "gnugo10-VS-gnugo10"
    print(f"Generating {name} data... \t0/{len_data}", end="\r")
    for i in range(len_data):
        data = generate_training_data(
            gnugoPlayer.myPlayer(0), gnugoPlayer.myPlayer(0))
        print(f"Generating {name} data... \t{i+1}/{len_data}", end="\r")
        export_training_data(name, data)
    print(f"Full training data {name} available in {path}/{name}.json")
