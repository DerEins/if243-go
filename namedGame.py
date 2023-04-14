''' Sorry no comments :).
'''
import Goban
import importlib
import time
from io import StringIO
import sys
import json

path = "."


def fileorpackage(name):
    if name.endswith(".py"):
        return name[:-3]
    return name


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


def run_game(playerPackage1="myPlayer", playerPackage2="myPlayer"):
    classNames = [fileorpackage(playerPackage1), fileorpackage(playerPackage2)]

    b = Goban.Board()

    players = []
    player1class = importlib.import_module(classNames[0])
    player1 = player1class.myPlayer()
    player1.newGame(Goban.Board._BLACK)
    players.append(player1)

    player2class = importlib.import_module(classNames[1])
    player2 = player2class.myPlayer()
    player2.newGame(Goban.Board._WHITE)
    players.append(player2)

    totalTime = [0, 0]  # total real time for each player
    nextplayer = 0
    nextplayercolor = Goban.Board._BLACK
    nbmoves = 1

    outputs = ["", ""]
    sysstdout = sys.stdout
    stringio = StringIO()
    wrongmovefrom = 0

    while not b.is_game_over():
        print("Referee Board:")
        b.prettyPrint()
        print("Before move", nbmoves)
        # legal moves are given as internal (flat) coordinates, not A1, A2, ...
        legals = b.legal_moves()
        # I have to use this wrapper if I want to print them
        print("Legal Moves: ", [b.move_to_str(m) for m in legals])
        nbmoves += 1
        otherplayer = (nextplayer + 1) % 2
        othercolor = Goban.Board.flip(nextplayercolor)

        currentTime = time.time()
        sys.stdout = stringio
        # The move must be given by "A1", ... "J8" string coordinates (not as an internal move)
        move = players[nextplayer].getPlayerMove()
        sys.stdout = sysstdout
        playeroutput = stringio.getvalue()
        stringio.truncate(0)
        stringio.seek(0)
        print(("[Player "+str(nextplayer) + "] ").join(playeroutput.splitlines(True)))
        outputs[nextplayer] += playeroutput
        totalTime[nextplayer] += time.time() - currentTime
        print("Player ", nextplayercolor,
              players[nextplayer].getPlayerName(), "plays: " + move)  # changed

        if not Goban.Board.name_to_flat(move) in legals:
            print(otherplayer, nextplayer, nextplayercolor)
            print("Problem: illegal move")
            wrongmovefrom = nextplayercolor
            break
        # Here I have to internally flatten the move to be able to check it.
        b.push(Goban.Board.name_to_flat(move))
        players[otherplayer].playOpponentMove(move)

        nextplayer = otherplayer
        nextplayercolor = othercolor

    print("The game is over")
    b.prettyPrint()
    result = b.result()
    print("Time:", totalTime)
    print("GO Score:", b.final_go_score())
    print("Winner: ", end="")
    winner = None
    if wrongmovefrom > 0:
        if wrongmovefrom == b._WHITE:
            print("BLACK")
            winner = Goban.Board._BLACK
        elif wrongmovefrom == b._BLACK:
            print("WHITE")
            winner = Goban.Board._WHITE
        else:
            print("ERROR")
    elif result == "1-0":
        print("WHITE")
        winner = Goban.Board._WHITE
    elif result == "0-1":
        print("BLACK")
        winner = Goban.Board._BLACK
    else:
        print("DEUCE")
    return {
        "black_player": playerPackage1,
        "white_player": playerPackage2,
        "time": totalTime,
        "score": b.final_go_score(),
        "winner": winner}


def has_won(playerPackage, d):
    return (d["black_player"] == playerPackage and d["winner"] == Goban.Board._BLACK) or (d["white_player"] == playerPackage and d["winner"] == Goban.Board._WHITE)


def games_stats(name):
    data = import_training_data(name)
    nbMyPlayerWins = len([d for d in data if has_won("myPlayer", d)])
    print(f"MyPlayer a gagné {nbMyPlayerWins} parties sur {len(data)}")


if __name__ == "__main__":
    hoursToPlay = 8
    gameTime = 0.5
    players = ["randomPlayer", "myPlayer"]
    for i in range(int(hoursToPlay/gameTime)):
        result = run_game(players[1-i % 2], players[i % 2])
        export_training_data("scores", result)
    games_stats("scores")
