import json
from copy import copy
import numpy as np
path = "./data"


def export_training_data(name, data):
    with open(f'{path}/{name}.json', 'w') as f:
        f.write(
            '[' +
            ',\n'.join(json.dumps(i) for i in data) +
            ']\n')


def import_training_data(name):
    with open(f'{path}/{name}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def rot90(move):
    if move == "PASS":
        return move
    if (move[0] == "J"):
        move = f"I{move[1]}"
    new_move = f"{chr(64 + int(move[1]))}{9 - (ord(move[0]) - 65)}"
    if new_move[0] == "I":
        new_move = f"J{new_move[1]}"
    return new_move


def flipud(move):
    if move == "PASS":
        return move
    return f"{move[0]}{10 - int(move[1])}"


def duplicate_opening_data(old_db, new_db):
    old_data = import_training_data(old_db)
    new_data = []
    for d in old_data:
        new_d = copy(d)

        for i in range(4):
            new_d = copy(new_d)
            new_d["moves"] = [rot90(m) for m in new_d["moves"]]
            new_data.append(new_d)
        for i in range(4):
            new_d = copy(new_data[-4])
            new_d["moves"] = [flipud(m) for m in new_d["moves"]]
            new_data.append(new_d)
            assert (new_data[-4] != new_data[-1])
    export_training_data(new_db, new_data)


if __name__ == "__main__":
    duplicate_opening_data("old_openings", "openings")
