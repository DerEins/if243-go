# -*- coding: utf-8 -*-
''' This is the file you have to modify for the tournament. Your default AI player must be called by this module, in the
myPlayer class.

Right now, this class contains the copy of the randomPlayer. But you have to change this!
'''

import time
import hashlib
import pickle
import Goban
from random import choice
from playerInterface import *
from tensorflow.keras.models import load_model
from copy import deepcopy
import numpy as np


class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and 
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self._model = load_model('model.keras')
        self._heuristic_cache = {}

    def getPlayerName(self):
        return "Random Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        move = self._getbestmove(2, float("-inf"), float("inf"), True)[1]
        self._board.push(move)

        # New here: allows to consider internal representations of moves
        print("I am playing ", self._board.move_to_str(move))
        print("My current board :")
        self._board.prettyPrint()
        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(move)

    def playOpponentMove(self, move):
        print("Opponent played ", move)  # New here
        #  the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
    ################ Tree-search algorithm #########################

    def _getbestmove(self, depth, alpha, beta, maximizing_player):
        max_score = float('-inf')
        best_move = None
        for move in self._board.legal_moves():
            self._board.push(move)
            score, _ = self._alphabeta(
                self._board, depth - 1, alpha, beta, False)
            self._board.pop()
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return max_score, best_move

    def _alphabeta(self, board, depth, alpha, beta, maximizing_player):
        key = myPlayer.hash_board(board)
        if key in self._heuristic_cache:
            return self._heuristic_cache[key], None

        if depth == 0 or board.is_game_over():
            # Calcul de l'heuristique et stockage dans le cache
            h = self._model.predict(np.array([myPlayer.translate_board(board.get_board())]), verbose=0)[
                0][self._mycolor]
            self._heuristic_cache[key] = h
            return h, None

        if maximizing_player:
            max_score = float('-inf')
            best_move = None
            for move in board.legal_moves():
                board.push(move)
                score, _ = self._alphabeta(
                    board, depth - 1, alpha, beta, False)
                board.pop()
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            self._heuristic_cache[key] = alpha
            return max_score, best_move

        else:
            min_score = float('inf')
            best_move = None
            for move in board.legal_moves():
                board.push(move)
                score, _ = self._alphabeta(
                    board, depth - 1, alpha, beta, True)
                board.pop()
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break
            self._heuristic_cache[key] = beta
            return min_score, best_move


####################################################################


    @staticmethod
    def hash_board(board):
        board_serialized = pickle.dumps(board)
        hash_object = hashlib.sha256(board_serialized)
        return hash_object.hexdigest()

    @staticmethod
    def translate_board(board):
        new_board = np.zeros((9, 9, 2))
        for i in range(9):
            for j in range(9):
                if board[i*9+j] != 0:
                    new_board[8-i][j][board[i*9+j]-1] = 1
        return new_board
