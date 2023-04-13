# -*- coding: utf-8 -*-
''' This is the file you have to modify for the tournament. Your default AI player must be called by this module, in the
myPlayer class.

Right now, this class contains the copy of the randomPlayer. But you have to change this!
'''

import time
import Goban 
from random import choice
from playerInterface import *
from tensorflow.keras.models import load_model
from math import inf
import numpy as np

class myPlayer(PlayerInterface):
    ''' Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and 
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self.model = load_model('model.keras')

    def getPlayerName(self):
        return "Random Player"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS" 
        move = AmiAlphaBeta(self._board, 2, self._mycolor, self.model) 
        self._board.push(move)

        # New here: allows to consider internal representations of moves
        print("I am playing ", self._board.move_to_str(move))
        print("My current board :")
        self._board.prettyPrint()
        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(move) 

    def playOpponentMove(self, move):
        print("Opponent played ", move) # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move)) 

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
#################### Tree-search algorithm #########################

def MaxValue(board, a, b, limit, color, model) : # Évaluation Ami
    if limit == 0 or board.is_game_over():
        return model.predict(np.array([translate_board(board.get_board())]),verbose = 0)[0][color] #return the answer 
        board.push(o)
        a = max(a,MinValue(board, a, b, limit-1, color, model))
        board.pop()
        if a >= b : return b
    return a

def MinValue(board, a, b, limit, color, model) : # Évaluation Ennemi
    if limit == 0 or board.is_game_over():
        return model.predict(np.array([translate_board(board.get_board())]),verbose = 0)[0][color]
    for n in board.legal_moves() :
        board.push(n)
        b = min(b,MaxValue(board, a, b, limit-1, color, model))
        board.pop()
        if a >= b : return a
    return b

def AmiAlphaBeta(board, limit, color, model) :
    a = -100000
    b = 100000
    best = -inf
    legalMoves = board.legal_moves() 
    bestMove = legalMoves[0]
    for m in legalMoves :
        board.push(m)
        a = max(best,MinValue(board, a, b, limit-1, color, model))
        board.pop()
        if a > best :
            bestMove = m
            best = a
    return bestMove


####################################################################
   

def translate_board(board):
    new_board = np.zeros((9,9,2))
    for i in range(9):
        for j in range(9):
            if board[i*9+j] != 0:
                new_board[8-i][j][board[i*9+j]-1] = 1
    return new_board






def maximum(model, board, color) :
    legalMoves = board.legal_moves() 
    bestMove = legalMoves[0]
    print(model.predict(np.array([translate_board(board.get_board())]),verbose = 0)[0][color],model.predict(np.array([translate_board(board.get_board())]),verbose = 0)[0][(1+color)%2])
    board.push(bestMove)
    maxi = model.predict(np.array([translate_board(board.get_board())]),verbose = 0)[0][color]
    board.pop()
    for move in legalMoves:
        board.push(move)
        score = model.predict(np.array([translate_board(board.get_board())]),verbose = 0)[0][color]
        board.pop()
        if (score >= maxi):
            maxi = score
            bestMove = move
    return bestMove
