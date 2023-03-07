import Goban 
import myPlayer,IA,gnugoPlayer
from io import StringIO
import sys
import torch
import random
import numpy as np
from collections import deque,namedtuple
from model2 import Linear_QNet, QTrainer
from helper import plot
import time
import pygame
from math import inf
from enum import Enum

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


Point = namedtuple('Point', 'x, y')

# rgb colors
BLACK = (0,0,0)
WHITE = (255, 255, 255)
BACKGROUND= (200,200,200)
RED =(255,0,0)

BLOCK_SIZE = 100
SPEED = 0.00000001


class GobbanGameAI:

    def __init__(self, w=900, h=900):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Goban')
        self.clock = pygame.time.Clock()


MAX_MEMORY = 100000000
BATCH_SIZE = 100000000
LR = 0.01

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(81,256,256,100)
        #self.model.load_state_dict(torch.load("./model/model5.pth"))
        #self.model.eval()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(b):
        state = b.get_board()
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 0
        if random.randint(0, 200) < self.epsilon:
            final_move = random.randint(0, 100)      
               
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = move
       
        return final_move


def MaxValue(board, a, b, limit, color, agent) : # Évaluation Ami
    if limit == 0 or board.is_game_over():
        if color==1:
            bo2 = board.get_board()
            for i in bo2:
                if i == 1:
                    i=2
                elif i == 2:
                    i=1
            state = np.array(bo2, dtype=int)
        else:
            state = np.array(board.get_board(), dtype=int)
        return agent.get_action(state)
    for o in board.legal_moves() :
        board.push(o)
        a = max(a,MinValue(board, a, b, limit-1, color, agent))
        board.pop()
        if a >= b : return b
    return a

def MinValue(board, a, b, limit, color, agent) : # Évaluation Ennemi
    if limit == 0 or board.is_game_over():
        if color==1:
            bo2 = board.get_board()
            for i in bo2:
                if i == 1:
                    i=2
                elif i == 2:
                    i=1
            state = np.array(bo2, dtype=int)
        else:
            state = np.array(board.get_board(), dtype=int)
        return agent.get_action(state)
    for n in board.legal_moves() :
        board.push(n)
        b = min(b,MaxValue(board, a, b, limit-1, color, agent))
        board.pop()
        if a >= b : return a
    return b

def AmiAlphaBeta(board, limit, color, agent) :
    a = -inf
    b = +inf
    best = -inf
    legalMoves = board.legal_moves() 
    bestMove = legalMoves[0]
    for m in legalMoves :
        board.push(m)
        a = max(best,MinValue(board, a, b, limit-1, color, agent))
        board.pop()
        if a > best :
            bestMove = m
            best = a
    return bestMove

def go_score(b):
    res=[0,0]
    for i in b.get_board():
        if i!=0:
            res[i-1]+=1
    return res

#nb define the color of ai (0 for white and 1 for black)
def train(nb):
    win = [0]
    plot_scores = []
    plot_mean_scores = []
    plot_reward = []
    total_score = 0
    record = 0
    agent = Agent()
    screen = GobbanGameAI()
    while True:
        b = Goban.Board()

        players = []
        player1 = gnugoPlayer.myPlayer()
        player1.newGame(Goban.Board._WHITE)
        players.append(player1)

        player2 = gnugoPlayer.myPlayer()
        player2.newGame(Goban.Board._BLACK)
        players.append(player2)

        nextplayer = 0
        nextplayercolor = Goban.Board._BLACK
        nbmoves = 1

        outputs = ["",""]
        sysstdout= sys.stdout
        stringio = StringIO()
        wrongmovefrom = 0

        e1score=0
        e2score=0
        a1score=0
        a2score=0
        times = 0
        plot_reward+=[0]
        while not b.is_game_over():
            
            #print("Referee Board:")
            #b.prettyPrint() 
            #print("Before move", nbmoves)
            legals = b.legal_moves() # legal moves are given as internal (flat) coordinates, not A1, A2, ...
            #print("Legal Moves: ", [b.move_to_str(m) for m in legals]) # I have to use this wrapper if I want to print them
            nbmoves += 1
            otherplayer = (nextplayer + 1) % 2
            othercolor = Goban.Board.flip(nextplayercolor)
            sys.stdout = stringio

            if nextplayer != nb:
                lscore = go_score(b)
                e1score = lscore[(nb+1)%2]-lscore[nb]

                move = players[nextplayer].getPlayerMove() # The move must be given by "A1", ... "J8" string coordinates (not as an internal move)
                sys.stdout = sysstdout
                playeroutput = stringio.getvalue()
                stringio.truncate(0)
                stringio.seek(0)
                #print(("[Player "+str(nextplayer) + "] ").join(playeroutput.splitlines(True)))
                outputs[nextplayer] += playeroutput
                #print("Player ", nextplayercolor, players[nextplayer].getPlayerName(), "plays: " + move) #changed 
                if not Goban.Board.name_to_flat(move) in legals:
                    #print(otherplayer, nextplayer, nextplayercolor)
                    #print("Problem: illegal move")
                    wrongmovefrom = nextplayercolor
                    break
                b.push(Goban.Board.name_to_flat(move)) # Here I have to internally flatten the move to be able to check it.
            
                lscore = go_score(b)
                e2score = lscore[(nb+1)%2]-lscore[nb]

                nextplayer = otherplayer
                nextplayercolor = othercolor
            else:   
                done = False
                if nb==1:
                    bo2 = b.get_board()
                    for i in bo2:
                        if i == 1:
                            i=2
                        elif i == 2:
                            i=1
                    state_old = np.array(bo2, dtype=int)
                else:
                    state_old = np.array(b.get_board(), dtype=int)
                move = AmiAlphaBeta(b, 2, nextplayer, agent)
                
                lscore = go_score(b)
                a1score = lscore[nb]-lscore[(nb+1)%2]

                b.push(move) # Here I have to internally flatten the move to be able to check it.
                players[otherplayer].playOpponentMove(Goban.Board.flat_to_name(move))
                lscore = go_score(b)
                a2score = lscore[nb]-lscore[(nb+1)%2]
                
                nextplayer = otherplayer
                nextplayercolor = othercolor
                if(a1score != 0 or e1score != 0):
                    reward = (abs(a2score-a1score)-abs(e2score-e1score))
                    if reward > 0:
                        times += 10
                        reward = times + (reward)
                    else:
                        times = 0
                        reward = reward - 1
                else:
                    reward = 0
                state_new = np.array(b.get_board(), dtype=int)
                plot_reward[-1]+=reward
                # train short memory
                agent.train_short_memory(state_old, move, reward, state_new, done)

                # remember
                agent.remember(state_old, move, reward, state_new, done)

                
                #print("Player ", nextplayercolor, "IA ", "plays: " + Goban.Board.flat_to_name(move)) #changed 
                nextplayer = otherplayer
                nextplayercolor = othercolor
            
            screen.display.fill(BACKGROUND)
            bo = b.get_board()
            for n in range(len(bo)):
                if bo[n] == 1:
                    pygame.draw.rect(screen.display, WHITE, pygame.Rect(100*(n//9), 100*(n%9), BLOCK_SIZE, BLOCK_SIZE))
                if bo[n] == 2:
                    pygame.draw.rect(screen.display, BLACK, pygame.Rect(100*(n//9),100*(n%9), BLOCK_SIZE, BLOCK_SIZE))
            lscore = go_score(b)
            text = font.render("Score: " + str(lscore[nb]-lscore[(nb+1)%2]), True, RED)
            screen.display.blit(text, [0, 0])
            pygame.display.flip()
            time.sleep(SPEED)

        #print("The game is over")
        #b.prettyPrint()
       
        agent.n_games += 1
        agent.train_long_memory()

        lscore = go_score(b)
        score = lscore[nb]-lscore[(nb+1)%2]
        if score > 0:
            win+=[(win[-1]/100*(len(win)-1)+1)*100/len(win)]
        else:
            win+=[(win[-1]/100*(len(win)-1))*100/len(win)]
        if score > record:
            record = score
        if (len(plot_scores)!=0 and len(plot_scores)%10==0):
            agent.model.save()

        print('Game', agent.n_games, 'Score', score, 'Record:', record)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores, win, plot_reward)



while 1==1:
    if __name__ == '__main__':
        train(random.randint(0,0))