import Goban 
import myPlayer,IA,gnugoPlayer
from io import StringIO
import sys
import torch
import random
import numpy as np
from collections import deque,namedtuple
from model import Linear_QNet, QTrainer
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

BLOCK_SIZE = 100 # size of square in pygame
SPEED = 0.00000001 #speed of the pygame


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

######################## AI related function ############################
class Agent:

    def __init__(self):
        self.n_games = 0 # number of game played 
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(81,256,512,512,256,100) # Create the neural network topology

        # Load the model memory
        self.model.load_state_dict(torch.load("./model/model6.pth"))

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # Load the QTrainer 

    # Function that gives the parameters we will put in the input of neural network
    def get_state(b):
        state = b.get_board() # Board of the game
        return np.array(state, dtype=int)

    # Function that save parameters in the memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    # Function that train the neural network with all the samples
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    # Function that train the neural network with just the actual parameters 
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Function that gives the neural network's answer or a random answer depending on epsilon's value
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

####################################################################

#################### Tree-search algorithm #########################

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
        return agent.get_action(state) #return the answer of the agent
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
        return agent.get_action(state) #return the answer of the agent
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


####################################################################

###################### Annexe function #############################

def go_score(b):
    res=[0,0]
    for i in b.get_board():
        if i!=0:
            res[i-1]+=1
    return res

#####################################################################


####################### Training Program ############################

#nb define the color of ai (0 for white and 1 for black)

def train(nb):

    #variable for plotting
    win = [0]
    plot_scores = []
    plot_mean_scores = []
    plot_reward = []
    total_score = 0
    record = 0
    
    agent = Agent() # Instanciation of the Agent (AI)

    screen = GobbanGameAI() # Initialisation of the pygame screen

    while True: #Infinite training

        b = Goban.Board() #Creation of a new board

        players = [] #List of players (we will use this only for the enemy)

        #Enemy player if he plays the white
        player1 = gnugoPlayer.myPlayer()
        player1.newGame(Goban.Board._WHITE)
        players.append(player1)

        #Enemy player if he plays the black
        player2 = gnugoPlayer.myPlayer()
        player2.newGame(Goban.Board._BLACK)
        players.append(player2)

        player = 0 # player is a variable that determines what player have to play (0 : White, 1: Black). Here 0 because the white begin

        playercolor = Goban.Board._WHITE
        nbmoves = 0

        outputs = ["",""]
        sysstdout= sys.stdout
        stringio = StringIO()
        wrongmovefrom = 0

        e1score=0 #enemy score before the enemy plays
        e2score=0 #enemy score after the enemy plays
        a1score=0 #agent score before the agent plays
        a2score=0 #agent score after the agent plays

        plot_reward+=[0]

        while not b.is_game_over():
            
            legals = b.legal_moves() # legal moves are given as internal (flat) coordinates, not A1, A2, ...

            nbmoves += 1

            otherplayer = (player + 1) % 2 # Define the other player number (the next player)
            othercolor = Goban.Board.flip(playercolor) # Define the other color (the next color)

            sys.stdout = stringio

            if player != nb: #If it is not the agent's round

                #enemy score before the enemy plays
                lscore = go_score(b)
                e1score = lscore[(nb+1)%2]-lscore[nb]

                move = players[player].getPlayerMove() # The move must be given by "A1", ... "J8" string coordinates (not as an internal move)

                sys.stdout = sysstdout
                playeroutput = stringio.getvalue()
                stringio.truncate(0)
                stringio.seek(0)
                outputs[player] += playeroutput

                #verification of the validity of the move
                if not Goban.Board.name_to_flat(move) in legals:
                    wrongmovefrom = playercolor
                    break


                b.push(Goban.Board.name_to_flat(move)) # Here I have to internally flatten the move to be able to check it.
            
                #enemy score after the enemy plays
                lscore = go_score(b)
                e2score = lscore[(nb+1)%2]-lscore[nb]

                #change the player move and player color to the next one
                player = otherplayer 
                playercolor = othercolor

            else:   #if not, it is the turn of the agent

                done = False #we consider that the game is not finished

                if nb == 1: #if the player plays the black, we invert the "1" and 2" in the board (we need that for the agent)
                    bo2 = b.get_board()
                    for i in bo2:
                        if i == 1:
                            i=2
                        elif i == 2:
                            i=1
                    
                    state_old = np.array(bo2, dtype=int) #define the state before playing

                else:
                    state_old = np.array(b.get_board(), dtype=int) #define the state before playing

                move = AmiAlphaBeta(b, 2, player, agent) #we call our functions to get the move
                
                #agent score before the agent plays
                lscore = go_score(b)
                a1score = lscore[nb]-lscore[(nb+1)%2]

                b.push(move) # Here I have to internally flatten the move to be able to check it.
                players[otherplayer].playOpponentMove(Goban.Board.flat_to_name(move)) #we send the move to the other player

                #agent score after the agent plays
                lscore = go_score(b)
                a2score = lscore[nb]-lscore[(nb+1)%2]
                
               


                if(a1score != 0 or e1score != 0):

                    reward = (abs(a2score-a1score)-abs(e2score-e1score)) #this is the reward I use

                    if reward > 0: #if the player plays well
                        reward = 10
                    else:          #if he doesn't
                        reward = -10

                else: 
                    reward = 0

                state_new = np.array(b.get_board(), dtype=int) #New state we calculate for training

                plot_reward[-1] += reward #for plotting rewards

                # train short memory
                agent.train_short_memory(state_old, move, reward, state_new, done)

                # remember
                agent.remember(state_old, move, reward, state_new, done)

                #change the player move and player color to the next one
                player = otherplayer
                playercolor = othercolor
            
            #Pygame display system
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

        agent.n_games += 1
        
        agent.train_long_memory()

        #plotting 
        lscore = go_score(b)
        score = lscore[nb]-lscore[(nb+1)%2]
        if score > 0:
            win+=[(win[-1]/100*(len(win)-1)+1)*100/len(win)]
        else:
            win+=[(win[-1]/100*(len(win)-1))*100/len(win)]
        if score > record:
            record = score

        #Every x parties, we save the model
        if (len(plot_scores)!=0 and len(plot_scores)%10==0):
            agent.model.save()

        print('Game', agent.n_games, 'Score', score, 'Record:', record)

        #plotting
        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores, win, plot_reward)



#starting lines 
while 1==1:
    if __name__ == '__main__':
        train(random.randint(0,1)) #choose randomly the start player