import numpy as np
import random
import sys
import signal
from players import connect4Player
from copy import deepcopy

class monteCarloAI(connect4Player):

  def play(self, env, move):
    random.seed(self.seed)
    # Find legal moves
    env = deepcopy(env)
    env.visualize = False
    possible = env.topPosition >= 0
    indices = []
    for i, p in enumerate(possible):
      if p: indices.append(i)
    # Init fitness trackers
    vs = np.zeros(7)
    # Play until told to stop
    counter = 0
    while True:
      first_move = random.choice(indices)
      turnout = self.playRandomGame(deepcopy(env), first_move)
      if turnout == self.position:
        vs[first_move] += 1
      elif turnout != 0:
        vs[first_move] -= 1
      if counter % 100 == 0:
        move[:] = [np.argmax(vs)]
      counter += 1

  def playRandomGame(self, env, first_move):
    switch = {1:2,2:1}
    move = first_move
    player = self.position
    self.simulateMove(env, move, player)
    while not env.gameOver(move, player):
      player = switch[player]
      possible = env.topPosition >= 0
      indices = []
      for i, p in enumerate(possible):
        if p: indices.append(i)
      move = random.choice(indices)
      self.simulateMove(env, move, player)
    if len(env.history[0]) == 42: return 0
    return player

  def simulateMove(self, env, move, player):
    env.board[env.topPosition[move]][move] = player
    env.topPosition[move] -= 1
    env.history[0].append(move)

  def signal_handler(self):
    print("SIGTERM ENCOUNTERED")
    sys.exit(0)

