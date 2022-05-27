import random
import time
from copy import deepcopy

import pygame
import math

class connect4Player(object):
	def __init__(self, position, seed=0):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)

	def play(self, env, move):
		move = [-1]

class human(connect4Player):

	def play(self, env, move):
		move[:] = [int(input('Select next move: '))]
		while True:
			if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
				break
			move[:] = [int(input('Index invalid. Select next move: '))]

class human2(connect4Player):

	def play(self, env, move):
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
					else:
						pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move[:] = [col]
					done = True

class randomAI(connect4Player):

	def play(self, env, move):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move[:] = [random.choice(indices)]

class stupidAI(connect4Player):

	def play(self, env, move):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move[:] = [3]
		elif 2 in indices:
			move[:] = [2]
		elif 1 in indices:
			move[:] = [1]
		elif 5 in indices:
			move[:] = [5]
		elif 6 in indices:
			move[:] = [6]
		else:
			move[:] = [0]


class minimaxAI(connect4Player):

	player1 = True

	def play(self, env, move):

		player = env.turnPlayer.position

		if self.player1 is True:

			move[:] = [3]
			self.player1 = False
			return

		else:
			move[:] = [self.minimax(env, 3, player, 3)[0]]

	def minimax(self, env, depth, player, prev_move):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p:
				indices.append(i)

		if len(indices) == 0:
			return prev_move, 0, None

		for column in indices:
			if depth == 0:
				return prev_move, self.evaluation(env), None

			elif player == 1:
				value = -math.inf
				env_copy = deepcopy(env)
				env_copy.visualize = False
				self.simulateMove(env_copy, column, 1)
				if env_copy.gameOver(column, player):
					return column, 999999999, None
				new_score = self.minimax(env_copy, depth - 1, 2, prev_move)
				if value < new_score[1]:
					best_move = new_score[0]
					value = new_score[1]
				else:
					best_move = column

			elif player == 2:
				value = math.inf
				env_copy = deepcopy(env)
				env_copy.visualize = False
				self.simulateMove(env_copy, column, 2)
				if env_copy.gameOver(column, player):
					return column, -999999999, None
				new_score = self.minimax(env_copy, depth - 1, 1, prev_move)
				if value >new_score[1]:
					best_move = new_score[0]
					value = new_score[1]
				else:
					best_move = column

		return best_move, value, None

	def simulateMove(self, env, move, player):
		env.board[env.topPosition[move]][move] = player
		env.topPosition[move] -= 1
		env.history[player - 1].append(move)
		env.turnPlayer.position = env.turnPlayer.opponent.position

	def evaluation(self, env):
		eval = 138
		total = 0
		for row in range(ROW_COUNT):
			for col in range(COLUMN_COUNT):
				if env.board[row][col] == 1:
					total += EVALUATIONTABLE[row][col]
				elif env.board[row][col] == 2:
					total -= EVALUATIONTABLE[row][col]
		return total + eval


class alphaBetaAI(connect4Player):
	player1 = True

	def play(self, env, move):

		player = env.turnPlayer.position

		if self.player1 is True:

			move[:] = [3]
			self.player1 = False
			return

		else:
			move[:] = [self.alphaBeta(env, 3, player, -math.inf, math.inf, 3)[0]]

	def alphaBeta(self, env, depth, player, alpha, beta, prev_move):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p:
				indices.append(i)

		if len(indices) == 0:
			return prev_move, 0, None

		for column in indices:
			if depth == 0:
				return prev_move, self.evaluation(env), None

			elif player == 1:
				value = -math.inf
				env_copy = deepcopy(env)
				env_copy.visualize = False
				self.simulateMove(env_copy, column, 1)
				if env_copy.gameOver(column, player):
					return column, 999999999, None
				new_score = self.alphaBeta(env_copy, depth - 1, 2, alpha, beta, prev_move)
				if value < new_score[1]:
					best_move = new_score[0]
					value = new_score[1]
				else:
					best_move = column
				alpha = max(alpha, value)
				if alpha >= beta:
					break

			elif player == 2:
				value = math.inf
				env_copy = deepcopy(env)
				env_copy.visualize = False
				self.simulateMove(env_copy, column, 2)
				if env_copy.gameOver(column, player):
					return column, -999999999, None
				new_score = self.alphaBeta(env_copy, depth - 1, 1, alpha, beta, prev_move)
				if value >new_score[1]:
					best_move = new_score[0]
					value = new_score[1]
				else:
					best_move = column
				beta = min(beta, value)
				if alpha >= beta:
					break

		return best_move, value, None

	def simulateMove(self, env, move, player):
		env.board[env.topPosition[move]][move] = player
		env.topPosition[move] -= 1
		env.history[player - 1].append(move)
		env.turnPlayer.position = env.turnPlayer.opponent.position

	def evaluation(self, env):
		eval = 138
		total = 0
		for row in range(ROW_COUNT):
			for col in range(COLUMN_COUNT):
				if env.board[row][col] == 1:
					total += EVALUATIONTABLE[row][col]
				elif env.board[row][col] == 2:
					total -= EVALUATIONTABLE[row][col]
		return total + eval


EVALUATIONTABLE = [ [3,4,5,7,5,4,3], [4,6,8,10,8,6,4], [5,8,11,13,11,8,5], [5,8,11,13,11,8,5], [4,6,8,10,8,6,4], [3,4,5,7,5,4,3] ]

SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)