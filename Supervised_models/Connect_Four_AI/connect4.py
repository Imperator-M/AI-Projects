import numpy as np
import math
import os, sys
import pygame
import random
import time
from threading import Thread
from thread import KillableThread, thread_with_exception, thread_with_trace
import multiprocessing
import signal
from copy import deepcopy

import threading

def time_limit(func, args, time_):

	'''Python tries very, very hard to make sure you can't kill threads,
	but with enough effort, anything is possible. Here, we uses traces
	to inject a system exit exception on the next line of whatever the
	thread is executing. I am fairly certain this can kill anything.

	You probably should not use this function because killing threads
	is bad practice. I am only doing it here because we need to make sure
	we have a level playing field ie no agent can cheat and get extra time
	per moves. If you want to do something similar you should keep an exit
	flag in your code, but asking every student to keep exit flags in their
	code in not feasible. This took an embarassingly long time to figure out.'''
	t = thread_with_trace(target=func, args=args)
	t.start()
	t.join(time_)
	if t.is_alive():
		t.kill()

class connect4():
	def __init__(self, player1, player2, board_shape=(6,7), visualize=False, game=0, save=False,
		limit_players=[-1,-1], time_limit=[-1,-1], verbose=False):

		global screen

		self.shape = board_shape
		width = self.shape[1] * SQUARESIZE
		height = (self.shape[0] + 1) * SQUARESIZE
		pygame.init()
		screen = pygame.display.set_mode(size)
		# Continue initialization
		self.board = np.zeros(board_shape).astype('int32')
		self.topPosition = (np.ones(board_shape[1]) * (board_shape[0]-1)).astype('int32')
		self.player1 = player1
		self.player2 = player2
		self.player1.opponent = self.player2
		self.player2.opponent = self.player1
		self.visualize = visualize
		self.turnPlayer = self.player1
		self.history = [[], []]
		self.game = game
		self.save = save
		self.limit = limit_players
		self.time_limits = time_limit
		self.verbose = verbose
		# Make sure time limits are formatted acceptably
		if len(self.time_limits) != 2:
			self.time_limits = [0.5,0.5]
		if self.time_limits[0] <= 0:
			self.time_limits[0] = 0.5
		if self.time_limits[1] <= 0:
			self.time_limits[1] = 0.5

	def playTurn(self):
		move = self.randMove()
		if self.turnPlayer.position in self.limit:
			time_limit(self.turnPlayer.play, (self,move,), self.time_limits[self.turnPlayer.position-1])
		else:
			self.turnPlayer.play(self, move)
		# Move returned as list because lists are mutable
		move = move[0]
		# Correct illegal move (assign random)
		if self.topPosition[move] < 0:
			possible = self.topPosition >= 0
			indices = []
			for i, p in enumerate(possible):
				if p: indices.append(i)
			move = random.choice(indices)
		self.board[self.topPosition[move]][move] = self.turnPlayer.position
		self.topPosition[move] -= 1
		playerID = self.turnPlayer.position
		self.history[playerID-1].append(move)
		self.turnPlayer = self.turnPlayer.opponent
		if self.visualize:
			self.draw_board()
		if self.verbose:
			print(self.board)
		return move

	def play(self):
		if self.visualize:
			self.draw_board()
		player = self.turnPlayer.position
		move = self.playTurn()
		while not self.gameOver(move, player):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()
			player = self.turnPlayer.position
			move = self.playTurn()
		if self.save:
			self.saveGame()
		if self.verbose:
			if len(self.history[0]) + len(self.history[1]) == self.shape[0] * self.shape[1]:
				print('The game has tied')
			else:
				print('Player ', self.turnPlayer.opponent.position, ' has won')
		spectating = True
		while spectating and self.visualize:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()
					spectating = False
					break

	def gameOver(self, j, player):
		# Find extrema to consider
		i = self.topPosition[j] + 1
		minRowIndex = max(j - 3, 0)
		maxRowIndex = min(j + 3, self.shape[1]-1)
		maxColumnIndex = max(i - 3, 0)
		minColumnIndex = min(i + 3, self.shape[0]-1)
		minLeftDiag = [max(j - 3, j), min(i + 3, self.shape[0]-1)]
		maxLeftDiag = [min(j + 3, self.shape[1]-1), max(i - 3, 0)]
		minRightDiag = [min(j + 3, j), min(i + 3, self.shape[0]-1)]
		maxRightDiag = [max(j - 3, 0), max(i - 3, 0)]
		# Iterate over extrema to find patterns
		# Horizontal solutions
		count = 0
		for s in range(minRowIndex, maxRowIndex+1):
			if self.board[i, s] == player:
				count += 1
			else:
				count = 0
			if count == 4:
				if self.visualize:
					pygame.draw.line(screen, WHITE, (int(s*SQUARESIZE+SQUARESIZE/2), int((i+1.5)*SQUARESIZE)), (int((s-4)*SQUARESIZE+SQUARESIZE+SQUARESIZE/2), int((i+1.5)*SQUARESIZE)), 5)
					pygame.display.update()
				return True
		# Verticle solutions
		count = 0
		for s in range(maxColumnIndex, minColumnIndex+1):
			if self.board[s, j] == player:
				count += 1
			else:
				count = 0
			if count == 4:
				if self.visualize:
					pygame.draw.line(screen, WHITE, (int(j*SQUARESIZE+SQUARESIZE/2), int((s+2)*SQUARESIZE)), (int(j*SQUARESIZE+SQUARESIZE/2), int((s-2)*SQUARESIZE)), 5)
					pygame.display.update()
				return True
		# Left diagonal
		row = i
		col = j
		count = 0
		up = 0
		while row > -1 and col > -1 and self.board[row][col] == player:
			count += 1
			row -= 1
			col -= 1
		down_count = count
		row = i + 1
		col = j + 1
		while row < self.shape[0] and col < self.shape[1] and self.board[row][col] == player:
			count += 1
			row += 1
			col += 1
		if count >= 4:
			if self.visualize:
					# top, bottom
					pygame.draw.line(screen, WHITE, (int((j+0.5-(down_count-1))*SQUARESIZE), int((i+1.5-(down_count-1))*SQUARESIZE)), (int((j+0.5+(4-down_count))*SQUARESIZE), int((i+1.5+(4-down_count))*SQUARESIZE)), 5)
					pygame.display.update()
			return True
		# Right diagonal
		row = i
		col = j
		count = 0
		while row < self.shape[0] and col > -1 and self.board[row][col] == player:
			count += 1
			row += 1
			col -= 1
		down_count = count
		row = i - 1
		col = j + 1
		while row > -1 and col < self.shape[1] and self.board[row][col] == player:
			count += 1
			row -= 1
			col += 1
		if count >= 4:
			if self.visualize:
					# top, bottom
					pygame.draw.line(screen, WHITE, (int((j+0.5-(down_count-1))*SQUARESIZE), int((i+1.5+(down_count-1))*SQUARESIZE)), (int((j+0.5+(4-down_count))*SQUARESIZE), int((i+1.5-(4-down_count))*SQUARESIZE)), 5)
					pygame.display.update()
			return True
		return len(self.history[0]) + len(self.history[1]) == self.shape[0]*self.shape[1]

	def saveGame(self):
		with open(os.path.join('history', 'game'+str(self.game)+'P1.txt'), 'w') as filehandle:
			for item in self.history[0]:
				filehandle.write('%s\n' % item)
		with open(os.path.join('history', 'game'+str(self.game)+'P2.txt'), 'w') as filehandle:
			for item in self.history[1]:
				filehandle.write('%s\n' % item)

	def randMove(self):
		possible = self.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		return [random.choice(indices)]

	def getBoard(self):
		return deepcopy(self.board)

	def getEnv(self):
		return deepcopy(self)

	'''Pygame code used with permission from Keith Galli.
	Refer to https://github.com/KeithGalli/Connect4-Python for licensing'''

	def draw_board(self):
		for c in range(self.shape[1]):
			for r in range(self.shape[0]):
				pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
				pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
		
		for c in range(self.shape[1]):
			for r in range(self.shape[0]):		
				if self.board[r][c] == 1:
					pygame.draw.circle(screen, RED, (int((c)*SQUARESIZE+SQUARESIZE/2), height-int((5-r)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
				elif self.board[r][c] == 2: 
					pygame.draw.circle(screen, YELLOW, (int((c)*SQUARESIZE+SQUARESIZE/2), height-int((5-r)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
		pygame.display.update()


SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
WHITE = (255,255,255)

ROW_COUNT = 6
COLUMN_COUNT = 7

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = None








