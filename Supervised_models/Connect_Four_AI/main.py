import argparse
from connect4 import connect4
from players import human2, stupidAI, randomAI, human, minimaxAI, alphaBetaAI
from montecarlo import monteCarloAI

parser = argparse.ArgumentParser(description='Run programming assignment 1')
parser.add_argument('-w', default=6, type=int, help='Rows of game')
parser.add_argument('-l', default=7, type=int, help='Columns of game')
parser.add_argument('-p1', default='human', type=str, help='Player 1 agent. Use any of the following: [human, humanTxt, stupidAI, randomAI, monteCarloAI, minimaxAI, alphaBetaAI]')
parser.add_argument('-p2', default='human', type=str, help='Player 2 agent. Use any of the following: [human, humanTxt, stupidAI, randomAI, monteCarloAI, minimaxAI, alphaBetaAI]')
parser.add_argument('-seed', default=0, type=int, help='Seed for random algorithms')
parser.add_argument('-visualize', default='True', type=str, help='Use GUI')
parser.add_argument('-verbose', default='True', type=str, help='Print boards to shell')
parser.add_argument('-limit_players', default='1,2', type=str, help='Players to limit time for. List players as numbers eg [1,2]')
parser.add_argument('-time_limit', default='0.5,0.5', type=str, help='Time limits for each player. Must be list of 2 elements > 0. Not used if player is not listed')

# Bools and argparse are not friends
bool_dict = {'True': True, 'False': False}

args = parser.parse_args()

w = args.w
l = args.l

seed = args.seed
visualize = bool_dict[args.visualize]
verbose = bool_dict[args.verbose]
limit_players = args.limit_players.split(',')
for i, v in enumerate(limit_players):
	limit_players[i] = int(v)
time_limit = args.time_limit.split(',')
for i, v in enumerate(time_limit):
	time_limit[i] = float(v)


agents = {'human': human2, 'humanTxt': human, 'stupidAI': stupidAI, 'randomAI': randomAI, 'monteCarloAI': monteCarloAI, 'minimaxAI': minimaxAI, 'alphaBetaAI': alphaBetaAI}

if __name__ == '__main__':
	player1 = agents[args.p1](1, seed)
	player2 = agents[args.p2](2, seed)
	c4 = connect4(player1, player2, board_shape=(w,l), visualize=visualize, limit_players=limit_players, time_limit=time_limit, verbose=verbose)
	c4.play()