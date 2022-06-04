import argparse
from time import time
from AIModule import StupidAI, Djikstras, AStarExp, AStarDiv, AStarMSH
from Map import Map

parser = argparse.ArgumentParser(description='Run programming assignment 1')
parser.add_argument('-w', default=500, type=int, help='Width of map')
parser.add_argument('-l', default=500, type=int, help='Length of map')
parser.add_argument('-start', nargs='+', type=int, help='Set the start point position')
parser.add_argument('-goal', nargs='+', type=int, help='Set the goal point position')
parser.add_argument('-seed', default=None, type=int, help='Seed for random generation')
parser.add_argument('-cost', default='exp', type=str, help='Cost function. Use any of the following: [exp, div]')
parser.add_argument('-AI', default='AStarExp', type=str, help='AI agent to use. Use any of the following: [AStarExp, AStarDiv, AStarMSH, Djikstra]')
parser.add_argument('-filename', default=None, type=str, help='Filepath for .npy file to be used for map')

args = parser.parse_args()

w = args.w
l = args.l
start = args.start
goal = args.goal
seed = args.seed
cost = args.cost
ai = args.AI
filename = args.filename

agents = {'AStarExp': AStarExp, 'AStarDiv': AStarDiv, 'AStarMSH': AStarMSH, 'Djikstra': Djikstras, 'StupidAI':StupidAI}

m = Map(w,l, seed=seed, filename=filename, cost_function=cost, start=start, goal=goal)
alg = agents[ai]()
t1 = time()
path = alg.createPath(m)
t2 = time()
print('Time (s): ', t2-t1)
m.createImage(path)