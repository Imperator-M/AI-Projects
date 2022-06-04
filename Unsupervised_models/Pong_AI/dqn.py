import matplotlib
matplotlib.use('Agg')
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)

            # TODO: Given state, you should write code to get the Q value and chosen action

            #state_detached = state.detach().cpu().numpy()
            #values = self.forward(state_detached)
            #action = np.argmax(self.forward(state_detached))

            #action = torch.argmax(self(state))

            #action = np.argmax(values)
            #action = torch.argmax(self(state_detached))
            #action = self(state)
            action = torch.argmax(self(state))

        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

estimated_next_q_state_values = []
estimated_next_q_value = []

def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    #state = Variable(torch.FloatTensor(np.float32(state)))
    #next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    #action = Variable(torch.LongTensor(action))
    #reward = Variable(torch.FloatTensor(reward))
    #done = Variable(torch.FloatTensor(done))

    # implement the loss function here
    #squared_error = (y_pred - y_true) ** 2
    #sum_squared_error = np.sum(squared_error)
    #loss = sum_squared_error / y_true.size

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    
    state_action_values = model(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_state_values = target_model(next_state).max(1)[0]
    #next_state_values[done] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + reward
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
   #q_values = model(state)
   # next_q_values = model(next_state)
   # next_q_state_values = target_model(next_state)

   # estimated_next_q_state_values.append(next_q_state_values)


   # q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
   # next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

   # estimated_next_q_value.append(next_q_value)

   # expected_q_value = reward + gamma * next_q_value * (1 - done)

    #loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
   # loss = nn.MSELoss(reduction = 'sum')(q_values, Variable(expected_q_value.data))

    #optimizer.zero_grad()
    #loss.backwards()
    #optimizer.step()
    #state_action_values = model(next_state).gather(1, action.unsqueeze(-1)).squeeze(-1)
    #next_state_values = target_model(next_state).max(1)[0]
    #next_state_values[torch.ByteTensor(done)] = 0.0
    #next_state_values = next_state_values.detach()

    #expected_state_action_values = next_state_values * gamma + reward

    #loss = torch.nn.MSELoss(reduction = "sum")(state_action_values, expected_state_action_values)

    #mse = torch.nn.MSELoss(reduction = "sum")(next_state, action)
    #loss = mse/action.size()

  #squarred_error = ((action - done)**2)
    #sum_squarred_error = .5*np.sum(squarred_error)
    #loss = sum_squarred_error/done.size

  ##N = len(y_train)
    ##loss = (1/N) * np.sum((y_train - y_pred)**2)

    return loss

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer

        indices = np.random.choice(len(self.buffer), batch_size, replace = False)
        state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in indices])

        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
