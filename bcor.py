import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cpu')


def scale(x):
    return ((x*1.8)/600) - 1.2


def collect_random_interaction_data(num_iters):
    state_next_state = []
    actions = []
    env = gym.make('MountainCar-v0')

    obs = env.reset()

    done = False
    while not done:
        a = env.action_space.sample()
        next_obs, reward, done, info = env.step(a)
        state_next_state.append(np.concatenate((obs,next_obs), axis=0))
        actions.append(a)
        obs = next_obs
    env.close()

    return np.array(state_next_state), np.array(actions)


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='single_rgb_array')
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos

def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s,a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).to(device)

    return obs_torch, acs_torch, obs2_torch

class Policy(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        #This layer has 2 inputs corresponding to car position and velocity
        self.fc1 = nn.Linear(2, 8)
        #This layer has three outputs corresponding to each of the three discrete actions
        self.fc2 = nn.Linear(8, 3)



    def forward(self, x):
        #this method performs a forward pass through the network, applying a non-linearity (ReLU) on the
        #outputs of the first layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class InvDynamicsModel(nn.Module):
    '''
        Neural network with that maps (s,s') state to a prediction
        over which of the three discrete actions was taken.
        The network should have three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        #This network should take in 4 inputs corresponding to car position and velocity in s and s'
        # and have 3 outputs corresponding to the three different actions
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
        #################
        #TODO:
        #################

    def forward(self, x):
        #this method performs a forward pass through the network
        ###############
        #TODO:
        ###############
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        return x


def train_inv(states,actions,inv_dyn,num_train_iters):
    optimizer = Adam(inv_dyn.parameters(), lr=0.02)
    # action space is discrete so our policy just needs to classify which action to take
    # we typically train classifiers using a cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()

    # train inverse dynamics model in one big batch
    ####################
    # TODO you may need to tune the num_train_iters
    ####################
    # number of times to run gradient descent on training data
    for i in range(num_train_iters):
        # zero out automatic differentiation from last time
        optimizer.zero_grad()
        # run each state in batch through policy to get predicted logits for classifying action
        pred_action_logits = inv_dyn(states)
        # now compute loss by comparing what the policy thinks it should do with what the demonstrator didd
        loss = loss_criterion(pred_action_logits, actions)
        #print("iteration", i, "bc loss", loss)
        # back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        # perform update on policy parameters
        optimizer.step()


def train_policy(obs, acs, nn_policy, num_train_iters):
    pi_optimizer = Adam(nn_policy.parameters(), lr=0.1)
    # action space is discrete so our policy just needs to classify which action to take
    # we typically train classifiers using a cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()

    # run BC using all the demos in one giant batch
    for i in range(num_train_iters):
        # zero out automatic differentiation from last time
        pi_optimizer.zero_grad()
        # run each state in batch through policy to get predicted logits for classifying action
        pred_action_logits = nn_policy(obs)
        # now compute loss by comparing what the policy thinks it should do with what the demonstrator didd
        loss = loss_criterion(pred_action_logits, acs)
        #print("iteration", i, "bc loss", loss)
        # back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        # perform update on policy parameters
        pi_optimizer.step()


def eval_inv(inv_dyn, s_s2_torch, acs):
    outputs = inv_dyn(s_s2_torch[:10])
    _, predicted = torch.max(outputs, 1)
    #print(type(s_s2_torch[:1]))
    print("checking predictions on first 10 actions from random interaction data")
    print("predicted actions", predicted)
    print("actual actions", acs[:10])


def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0",render_mode='human')
    else:
        env = gym.make("MountainCar-v0")

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the
            #argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)
            obs, rew, done, info = env.step(action)
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))



def reinforce(num_iter,inv_dyn,pi):
    env = gym.make('MountainCar-v0')
    env.reset()
    done = False
    #x = 1000
    #predicted = env.action_space.sample()
    observation = env.reset()
    total_reward = 0
    s = []
    a = []
    reinf = False
    while not done:
        #if isinstance(predicted,int):
            #observation, reward, done, info = env.step(env.action_space.sample())
        if not reinf:
            predicted = torch.argmax(pi(torch.from_numpy(observation).unsqueeze(0))).item()
            observation, reward, done, info = env.step(predicted)
        else:
            observation, reward, done, info = env.step(predicted[0].item())
        s.append(observation)
        total_reward += reward
        env.render(mode="human")
        left, middle, right = pygame.mouse.get_pressed()
        input = []
        input_torch = []
        if left:
            x, y = pygame.mouse.get_pos()
            x = scale(x)
            reinf = True
        if reinf:
            if abs(observation[0] - x) <= 0.01:
                reinf = False
            input.append(x)
            input.append(observation[1])
            input = np.array(input)
            #print(input)
            #print(observation)
            input_torch.append(np.concatenate((observation, input), axis=0))
            #print(input_torch)
            input_torch = torch.from_numpy(np.array(input_torch)).float().to(device)
            #print(input_torch)
            outputs = inv_dyn(input_torch)
            _, predicted = torch.max(outputs, 1)
        if not reinf:
            if isinstance(predicted,int):
                a.append(predicted)
            else:
                a.append(predicted[0].item())
        else:
            a.append(predicted[0].item())
    print(total_reward)
    return np.array(s), np.array(a)


if __name__ == "__main__":
    ###### collecting random data to train inverse dynamic model
    states, actions = collect_random_interaction_data(50)
    ###### converting data to tensor format
    s_s2_torch = torch.from_numpy(np.array(states)).float().to(device)
    a_torch = torch.from_numpy(np.array(actions)).to(device)
    ###### initialise inv dynamic model
    inv_dyn = InvDynamicsModel()
    ###### train inv dynamic model
    train_inv(s_s2_torch,a_torch,inv_dyn,2000)
    ###### evaluate inv dynamic model
    eval_inv(inv_dyn,s_s2_torch,actions)
    demos = collect_human_demos(1)
    obs, acs, _ = torchify_demos(demos)

    # train policy
    pi = Policy()
    train_policy(obs, acs, pi, 2000)

    s,a = reinforce(1,inv_dyn,pi)
    obs_torch = torch.from_numpy(np.array(s)).float().to(device)
    acs_torch = torch.from_numpy(np.array(a)).to(device)
    policy = Policy()
    train_policy(obs_torch, acs_torch, policy, 2000)
    evaluate_policy(policy, 2)
