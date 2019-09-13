#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
 
import keras
import numpy as np
from collections import deque
import random

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
#env = gym.make("SuperMarioBros-1-1-v0")

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=None))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(env.action_space.n, init="uniform", activation="linear"))

### Parameters
# Register where the actions will be stored
D = deque()
# Number of timesteps we will be acting on the game and observing results
observetime = 500
# Probability of doing a random move
epsilon = 0.7
# Discounted future reward. How much we care about steps further in time
gamma = 0.9
# Learning minibatch size
mb_size = 50


### Step 1: Make Observations
# start game
observation = env.reset()
# initialize state
state = np.stack((observation, observation), axis=1)

done = False
# start observing
for t in range(observetime):
    # Q-values predictions
    Q = model.predict(state)
    # Move with highest Q-value is the chosen one
    action = np.argmax(Q)
    # perform an action
    state, reward, done, info = env.step(action)
    # Remember action and consequence
    D.append((state, action, reward, state_new, done))
    # update state
    state = state_new
    # restart game if it's finished
    if done:
        env.reset()
        
        
        
### Step 2: Learning from Observations
# Sample some moves
minibatch = random.sample(D, mb_size)
for i in range(0, mb_size):
    state = minibatch[i][0]
    action = minibatch[i][1]
    reward = minibatch[i][2]
    state_new = minibatch[i][3]
    done = minibatch[i][4]
    
    # Build Bellman equation for the Q function
    inputs[i:i+1] = np.expand_dims(state, axis=0)
    targets[i] = model.predict(state)
    Q_sa = model.predict(state_new)
    
    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)

    # Train network to output the Q function
    model.train_on_batch(inputs, targets)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    