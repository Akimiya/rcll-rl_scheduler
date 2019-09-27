#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import time
import struct
import traceback
import datetime as dt
import numpy as np
from enum import Enum
from copy import deepcopy # more performant then dict()

# TODO: need to make refbox_comm into a class or uglily import it to get running globals

# utility objects
class Base():
    empty = 0
    red = 1
    black = 2
    silver = 3
BASE = Base()

class Ring():
    empty = 0
    blue = 1
    green = 2
    orange = 3
    yellow = 4
RING = Ring()

class Cap():
    empty = 0
    black = 1
    grey = 2
CAP = Cap()



def create_order(fill=False, amount=False, compet=False, window=False):
    # enum bases red = 1, black, silver
    base = int(np.random.uniform(1, 4))
    
    # number of rings (no information on distribution)
    rnd = np.random.uniform()
    if rnd <= 0.5:
        num_rings = 1
    elif rnd <= 0.8:
        num_rings = 2
    elif rnd <= 1:
        num_rings = 3
    # enum rings blue = 1, green, oragne, yellow
    rings = [int(np.random.uniform(1, 5)) for _ in range(num_rings)] + [0] * (3 - num_rings)
    
    # enum caps black = 1, grey
    cap = int(np.random.uniform(1, 3))
    
    # number of requested products
    if amount == True:
        num_products = [int(np.random.uniform(1, 3))]
    else:
        num_products = [0] if fill else []
    
    # if order is competitive
    if compet == True:
        competitive = [int(np.random.uniform(0, 2))]
    else:
        competitive = [0] if fill else []
        
    # the delivery window
    minimum_window = 120
    if window == True:
        start = int(np.random.uniform(1, 1021 - minimum_window))
        end = int(np.random.uniform(start + minimum_window, 1021))
        delivery_window = [start, end]
    else:
        delivery_window = [0, 0] if fill else []
    
    return [base] + rings + [cap] + num_products + competitive + delivery_window


class env_rcll():
    def __init__(self):
        # rewards
        self.SENSELESS_ACTION = -30
        self.CORRECT_STEP = 1
        self.INCORRECT_STEP = -10
    
    
    # reset all parameters to an initial game state
    def reset(self):
        # utility parameters
        self.episode_step = 0
        
        self.num_orders = 3
        # main features
        self.orders = [create_order() for _ in range(self.num_orders)]
        self.order_stage = 0 # track what assembly step we are at
        self.pipeline = [0] * 4
        self.pipeline_cap = 0 # track cap seperately for each pipeline as not needed in state
        
        # as shape does not matter, we stack a vector for now
        observation = sum(self.orders, []) + self.pipeline
        
        return observation

    def step(self, action):
        self.episode_step += 1
        done = False
        
        ### make an action (without rewards jet)
        # format is a  two digit integer, first the category, second the color
        assert action >= 0 and action <= 99
        action_type = int(action / 10)
        action_color = action % 10
        
        # getting a base
        if action_type == 1:
            self.pipeline[0] = action_color
            # TODO: random delay model, best a function
        elif action_type == 2:
            # find next free ring slot
            free_ring = next(i for i in self.pipeline[1:] if i == 0) + 1
            self.pipeline[free_ring] = action_color
        # getting a cap
        elif action_type == 3:
            self.pipeline_cap = action_color
        # discard product
        elif action_type == 4:
            self.pipeline[:] = [0] * 4
            self.pipeline_cap = 0
        
        ### getting reward
        if self.order_stage == 0:
            # senseless action
            if action_type != 1:
                reward = self.SENSELESS_ACTION
#                done = True
            else:
                for order in self.orders:
                    if order[0] == action_color:
                        reward = self.CORRECT_STEP
                        no_such_order = False
                        break
                    else:
                        no_such_order = True
                if no_such_order:
                    reward = self.INCORRECT_STEP
            
                self.order_stage = 1
            
        elif self.order_stage == 1:
            if action_type != 2:
                reward = self.SENSELESS_ACTION
#                done = True
            else:
                for order in self.orders:
                    # using free_ring from the action step
                    if order[free_ring] == action_color:
                        reward = self.CORRECT_STEP
                        no_such_order = False
                        break
                    else:
                        no_such_order = True
                if no_such_order:
                    reward = self.INCORRECT_STEP
            
                self.order_stage = 2 # have at least one ring
        elif self.order_stage == 2:
            # senseless action
            if action_type != 2 and action_type != 3:
                reward = self.SENSELESS_ACTION
#                done = True
            else:
                for order in self.orders:
                    if order[0] == action_color:
                        reward = self.CORRECT_STEP
                        no_such_order = False
                        break
                    else:
                        no_such_order = True
                if no_such_order:
                    reward = self.INCORRECT_STEP
            
                self.order_stage = 1
        

        if reward == self.FOOD_REWARD or reward == - self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done


if __name__ == "__main__":
    while True:
        print(create_order())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("Main closed")