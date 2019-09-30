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

def create_order(fill=False, amount=False, compet=False, window=False):
    # enum bases red = 1, black, silver
    base = int(np.random.uniform(1, 4))
    
    # number of rings (no information on distribution)
    rnd = np.random.uniform()
    if rnd <= 0.4:
        num_rings = 1
    elif rnd <= 0.7:
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
        self.SENSELESS_ACTION = -20
        self.CORRECT_STEP = 10
        self.INCORRECT_STEP = -10
        self.FINISHED_ORDER = 20
        
        self.ACTION_SPACE_SIZE = 3 + 4 + 2 + 1
    
    def get_observation(self):
        # as shape does not matter, we stack a vector for now
        # these are all orders and the pipeline in one
        observation = sum(self.orders, []) + self.pipeline
        
        return observation
    
    def render(self):
        print("orders: {} | pipeline: {}{}".format(self.orders, self.pipeline, self.pipeline_cap))
    
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
        self.doing_order = None
        
        return self.get_observation()

    def step(self, action):
        self.episode_step += 1
        done = False
        
        # action conversion for default networks with actions 0 to 9
        if action >= 9:
            action = 40
        elif action >= 7:
            action = 30 + action - 7 + 1
        elif action >= 3:
            action = 20 + action - 3 + 1
        elif action >= 0:
            action = 10 + action + 1
            
        
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
            # find next free (=0) ring slot
            if 0 in self.pipeline: # TODO: remove ugly
                free_ring = self.pipeline.index(0)
                self.pipeline[free_ring] = action_color
            else:
                # prelimary end if we have too many rings
                reward = self.INCORRECT_STEP
                done = True
                return self.get_observation(), reward, done
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
                done = True
            else:
                for idx, order in enumerate(self.orders):
                    if order[0] == action_color:
                        reward = self.CORRECT_STEP
                        no_such_order = False
                        self.doing_order = idx
                        break
                    else:
                        no_such_order = True
                if no_such_order:
                    reward = self.INCORRECT_STEP
                    done = True
            
                self.order_stage = 1
            
        elif self.order_stage == 1:
            if action_type != 2:
                reward = self.SENSELESS_ACTION
                done = True
            else:
                if self.orders[self.doing_order][free_ring] == action_color:
                    reward = self.CORRECT_STEP
                else:
                    reward = self.INCORRECT_STEP
                    done = True
            
                self.order_stage = 2 # have at least one ring
        elif self.order_stage == 2:
            # senseless action
            if action_type != 2 and action_type != 3:
                reward = self.SENSELESS_ACTION
                done = True
            else:
                # got another ring
                if action_type == 2:
                    if self.orders[self.doing_order][free_ring] == action_color:
                        reward = self.CORRECT_STEP
                        no_such_order = False
                    else:
                        no_such_order = True
                # got the cap
                elif action_type == 3:
                    if self.orders[self.doing_order][4] == action_color:
                        # check is full order is complete, pipeline does not include cap
                        for idx, part in  enumerate(self.orders[self.doing_order][:-1]):
                            if self.pipeline[idx] != part:
                                no_such_order = True
                                break
                            else:
                                no_such_order = False
                        
                        # will be overwritten if wrong, done in any case
                        reward = self.FINISHED_ORDER
                        done = True
                    else:
                        no_such_order = True
                            
                if no_such_order:
                    reward = self.INCORRECT_STEP
                    done = True
        
        return self.get_observation(), reward, done


if __name__ == "__main__":
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("Main closed")