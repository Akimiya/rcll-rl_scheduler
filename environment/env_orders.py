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
    if rnd <= 0.33:
        num_rings = 1
    elif rnd <= 0.66:
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
    
    return [base] + rings + [cap] + num_products + competitive + delivery_window, num_rings


class env_rcll():
    def __init__(self):
        # rewards
        self.SENSELESS_ACTION = -20
        self.CORRECT_STEP = 10
        self.DISCART_ORDER = -2
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
        self.orders = []
        self.complexities = []
        for _ in range(self.num_orders):
            order, complexity = create_order()
            self.orders.append(order)
            self.complexities.append(complexity)
        self.order_stage = 0 # track what assembly step we are at
        self.pipeline = [0] * 4
        self.pipeline_cap = 0 # track cap seperately for each pipeline as not needed in state
        self.doing_order = []
        
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
            
        
        ##### UPDATING PRODUCT
        # format is a  two digit integer, first the category, second the color
        assert action >= 0 and action <= 99
        action_type = int(action / 10)
        action_color = action % 10
        
        # applying the base
        if action_type == 1:
            self.pipeline[0] = action_color
            # TODO: random delay model, best a function
        # applying the ring
        elif action_type == 2:
            # find next free (=0) ring slot
            if 0 in self.pipeline[1:4]:
                free_ring = self.pipeline.index(0) # TODO: Need to track manuaally at which ring
                self.pipeline[free_ring] = action_color
            else:
                # prelimary end if we have too many rings
                reward = self.INCORRECT_STEP
                done = True
                return self.get_observation(), reward, done
        # applying the cap
        elif action_type == 3:
            self.pipeline_cap = action_color
        # discarding the product
        elif action_type == 4:
            # TODO: needs to handle cases where it gets positive reward and discards indefinitely
            self.order_stage = 0 # track what assembly step we are at
            self.pipeline = [0] * 4
            self.pipeline_cap = 0 # track cap seperately for each pipeline as not needed in state
            self.doing_order = []
            
            reward = self.DISCART_ORDER
            if self.episode_step >= 11:
                done = True
            return self.get_observation(), reward, done
        
        
        # TODO: update reward and update into one
        ##### GETTING REWARD
        if self.order_stage == 0 and action_type == 1:
            found = False
            for idx, order in enumerate(self.orders):
                if order[0] == action_color:
                    reward = self.CORRECT_STEP
                    self.doing_order.append(idx)
                    found = True
                    
            if not found:
                reward = self.INCORRECT_STEP
                done = True
            else:
                self.order_stage = 1
            
        elif self.order_stage == 1 and action_type == 2:
            found = []
            for idx in self.doing_order:
                if self.orders[idx][free_ring] == action_color:
                    reward = self.CORRECT_STEP
                    found.append(idx)
                    
            if not found:
                reward = self.INCORRECT_STEP
                done = True
            else:
                self.doing_order[:] = found
                self.order_stage = 2 # have at least one ring

        # got another ring
        elif self.order_stage == 2 and action_type == 2:
            found = []
            has_ring = []
            for idx in self.doing_order:
                if self.orders[idx][free_ring] == action_color:
                    reward = self.CORRECT_STEP
                    found.append(idx)
                if self.orders[idx][free_ring] != 0:
                    has_ring.append(idx)

            if not found:
                reward = self.INCORRECT_STEP
#                if not has_ring:
                done = True # as it is not recoverable from having too many rings
            else:
                self.doing_order[:] = found
                    
        # got the cap
        elif self.order_stage == 2 and action_type == 3:
            done = True # as it is not recoverable or finished
            reward = self.INCORRECT_STEP # default to be overwritten
            
            tmp = []
            for idx in self.doing_order:
                if self.orders[idx][4] == action_color:
                    # check is full order is complete, pipeline does not include cap
                    for i, part in  enumerate(self.orders[idx][:-1]):
                        if self.pipeline[i] == part:
                            found = True # we want always this branch
                        else:
                            found = False
                            break
                    
                    if found:
                        reward = self.FINISHED_ORDER
                        tmp.append(idx)
                        break # if the complete check passes we leave completely
                    
            self.doing_order[:] = tmp
            
        # senseless action
        else:
            reward = self.SENSELESS_ACTION
            done = True
        
        # we stop if we try for too long
        if self.episode_step >= 11:
            done = True
        
        return self.get_observation(), reward, done


if __name__ == "__main__":
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("Please import the file")