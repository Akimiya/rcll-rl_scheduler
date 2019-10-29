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

def create_order(C=-1, fill=False, amount=False, compet=False, window=-1):
    # enum bases red = 1, black, silver
    base = int(np.random.uniform(1, 4))
    
    # number of rings
    if C == -1:
        rnd = np.random.uniform()
        if rnd <= 1 / 9:
            num_rings = 3
        elif rnd <= 2 / 9:
            num_rings = 2
        elif rnd <= 4 / 9:
            num_rings = 1
        elif rnd <= 1:
            num_rings = 0
            
    elif C >= 0 and C <= 3:
        num_rings = C
    # enum rings blue = 1, green, oragne, yellow; cant have more then one of same ring
    ring_options = [x for x in range(1,5)]
    np.random.shuffle(ring_options)        
    rings = ring_options[:num_rings] + [0] * (3 - num_rings)
    
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
    if window >= 0:
        start = int(np.random.uniform(window + 1, 1021 - minimum_window))
        end = int(np.random.uniform(start + minimum_window, 1021))
        delivery_window = [start, end]
    else:
        delivery_window = [0, 0] if fill else []
    
    return [base] + rings + [cap] + num_products + competitive + delivery_window

class field_pos():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return "({}, {})".format(self.x, self.y)
    
    def __eq__(self, other):
        return (self.x, self.y) == other

    def __ne__(self, other):
        return not(self.__eq__(other))
    
    def __sub__(self, other):
        return field_pos(self.x - other.x, self.y - other.y)
    
    def __add__(self, other):
        return field_pos(self.x + other.x, self.y + other.y)
    
    def distance(self, other):
        tmp = self - other
        return np.sqrt(tmp.x**2 + tmp.y**2)
        

class env_rcll():
    def __init__(self, normalize=False):
        # rewards
        self.SENSELESS_ACTION = -20
        self.CORRECT_STEP = 10
        self.DISCART_ORDER = -2
        self.INCORRECT_STEP = -10
        self.FINISHED_ORDER = 20
        
        self.ACTION_SPACE_SIZE = 3 + 4 + 2 + 1
        self.TOTAL_NUM_ORDERS = 9
        
        # there are 3 rings, so 4 repeats
        self.ORDER_NORM_FACTOR = [3, 4, 4, 4, 2, 1, 1, 1020]
        
        self.normalize = normalize
    
    def get_observation(self):
        # compute all the expectation values
        
        # self.order_stages:
        # "BS" "R1" "R2" "R3" "CS" "DS"
        
        # expected time and reward
        E_rewards = []
        for idx, order in enumerate(self.orders):
            if order[0] == 0:
                E_rewards.append(0)
                continue
            
            ### TIME
            # TODO: depending on state of product
            self.robots[0].distance(self.machines["BS"])
            
            
            ### REWARD
            # no reward for getting a base
            E_reward = 0
            
            # depending on cap color => CC, for all 3 rings:
            for i in range(3):
                if order[1 + i] == self.ring_additional_bases[0]: # 2 bases
                    E_reward += 20
                    # additional points for base feeded into RS
                    E_reward += 4
                elif order[1 + i] == self.ring_additional_bases[1]: # 1 bases
                    E_reward += 10
                    # additional points for base feeded into RS
                    E_reward += 2
                elif order[1 + i] != 0: # 0 bases
                    E_reward += 5
            
            # depending on number of rings => C
            num_rings = 3 - order[1:4].count(0)
            if num_rings == 1:
                E_reward += 10
            elif num_rings == 2:
                E_reward += 30
            elif num_rings == 3:
                E_reward += 80
                
            # buffering and mounting the cap
            E_reward += 2 + 10
            
            # comsidering delivery window
            # TODO: need the expected time for it
            
            E_rewards.append(E_reward)
        
        
        
        
        # normalize output
        if self.normalize:
            orders_norm = []
            for order in self.orders:
                order_norm = []
                for idx, part in enumerate(order):
                    order_norm.append(part / self.ORDER_NORM_FACTOR[idx])
                orders_norm.append(order_norm)
            
            pip_norm = []
            for idx, part in enumerate(self.pipeline):
                pip_norm.append(part / self.ORDER_NORM_FACTOR[idx])
                
            observation = sum(orders_norm, []) + pip_norm
        else:            
            observation = sum(self.orders, []) + self.pipeline
        
        return observation
    
    def render(self):
        print("orders: {} | pipeline: {}{}".format(self.orders, self.pipeline, self.pipeline_cap))
    
    # reset all parameters to an initial game state
    def reset(self):
        np.random.seed()
        # utility parameters
        self.episode_step = 0


        # current roboter position
        self.robots = [field_pos(4.5, 0.5), field_pos(5.5, 0.5), field_pos(6.5, 0.5)]


        # field generation (currently no respect on blocking and rotation)
        # can be placed on full 7x8 grid with exception of 51 61 71 52
        self.machines = {"CS1": None, "CS2": None, "RS1": None, "RS2": None, "SS": None, "BS": None, "DS": None}
        for machine in self.machines:
            # filter impossible and overlapping positions
            while True:
                x_pos = int(np.random.uniform(0, 7)) + 0.5
                y_pos = int(np.random.uniform(0, 8)) + 0.5
                pos = field_pos(x_pos, y_pos)
                
                if pos not in self.robots and pos != (4.5, 1.5) and pos not in self.machines.values():
                    break
            self.machines[machine] = pos
        
        # swap/flip ONE random CS and ONE random RS to the other side
        if int(np.random.uniform(0, 2)):
            self.machines["RS1"].x *= -1
        else:
            self.machines["RS2"].y *= -1
        if int(np.random.uniform(0, 2)):
            self.machines["CS1"].x *= -1
        else:
            self.machines["CS2"].y *= -1


        # assign the rings to RS1 and RS2 respectively, each getting one complicated
        self.rings = [[0, 0], [0, 0]]
        rnd = int(np.random.uniform(0, 2))
        self.rings[0][0] = self.ring_additional_bases[rnd]
        self.rings[1][0] = self.ring_additional_bases[1 - rnd]
        rnd = int(np.random.uniform(0, 2))
        self.rings[0][1] = self.ring_additional_bases[2 + rnd]
        self.rings[1][1] = self.ring_additional_bases[2 + 1 - rnd]
        
        
        # current time
        self.time = 0


        # defining additional ring bases
        # first needs 2 bases, 2nd one, 3rd and 4th zero
        self.ring_additional_bases = [x for x in range(1, 5)]
        np.random.shuffle(self.ring_additional_bases)


        # RefBox like behavoir (=distribution); creating full matrix
        self.orders = [[0] * 9] * self.TOTAL_NUM_ORDERS
        # id1 is C0 full window
        self.orders[0] = create_order(C=0, fill=True)
        self.orders[0][-1] = 1021
        self.orders[0][-2] = 1
        # id2 is C1 with a ring requiring additional base, full window
        self.orders[1] = create_order(C=0, fill=True)
        self.orders[1][1] = self.ring_additional_bases[int(np.random.uniform(0, 2))]
        self.orders[1][-1] = 1021
        self.orders[1][-2] = 1
        # id4 is C3, window around 600+
        self.orders[3] = create_order(C=3, fill=True, window=600)
        
        # track what assembly step we are at
        self.order_stage = ["BS"] * self.TOTAL_NUM_ORDERS


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
        
        # TODO: test without pipeline converge?
        # TODO: random delay model, best a function
        # applying the base
        if self.order_stage == 0 and action_type == 1:
            # updating tracking
            self.pipeline[0] = action_color
            
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
            
            
        # applying the ring
        elif self.order_stage in [1, 2] and action_type == 2:
            # find next free (=0) ring slot
            if 0 in self.pipeline[1:4]:
                free_ring = self.pipeline.index(0) # TODO: Need to track manuaally at which ring
                self.pipeline[free_ring] = action_color
            else:
                # prelimary end if we have too many rings
                reward = self.INCORRECT_STEP
                done = True
                return self.get_observation(), reward, done
            
            if self.order_stage == 1:
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
            elif self.order_stage == 2:
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
            
            
        # applying the cap
        elif self.order_stage == 2 and action_type == 3:
            self.pipeline_cap = action_color
            
            
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
        
            reward = self.SENSELESS_ACTION
            done = True
            
        
        # we stop if we try for too long
        if self.episode_step >= 11:
            done = True
        
        return self.get_observation(), reward, done


if __name__ == "__main__":
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("Please import the file")