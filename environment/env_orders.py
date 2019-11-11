#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import time
import struct
import traceback
import datetime as dt
import numpy as np
import math
from enum import Enum
from copy import deepcopy # more performant then dict()
import matplotlib.pyplot as plt

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
        self.TOTAL_NUM_ORDERS = 9 # warning: currently does not scale all
        
        # there are 3 rings, so 4 repeats
        self.ORDER_NORM_FACTOR = [3, 4, 4, 4, 2, 1, 1, 1021, 1021]
        
        # order of steps needed to fulfill an oder
        self.processing_order = ["BS", "R1", "R2", "R3", "CS", "DS", "FIN"]
        
        # additional time factor for time lost on grapping products with machine arm
        # 2017 Carologistics needs 68sec for adjusting grapping and plaing on CS
        # they need 45sec for grap, move and place products (10~30sec move and adjust)
        self.grap_and_place_delay = 20
        
        # TODO: return normalized parameters
        self.normalize = normalize
    
    def expectation_order(self, order, current_stage, current_pos, processing=None):
        E_time = 0
        E_reward = 0
        E_time_next = None
        E_reward_next = None
        
        ##### track machine path, looping over future machines
        if processing == None:
            cont = self.processing_order.index(current_stage)
            processing = self.processing_order
        else:
            cont = processing.index(current_stage)
            
        for stage in processing[cont:]:
            if stage == "FIN":
                continue
            
            distance = 0 # the traveled/movement distance 
            wait = 0 # machine processing time + arm-movement delay
            reward = 0 # reward for that step
            # for safety of reusing
            ring_col = None
            ring_pos = None
            need_bases = None
            missing_bases = None
            cap_col = None
            next_pos = None
            
            ##### correct processing_order to actual next machine
            # decide which RS
            if stage[0] == 'R':
                # find which ring
                ring_pos = int(stage[1])
                ring_col = order[ring_pos]
                ring_num = 3 - order[1:4].count(0)
                
                # check if it has ring on this slot; otherwise it is an non-existent processing step
                if ring_col == 0:
                    continue
                
                # figure out where we can get it
                if ring_col in self.rings[0]:
                    to_machine = "RS1"
                elif ring_col in self.rings[1]:
                    to_machine = "RS2"
                    
            # decide which CS
            elif stage == "CS":
                cap_col = order[4]
                if cap_col == 2:
                    to_machine = "CS1"
                elif cap_col == 1:
                    to_machine = "CS2"
                    
            # otherwise stage matches machine name
            else:
                to_machine = stage
            
            
            ##### get specific machine position and compute travling distance
            next_pos = self.machines[to_machine]
#                print("at: {}, to: {}, distance: {}".format(current_pos, next_pos, current_pos.distance(next_pos)))
            distance += current_pos.distance(next_pos)
            current_pos = next_pos # we now made the step to next machine
            
            # assume each step involves grappign and plaing a product at least once
            wait += self.grap_and_place_delay
            
            ###### additional wait time per machine; assumed after we arrive and do process there
            ###### computation of reward for this step
            if to_machine == "BS":
                wait += 5 # estimate/assumption
                
                # no reward for getting a base
                reward += 0
                
            elif to_machine == "RS1" or to_machine == "RS2":
                wait += 50 # mean official delay
                
                ### consider additional bases and reward
                # figure out if current ring need additional bases
                if ring_col == self.ring_additional_bases[0]: # 2 bases
                    need_bases = 2
                    
                    # reward for CC2
                    reward += 20
                                            
                elif ring_col == self.ring_additional_bases[1]: # 1 bases
                    need_bases = 1
                    
                    # reward for CC1
                    reward += 10
                else:
                    need_bases = 0
                    
                    # reward for CC0
                    reward += 5
                
                # check if need gather additional bases; need minus have
                missing_bases = need_bases - self.rings_buf_bases[int(to_machine[2]) - 1]
                if missing_bases >= 1:
                    # we condsider an additional back and forth to a BS from current RS *per* missing base
                    extra = current_pos.distance(self.machines["BS"]) * 2 # 2 for back-forth
                    
                    distance += extra * missing_bases
                    wait += self.grap_and_place_delay * missing_bases # assumption on lost time grapping bases
                    
                    # additional points for base feeded into RS
                    reward += 2 * missing_bases
                
                # for final ring, depending on number of rings
                if ring_num == ring_pos:
                    if ring_num == 1:
                        # reward for C1
                        reward += 10
                    elif ring_num == 2:
                        # reward for C2
                        reward += 30
                    elif ring_num == 3:
                        # reward for C3
                        reward += 80
                        
            elif to_machine == "CS1" or to_machine == "CS2":
                # additional time to buffer
                wait += 20 # for buffer cap first
                wait += 3 * 2 # traveling around the machine
                
                # dispose to nearest RS for now
                # TODO: optimize use; depending wheter we still can reuse those
                distance_rs1 = current_pos.distance(self.machines["RS1"])
                distance_rs2 = current_pos.distance(self.machines["RS2"])
                wait += min(distance_rs1, distance_rs2) * 2 # 2 for back-forth
                wait += self.grap_and_place_delay # assumption on lost time grapping clear bases
                
                # reward for buffering a cap
                reward += 2
                
                # mount cap
                wait += 20
                # reward fo mount cap
                reward += 10
                
            elif to_machine == "DS":
                wait += 30
                
                # comsidering delivery window; accounting for next E_time update
                E_delivery = self.time + E_time + distance * 2 + wait
                if E_delivery < order[-2]:
                    reward += 1 # wrong delivery
                elif E_delivery < order[-1]:
                    reward += 20 # (correct) delivery
                elif E_delivery < order[-1] + 10:
                    tmp = 15 - (E_delivery - order[-1]) * 1.5 + 5
                    assert tmp >= 5 and tmp <= 20
                    reward += tmp # delayed delivery
                else:
                    reward += 5 # late delivery
                
            elif to_machine == "SS":
                wait += 10 # estimate
                
                reward -= 10 # listed cost
            
            # accumulate time; assume 1m per 2s
            E_time += distance * 2 + wait
            
            # accumulate reward, as long game not over (expected)
            # TODO: scale or consider variance for more fluent drop to 0 points (as we "might" make it in time)
            if E_time + self.time <= 1021:
                E_reward += reward
            
            # save the step to next machine
            if E_time_next == None:
                E_time_next = E_time
            if E_reward_next == None:
                E_reward_next = E_reward
            
#            print("E_time: {} | E_time_next: {} | E_reward: {} | E_reward_next: {}".format(E_time, E_time_next, E_reward, E_reward_next))
        
        return E_time, E_time_next, E_reward, E_reward_next
    
    def get_observation(self):
        
        # expected time and reward
        options = len(self.orders)
        E_rewards = [None] * options # accumulated for full order
        E_times = [None] * options # accumulated for full order
        E_rewards_next = [None] * options # next step
        E_times_next = [None] * options # next step
        E_multi_order = [0, 0, 0, 0] # additional parameters for the one order of two products
        
        for idx, order in enumerate(self.orders):
            # have no order here yet
            if order[0] == 0:
                E_rewards[idx] = 0
                E_times[idx] = 0
                E_rewards_next[idx] = 0
                E_times_next[idx] = 0
                continue
            
            # account for partial processed products => in step() (update self.order_stage for all)
            # we start in the order processing pipeline from the step it currently is in
            current_stage = self.order_stage[idx]
            # we can have list in case of 2 requested products; work with first initially
            if type(current_stage) == list:
                current_stage2 = current_stage[1]
                current_stage = current_stage[0]
            
            # TODO: consider multiple robots (need outside self-loop with robot selection). search closest free robot?
            # robots are reasonably fast that pathing, thus which robot, is a more minor problem
            current_pos = self.robots[0]
            
            # TODO: consider switching insode the expectations
            # TODO: Expectation of when already have fitting partial product => consider in step
            E_time, E_time_next, E_reward, E_reward_next = self.expectation_order(order, current_stage, current_pos)
            
            # consider competitive orders, when we deliver on time; apply sigmoid-like scaling for delivery window
            E_delivery = self.time + E_time
            if order[6] == 1 and E_delivery >= order[-2] and E_delivery <= order[-1]:
                # compute bonus points for competitive
                ratio_scaling = 3.53 # selected as a constant so that we get 3/4 of points at 1/4 window
                tmp = 1 + math.exp(ratio_scaling)
                width_scaling = (-20 * tmp) / (2 - tmp) # so that we have about +-10 on each side; exact=20.82640446380697375952
                length_scaling = (order[-1] - order[-2]) / 2
                
                at_time = E_delivery - order[-2] - length_scaling # we make sure we scale inside the window
                comp_bonus = width_scaling / (1 + math.exp((ratio_scaling * at_time) / length_scaling)) - width_scaling/2
                
                # add to the reward
                E_reward += comp_bonus
            
            
            # consider if order need mupltiple products
            # we will definitely need to build one order normally; other can be normal or from SS
            if order[5] == 1:
                # case 1) we make the whole process again starting from last machine
                E_time2, E_time_next2, E_reward2, E_reward_next2 = self.expectation_order(order, current_stage2, self.machines["DS"])
                # case 2) take one from the SS; consider it as extra sub-order
                E_time3, E_time_next3, E_reward3, E_reward_next3 = self.expectation_order(order, "SS", self.machines["DS"], ["SS", "DS", "FIN"])
                
                # assemble the extra vector before updating E's; delivery window already present in other
                E_multi_order = [E_reward + E_reward3, 
                                 E_reward_next,
                                 E_time + E_time3,
                                 E_time_next]
                
                E_time += E_time2
                E_reward += E_reward2
                # when the first product is finished we have intermediate second
                if current_stage == "FIN":
                    E_reward_next = E_reward_next2
                    E_time_next = E_time_next2
                    E_multi_order[1] = E_reward_next3
                    E_multi_order[3] = E_time_next3
            
            
            ##### for each order we add to vector
            E_times[idx] = E_time
            E_rewards[idx] = E_reward
            E_times_next[idx] = E_time_next
            E_rewards_next[idx] = E_reward_next
        
        
        # formulate all in a matrix
        # TODO: also consider variance?
        observation_ = np.array([E_rewards] + [E_rewards_next] + [E_times] + [E_times_next]).T
        del_windows = np.array([o[-2:] for o in self.orders])
        observation = np.concatenate((observation_, del_windows), axis=1)
        # rest parameters; handling double order and time
        remainder = E_multi_order +  [1021 - self.time]
        
        
        # normalize output
#        if self.normalize:
#            orders_norm = []
#            for order in self.orders:
#                order_norm = []
#                for idx, part in enumerate(order):
#                    order_norm.append(part / self.ORDER_NORM_FACTOR[idx])
#                orders_norm.append(order_norm)
#            
#            pip_norm = []
#            for idx, part in enumerate(self.pipeline):
#                pip_norm.append(part / self.ORDER_NORM_FACTOR[idx])
#                
#            observation = sum(orders_norm, []) + pip_norm
#        else:            
#            observation = sum(self.orders, []) + self.pipeline
        
        return observation, remainder
    
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


        # defining additional ring bases
        # first needs 2 bases, 2nd one, 3rd and 4th zero
        self.ring_additional_bases = [x for x in range(1, 5)]
        np.random.shuffle(self.ring_additional_bases)


        # assign the rings to RS1 and RS2 respectively, each getting one complicated
        self.rings = [[0, 0], [0, 0]]
        rnd = int(np.random.uniform(0, 2))
        self.rings[0][0] = self.ring_additional_bases[rnd]
        self.rings[1][0] = self.ring_additional_bases[1 - rnd]
        rnd = int(np.random.uniform(0, 2))
        self.rings[0][1] = self.ring_additional_bases[2 + rnd]
        self.rings[1][1] = self.ring_additional_bases[2 + 1 - rnd]
        # track how many bases a ring station has buffered
        self.rings_buf_bases = [0, 0]
        
        
        # current time
        self.time = 1 # as delivery windows are offset by 1 sec we start at 1sec



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
            
        # TODO: make products appear randomly on time for specified order slots; account quantity & characteristics
        
        ##### UPDATING PRODUCT
        # format is a  two digit integer, first the category, second the color
        assert action >= 0 and action <= self.TOTAL_NUM_ORDERS
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
    print("Please import the file.")
#    assert False
    
    
    #### for debug scenario
    self = env_rcll()
    self.reset()
    self.orders = [[1, 0, 0, 0, 2, 0, 0, 1, 1021], # @ 0006
                   [2, 3, 0, 0, 2, 0, 0, 1, 1021], # @ 0006
                   [1, 4, 1, 0, 2, 0, 0, 848, 1009], # @ 0239
                   [1, 1, 3, 4, 2, 0, 0, 678, 834], # @ 0006
                   [2, 0, 0, 0, 1, 0, 0, 421, 572], # @ 0268
                   [3, 0, 0, 0, 2, 1, 0, 640, 748], # @ 0403
                   [2, 0, 0, 0, 2, 0, 1, 841, 1021], # @ 0661
                   [3, 2, 0, 0, 2, 0, 0, 710, 817]] # @ 0209
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    self.order_stage[5] = [self.order_stage[5], "BS"]
#    self.orders = [[1, 0, 0, 0, 2, 0, 0, 1, 1021], # @ 0006
#                   [2, 3, 0, 0, 2, 0, 0, 1, 1021], # @ 0006
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [1, 1, 3, 4, 2, 0, 0, 678, 834], # @ 0006
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [3, 2, 0, 0, 2, 0, 0, 710, 817], # @ 0209
#                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    self.machines = {'CS1': field_pos(-3.5, 4.5),
                     'CS2': field_pos(2.5, 0.5),
                     'RS1': field_pos(-1.5, 2.5),
                     'RS2': field_pos(1.5, 7.5),
                     'SS': field_pos(3.5, 1.5),
                     'BS': field_pos(4.5, 7.5),
                     'DS': field_pos(2.5, 4.5)}
    self.ring_additional_bases = [3, 1, 2, 4]
    self.rings = [[3, 4], [1, 2]]
    
    assert False
    
    
    # testing code
    
    # deactivate numpy scientific notation printing..
#    np.set_printoptions(suppress=True)
#    
#    
#    obs = get_observation(self)
#    get_observation(self)[0][:, :2].tolist() + [get_observation(self)[1][:2]]
#    get_observation(self)[0] + [get_observation(self)[1]]

    
    observations = []
    for i in range(1, 1022):
        self.time = i
        
        observations.append(get_observation(self))
        
    t_rewards = []
    for obs in observations:
        t_rewards.append(obs[0][:, 0].tolist() + [obs[1][0]])
    t_rewards = np.array(t_rewards).T.tolist()
    
    
    t = range(1,1022)
    labels = [r'$O_{}$'.format(x) for x in range(1,10)]
    
    plt.figure(figsize=(15,7))
    for y, l in zip(t_rewards, labels):
        plt.plot(t, y, label=l)
    
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend(loc='best')
    plt.show()
    
    
    
    
    
