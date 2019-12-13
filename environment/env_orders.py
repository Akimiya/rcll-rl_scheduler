#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import time
import datetime as dt
import numpy as np
import math
import subprocess
from random import SystemRandom
import matplotlib.pyplot as plt

# TODO: need to make refbox_comm into a class or uglily import it to get running globals

class field_pos():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rotation = None
        self.obj_type = ""
    
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

# Class copying numerical/statistical functions & propperties of the RefBox
# we create refbox-like behavoir by taking their parameters and functions
""" NEEDS TO BE UPDATED ON CHANGE (in the *.clp files and <mps_placing_clips> folder) """
class RefBox_recreated():
    # from the (deffacts orders) from facts.clp, including default order values
    # defined as [complexity, number, competitive, start_range, activation_range, duration_range]; range->touple
    ORDER_PARAMETERS = [
            [0, 1, 0, (0, 0), (1020, 1020), (1020, 1020)], # 1
            [1, 1, 0, (0, 0), (1020, 1020), (1020, 1020)], # 2
            [2, 1, 0, (650, 850), (500, 900), (100, 200)], # 3
            [3, 1, 0, (650, 850), (1020, 1020), (150, 200)], # 4
            [0, 1, 0, (200, 450), (120, 240), (60, 180)], # 5
            [0, 2, 0, (350, 800), (120, 240), (60, 180)], # 6
            [0, 1, 0, (800, 1020), (120, 240), (60, 180)], # 7
            [1, 1, 0, (550, 800), (350, 550), (100, 200)], # 8
            [0, 1, 1, (1020, 1020), (0, 0), (300, 300)] # 9 for overtime
            ]
    
    # from facts.clp inside "deffacts startup" the base definitions; order=> {cyan: magenta}
    mirror_orientation_def = {0: 180,
                                  45: 135,
                                  90: 90,
                                 135: 45,
                                 180: 0,
                                 225: 315,
                                 270: 270,
                                 315: 225}

    # from globals.clp
    DOWN_TYPES = ["RS", "CS"]
    DOWN_TIME_MIN = 30 #  30
    DOWN_TIME_MAX = 60 # 120
    PRODUCTION_TIME = 1020
    
    def __init__(self):
        # just in case we want same seed, we need take randomness like the other module
        self.random = SystemRandom()
        
        # debug for tracking number of errors
        self.solver_errors = 0
        self.solver_runs = 0
        self.solver_time = 0
        
    
    def randomize(self, my_list):
        # custom used randomize runction from RefBox
        l = len(my_list)
        for _ in range(200):
            a = self.random.randrange(l) # we start from 0 to len, excluding last
            b = self.random.randrange(l)
            tmp = my_list[a]
            my_list[a] = my_list[b]
            my_list[b] = tmp
        return my_list

    
    # running code from mps_placing_clips.cpp
    # CLIPS calls this function binding C function to it
    # function/class sets constraints and then passes them to the GECODE constraint solver
    def mps_generator_get_generated_field(self):
        self.solver_runs += 1 # local debug
        start = time.time()
        
        field = None
        while field == None:
            try:
                field = subprocess.check_output("./field_gen/main")
            except subprocess.CalledProcessError as e:
                print("Solver returned {}".format(e.returncode))
                self.solver_errors += 1
            
        # TODO: make sure that we actually get a different field, as it is similar sometimes for some reason
        self.solver_time += time.time() - start
        print("field generation took", time.time() - start)
        
        return field


    # function from utils.clp; note: all coordinates are -0.5 in the environment
    def mirror_orientation_function(self, mtype, pos):
        t = pos.obj_type[0]
        
        # TODO: consider ML cases where we learn for both firled rotations (do they make a difference?)
        # function from utils.clp summarized in one expression
        want_mirrored_rotation = mtype in ["BS", "DS", "SS"] or \
        (not (pos.x == 6.5 or pos.y == 7.5 or pos.y == 0.5 or (pos.x in [4.5, 5.5, 6.5] and pos.y == 1.5)))
        
        if want_mirrored_rotation:
            if t == "C":
                m_ori = self.mirror_orientation_def[pos.rotation]
            else:
                # inverse lookup of {magenta: cyan}
                m_ori = next(c for c, m in self.mirror_orientation_def.items() if m == pos.rotation)
                
            return m_ori
        
        else:
            x = pos.x
            y = pos.y
            
            if y == 7.5:
                return 180
            if y == 0.5 or y == 1.5:
                return 0
            if x == 6.5 and t == "M":
                return 90
            if x == 6.5 and t == "C":
                return 270
            
            return pos.rotation


    # translated from machines.clp
    def machine_init_randomize(self, ring_colors):
        # resets machines, first set all parameters to 0, all lights on and state to IDLE
        # this creates 
        # read field from "/llsfrb/game/random-field" if available
        # Randomizing from scratch if some machine is still zone TBD or overwrite-generating flag set
        machines_cyan = {"CS1": None, "CS2": None, "RS1": None, "RS2": None, "SS": None, "BS": None, "DS": None}
        machines_magenta = {"CS1": None, "CS2": None, "RS1": None, "RS2": None, "SS": None, "BS": None, "DS": None}
        # defines some unused ?zones-magenta
        
        # generate new full machine field; all code before this is mostly irrellevant
        # (bind ?machines (mps-generator-get-generated-field))
        field = self.mps_generator_get_generated_field()
        
        # process the output generated by field_gen
        for m_desc in field.decode("utf-8").split('\n')[:-1]:
            machine, zone, rotation = m_desc.split(", ")
            
            # preprocess; note: we use -0.5 positions in the envronment
            # like in files generation is assumed as for team MAGENTA
            machine = machine[2:] # strip "M-"
            x_pos = - (int(zone[-2]) - 0.5) # minus for position MAGENTA
            y_pos = int(zone[-1]) - 0.5
            rotation = int(rotation)
            
            pos = field_pos(x_pos, y_pos)
            pos.rotation = rotation
            pos.obj_type = "M_" + machine
            
            machines_magenta[machine] = pos
        
        
        # Mirror machine position and rotation for other team
        for mtype, pos in machines_magenta.items():
            # for the actual zone name only the prefix changes (e.g. from "M_" to "C_"); for us its the sign
            x = - pos.x
            y = pos.y
            machines_cyan[mtype] = field_pos(x, y)
            # for consistency..
            machines_cyan[mtype].obj_type = "C" + pos.obj_type[1:]
            
            machines_cyan[mtype].rotation = self.mirror_orientation_function(mtype, pos)
            
        
        # Swap machines
        machines_to_swap = ["RS" + str(self.random.randint(1, 2)), 
                            "CS" + str(self.random.randint(1, 2))]
        # we just literally rewrite, even though it is just two python lines..
        for ms in machines_to_swap:
            m_cyan = machines_cyan[ms]
            m_magenta = machines_magenta[ms]
            
            zr_cyan = m_cyan
            zr_magenta = m_magenta
            
            machines_cyan[ms] = zr_magenta
            machines_magenta[ms] = zr_cyan
        
        
        # assign random down times
        # assume ?*RANDOMIZE-GAME* = TRUE
        candidates = []
        for t in self.DOWN_TYPES:
            t_candidates = [m for m in machines_cyan if m[:2] == t]
            t_candidates = self.randomize(t_candidates) # randomize$
            candidates.append(t_candidates[0]) # first$ and append$
        
        down_period = []
        for c in candidates:
            duration = self.random.randint(self.DOWN_TIME_MIN, self.DOWN_TIME_MAX)
            start_time = self.random.randint(1, self.PRODUCTION_TIME - duration)
            end_time = start_time + duration
            
            # Copy to magenta machine; we don't worry about it internally
            down_period.append([c, start_time, end_time])
        
        
        # Randomize ring colors per machine
        # select RS1 machins for each team and assign rings
        rs1 = ring_colors[:2]
        # select RS2 machins for each team and assign rings
        rs2 = ring_colors[2:]
        rings = [rs1, rs2]
            
        # we currently only consider CYAN team in environment
        return machines_cyan, down_period, rings
            
    
    # from game.clp
    def game_parametrize(self):
        # (bind ?ring-colors (randomize$ ?ring-colors))
        ring_colors = self.randomize(list(range(1, 5)))
        c1_first_ring = ring_colors[0]
        c2_first_ring = ring_colors[1]
        c3_first_ring = ring_colors[2]
        cx_first_ring = ring_colors[3]
        
        c1_counter = 0
        c2_counter = 0
        c3_counter = 0
        
        machines, down_period, rings = self.machine_init_randomize(ring_colors)
        
        # reset orders, assign random times
        order_declarations = []
        orders_full = []
        
        # [complexity, number, proba_competitive, start_range, activation_range, duration_range]
        for oid, (complexity, number, competitive, start_range, activation_range, duration_range) in enumerate(self.ORDER_PARAMETERS):
            oid += 1 # correction start from 1
            
            # selecting delivery window
            deliver_start = self.random.randint(start_range[0], start_range[1])
            deliver_end = deliver_start + self.random.randint(duration_range[0], duration_range[1])
            # correct delivery winow to before game end
            if deliver_end > self.PRODUCTION_TIME and oid != 9: # 9th order is for overtime
                deliver_start -= deliver_end - self.PRODUCTION_TIME
                deliver_end = self.PRODUCTION_TIME
            
            # time order is announced
            activation_pre_time = self.random.randint(activation_range[0], activation_range[1])
            activate_at = max(deliver_start - activation_pre_time, 0)
            
            order_declarations.append([activate_at, oid, complexity, number, competitive, (deliver_start, deliver_end)])
        
            # assumptions for the 2016 game and order to workpiece assignment!
            order_ring_colors = []
            if complexity == 0:
                pass # for C0 we have nothing to do, no ring color
                
            elif complexity == 1:
                c1_counter += 1
                if c1_counter <= 1:
                    first_ring = c1_first_ring
                else:
                    first_ring = cx_first_ring
                order_ring_colors = [first_ring]
                
            elif complexity == 2:
                c2_counter += 1
                if c2_counter <= 1:
                    first_ring = c2_first_ring
                else:
                    first_ring = cx_first_ring
                tmp = [r for r in ring_colors if r != first_ring] # the remove$ step
                tmp = self.randomize(tmp) # the randomize$ step
                order_ring_colors = [first_ring] + tmp[:1]
            
            elif complexity == 3:
                c3_counter += 1
                if c3_counter <= 1:
                    first_ring = c3_first_ring
                else:
                    first_ring = cx_first_ring
                tmp = [r for r in ring_colors if r != first_ring] # the remove$ step
                tmp = self.randomize(tmp) # the randomize$ step
                order_ring_colors = [first_ring] + tmp[:2]
            
            # they just take a random of the allowed values
            order_base_color = self.random.randint(1, 3) # 1 out of 3
            order_cap_color = self.random.randint(1, 2) # 1 out of 2
            
            orders_full.append([order_base_color] + order_ring_colors + [0] * (3 - complexity) + 
                          [order_cap_color] + [deliver_start, deliver_end])
        
        # Randomize number of required additional bases
        m_add_bases = self.randomize([1, 3])
        # (do-for-fact ((?ring ring-spec)) (eq ?ring:color (nth$ (nth$ 1 ?m-add-bases) ?ring-colors))
        req_bases_2 = ring_colors[m_add_bases[0] - 1] # correction for indexing
        # (do-for-fact ((?ring ring-spec)) (eq ?ring:color (nth$ (nth$ 2 ?m-add-bases) ?ring-colors))
        req_bases_1 = ring_colors[m_add_bases[1] - 1] # correction for indexing
        # (or (eq ?ring:color (nth$ 2 ?ring-colors)) (eq ?ring:color (nth$ 4 ?ring-colors)))
        # here we just assemble result as 2th and 4th element set to 0
        ring_additional_bases = [req_bases_2, req_bases_1, ring_colors[1], ring_colors[3]]
        
        
        # Randomly assign an order to be a competitive order
        potential_competitive_orders = []
        for _, oid, complexity, number, _, (deliver_start, deliver_end) in order_declarations:
            if complexity == 0 and number == 1 and oid != 9 and (deliver_start != 0 or deliver_end != self.PRODUCTION_TIME):
                potential_competitive_orders.append(oid)
        # (bind ?competitive-order-id (nth$ (random 1 (length$ ?potential-competitive-orders)) ?potential-competitive-orders))
        competitive_order_id = self.random.randrange(len(potential_competitive_orders))
        modify = potential_competitive_orders[competitive_order_id] - 1
        order_declarations[modify][4] = 1 # set competitive
        
        return machines, down_period, rings, ring_additional_bases, order_declarations, orders_full

class env_rcll():
    """
    Main environment class handlin simulation
    For perfect recreation of live behavoir pass a "RefBox_recreated" object,
    alternatively some more general randomization is used (much bigger state space!).
    """
    
    def __init__(self, RefBox_recreated=None, normalize=False):
        self.random = SystemRandom() # local random number generator
        self.TOTAL_NUM_ORDERS = 8 # warning: currently does not scale everything
        self.ACTION_SPACE_SIZE = self.TOTAL_NUM_ORDERS + 1 # plus one slot for double amount order
        
        # there are 3 rings, so 4 repeats
        self.ORDER_NORM_FACTOR = [] # old: [3, 4, 4, 4, 2, 1, 1, 1020, 1020]
        
        # TODO: return normalized parameters
        self.normalize = normalize
        
        # order of steps needed to fulfill an oder; R is ring 1 to 3, FIN is completed
        self.processing_order = ["BS", "R1", "R2", "R3", "CS", "DS", "FIN"]
        
        
        # if passed use game code as similar to the real thing as possible
        # make sure class is up-to-date with actual RefBox!
        self.RefBox = RefBox_recreated
        
        
        # rewards
        self.SENSELESS_ACTION = -20
        self.CORRECT_STEP = 10
        self.DISCART_ORDER = -2
        self.INCORRECT_STEP = -10
        self.FINISHED_ORDER = 20
        
        ############################ PROBABILITIES ############################
        # distribution for movement
        self.move_mean = 2
        self.move_var = 0.4
        
        # additional time factor for time lost on grapping products with machine arm
        # 2017 Carologistics needs 68sec for adjusting grapping and plaing on CS
        # they need 45sec for grap, move and place products (10~30sec move and adjust)
        # Rayleigh (or Chi) distribution may also be an option
        self.grap_and_place_mean = 30
        self.grap_and_place_var = 7
        
        # each machine has its own processing time; some are improvised gaussian format: [mean, var]
        self.machine_times = {
                "BS": [5, 0.15], # estimate/assumption
                "RS": [40, 60], # official low and high
                "CS": [15, 25], # official low and high
                "DS": [20, 40], # official low and high
                "SS": [5, 0.15] # estimate/assumption
                }
        
    def generate_game_setting(self):
        """
        Fully random function positioning the machines on the field.
        Main result is final definition of "self.machines".
        Only used without defined RefBox.
        """
        
        # field generation (no respect for blocking and rotation)
        # can be placed on full 7x8 grid with exception of 51 61 71 52
        self.machines = {"CS1": None, "CS2": None, "RS1": None, "RS2": None, "SS": None, "BS": None, "DS": None}
        for machine in self.machines:
            # filter impossible and overlapping positions
            while True:
                x_pos = self.random.randint(0, 6) + 0.5
                y_pos = self.random.randint(0, 7) + 0.5
                pos = field_pos(x_pos, y_pos)
                
                # to not have overlaps
                if pos not in self.robots and pos != (4.5, 1.5) and pos not in self.machines.values():
                    break
            self.machines[machine] = pos
        
        # swap/flip ONE random CS and ONE random RS to the other side
        if self.random.randint(0, 1):
            self.machines["RS1"].x *= -1
        else:
            self.machines["RS2"].y *= -1
        if self.random.randint(0, 1):
            self.machines["CS1"].x *= -1
        else:
            self.machines["CS2"].y *= -1
        

        # defining additional ring bases
        np.random.shuffle(self.ring_additional_bases)

        # assigning which RS gets which colors
        rnd = int(np.random.uniform(0, 2))
        self.rings[0][0] = self.ring_additional_bases[rnd]
        self.rings[1][0] = self.ring_additional_bases[1 - rnd]
        rnd = int(np.random.uniform(0, 2))
        self.rings[0][1] = self.ring_additional_bases[2 + rnd]
        self.rings[1][1] = self.ring_additional_bases[2 + 1 - rnd]
        
        # using similar to RefBox parameters & computation for activation times
        p = []
        self.order_declarations = []
        for oid, (complexity, number, competitive, start_range, activation_range, duration_range) in enumerate(RefBox_recreated.ORDER_PARAMETERS):
            oid += 1 # correction start from 1
            
            # selecting delivery window
            deliver_start = self.random.randint(start_range[0], start_range[1])
            deliver_end = deliver_start + self.random.randint(duration_range[0], duration_range[1])
            # correct delivery winow to before game end
            if deliver_end > RefBox_recreated.PRODUCTION_TIME and oid != 9: # 9th order is for overtime
                deliver_start -= deliver_end - RefBox_recreated.PRODUCTION_TIME
                deliver_end = RefBox_recreated.PRODUCTION_TIME
            
            # time order is announced
            activation_pre_time = self.random.randint(activation_range[0], activation_range[1])
            activate_at = max(deliver_start - activation_pre_time, 0)
            
            # all (non-1) proba_competitive sum up to 1
            if competitive == 1:
                p.append(0)
            else:
                if complexity == 0 and number == 1 and oid != 9 and (deliver_start != 0 or deliver_end != RefBox_recreated.PRODUCTION_TIME):
                    p.append(1)
                else:
                    p.append(0)
            
            self.order_declarations.append([activate_at, oid, complexity, number, competitive, (deliver_start, deliver_end)])
        
        # figure out if competitive; currently only one order is
        p = [x / sum(p) for x in p] # normazize to sum up to 1
        rnd = np.random.choice(np.arange(len(self.order_declarations)), p=p)
        self.order_declarations[rnd][-2] = 1
        
    
    def create_order(self, C=-1, fill=False, amount=False, compet=False, window=-1):
        # enum bases red = 1, black, silver
        base = self.random.randint(1, 3)
        
        # number of rings
        if C == -1:
            rnd = self.random.random()
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
        self.random.shuffle(ring_options)        
        rings = ring_options[:num_rings] + [0] * (3 - num_rings)
        
        # enum caps black = 1, grey
        cap = self.random.randint(1, 2)
        
        # number of requested products
        if amount == True:
            num_products = [self.random.randint(1, 2)]
        else:
            num_products = [0] if fill else []
        
        # if order is competitive
        if compet == True:
            competitive = [self.random.randint(1, 2)]
        else:
            competitive = [0] if fill else []
            
        # the delivery window
        minimum_window = 120
        if type(window) == tuple:
            delivery_window = [window[0], window[1]]
        else:
            if window >= 0:
                start = self.random.randint(window, 1020 - minimum_window)
                end = self.random.randint(start + minimum_window, 1020)
                delivery_window = [start, end]
            else:
                delivery_window = [0, 0] if fill else []
        
        return [base] + rings + [cap] + num_products + competitive + delivery_window
    
    
    def update_orders(self):
        """
        Take declarations for this game and update self.orders accordingly .
        The declerations are sorted by activation time.
        """
        
        for activate, oid, C, n, compet, delivery in self.order_declarations:
            # checking activation time
            if activate <= self.time:
                if self.orders[oid - 1][0] != 0:
                    continue # already defined
                
                # need always be equal making sure this function is called exactly on time for order
                assert activate == self.time
                assert n >= 1 and n <= 2
                
                if self.orders_full:
                    self.orders[oid - 1] = self.orders_full[oid - 1]
                else: 
                    self.orders[oid - 1] = self.create_order(C=C, fill=True, amount=n - 1, compet=compet, window=delivery)
                
            else:
                # tracking when this function needs to be called next
                self.orders_next_activation = activate
                break
    
    def stage_to_machine(self, stage, order):
        """ correct processing_order encoding to actual next machine """
        
        ring_col = None
        ring_pos = None
        ring_num = None
        to_machine = None
        
        # decide which RS
        if stage[0] == 'R':
            # deduct ring parameters
            ring_pos = int(stage[1])
            ring_col = order[ring_pos]
            ring_num = 3 - order[1:4].count(0)
            
            # figure out where we can get it
            if ring_col in self.rings[0]:
                to_machine = "RS1"
            elif ring_col in self.rings[1]:
                to_machine = "RS2"
            elif ring_col == 0:
                return to_machine, ring_pos, ring_col, ring_num
            else:
                assert False and "No such ring color!"
                
        # decide which CS
        elif stage == "CS":
            cap_col = order[4]
            if cap_col == 2:
                to_machine = "CS1"
            elif cap_col == 1:
                to_machine = "CS2"
            else:
                assert False and "No such cap color!"
                
        # otherwise stage matches machine name
        else:
            to_machine = stage
        
        return to_machine, ring_pos, ring_col, ring_num
    
    def expectation_order(self, order, current_stage, current_pos, delay=None, processing=None):
        E_time = 0 if delay == None else delay # delay will be 0 if first product is FIN
        E_reward = 0
        E_time_next = None
        E_reward_next = None
        
        ##### track machine path, looping over future machines
        if processing == None:
            processing = self.processing_order
        cont = processing.index(current_stage)
        
        for stage in processing[cont:]:
            if stage == "FIN":
                continue
            
            distance = 0 # the traveled/movement distance 
            wait = 0 # machine processing time + arm-movement delay
            reward = 0 # reward for that step
            # for safety of reusing
            need_bases = None
            missing_bases = None
            next_pos = None
            
            # correct processing_order to actual next machine
            to_machine, ring_pos, ring_col, ring_num = self.stage_to_machine(stage, order)
            # check if it has ring on this slot; otherwise it is an non-existent processing step
            if ring_col == 0:
                continue
            
            ##### get specific machine position and compute travling distance
            next_pos = self.machines[to_machine]
#                print("at: {}, to: {}, distance: {}".format(current_pos, next_pos, current_pos.distance(next_pos)))
            distance += current_pos.distance(next_pos)
            current_pos = next_pos # we now made the step to next machine
            
            # assume each step involves grappign and plaing a product at least once
            wait += self.grap_and_place_mean
            
            ###### additional wait time per machine; assumed after we arrive and do process there
            ###### computation of reward for this step
            if to_machine == "BS":
                wait += self.machine_times["BS"][0]
                
                # no reward for getting a base
                reward += 0
                
            elif to_machine == "RS1" or to_machine == "RS2":
                wait += self.machine_times["RS"][0] # mean official delay
                
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
                    wait += self.grap_and_place_mean * missing_bases # assumption on lost time grapping bases
                    
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
                wait += self.machine_times["CS"][0] # for buffer cap first
                wait += 3 * 2 # traveling around the machine
                
                # dispose to nearest RS for now
                # TODO: optimize use; depending wheter we still can reuse those
                distance_rs1 = current_pos.distance(self.machines["RS1"])
                distance_rs2 = current_pos.distance(self.machines["RS2"])
                wait += min(distance_rs1, distance_rs2) * 2 # 2 for back-forth
                wait += self.grap_and_place_mean # assumption on lost time grapping clear bases
                
                # reward for buffering a cap
                reward += 2
                
                # mount cap
                wait += self.machine_times["CS"][0]
                # reward fo mount cap
                reward += 10
                
            elif to_machine == "DS":
                wait += self.machine_times["DS"][0]
                
                # comsidering delivery window; accounting for next E_time update
                E_delivery = self.time + E_time + distance * self.move_mean + wait
                if E_delivery < order[-2]:
                    # function call without set delay for first call for order 1 out of 2
                    if order[5] == 1 and delay == None:
                        reward += 20 # assume we will deliver in time, if we start earlier
                    else:
                        reward += 1 # wrong delivery => default
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
            E_time += distance * self.move_mean + wait
            
            # accumulate reward, as long game not over (expected)
            E_reward += reward
            # TODO: scale or consider variance for more fluent drop to 0 points (as we "might" make it in time)
            if self.time + E_time > 1020:
                E_reward = 0
            
            # save the step to next machine
            if E_time_next == None:
                E_time_next = E_time
            if E_reward_next == None:
                E_reward_next = E_reward
            
#            print("E_time: {} | E_time_next: {} | E_reward: {} | E_reward_next: {}".format(E_time, E_time_next, E_reward, E_reward_next))
        
        # based on Rulebook Ch. 5.8, we get only points for stages which are not later then the delivery window
        if self.time > order[-1]:
            E_reward = 0
            E_reward_next = 0
        
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
                width_scaling = (-20 * tmp) / (2 - tmp) # so that we have about +-10 on each side
                length_scaling = (order[-1] - order[-2]) / 2
                
                at_time = E_delivery - order[-2] - length_scaling # we make sure we scale inside the window
                comp_bonus = width_scaling / (1 + math.exp((ratio_scaling * at_time) / length_scaling)) - width_scaling/2
                
                # add to the reward
                E_reward += comp_bonus
            
            
            # TODO: consider the case 1) in E_time influencing E_reward, as both need to happen in delivery window
            # TODO: case 2) also not working properly
            # consider if order need mupltiple products
            # we will definitely need to build one order normally; other can be normal or from SS
            if order[5] == 1:
                # case 1) we make the whole process again starting from last machine
                E_time2, E_time_next2, E_reward2, E_reward_next2 = self.expectation_order(order, current_stage2, self.machines["DS"], E_time)
                # case 2) take one from the SS; consider it as extra sub-order
                E_time3, E_time_next3, E_reward3, E_reward_next3 = self.expectation_order(order, "SS", self.machines["DS"], E_time, ["SS", "DS", "FIN"])
                
                # assemble the extra vector before updating E's; delivery window feature already present in other
                E_multi_order = [E_reward, 
                                 E_reward_next,
                                 E_time,
                                 E_time_next]
                
                # only consider the 2nd order when we still have time to do the mandratory one!
                # until then we consider 1st order to be delivered on time
                if self.time + E_time < order[-2]:
                    # update respective rewards as we are on time for 2nd order
                    E_multi_order[0] += E_reward3
                    E_multi_order[2] += E_time3
                    E_time += E_time2
                    E_reward += E_reward2
                    
                    # when the first product is finished we have next the intermediate of the second
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
        remainder = E_multi_order +  [1020 - self.time]
        
        
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
        # utility parameters
        self.episode_step = 0

        # current roboter position
        self.robots = [field_pos(4.5, 0.5), field_pos(5.5, 0.5), field_pos(6.5, 0.5)]
        
        # assign the rings to RS1 and RS2 respectively, each getting one complicated
        self.rings = [[0, 0], [0, 0]]
        
        # first needs two bases, 2nd one, 3rd and 4th zero
        self.ring_additional_bases = [x for x in range(1, 5)]
        
        # create initial orders
        self.orders = [[0] * 9] * self.TOTAL_NUM_ORDERS
        self.orders_next_activation = 0
        self.orders_full = None # only used for RefBox, default behavoir creates orders on time based on declarations
        
        # track what assembly step we are at
        self.order_stage = ["BS"] * self.ACTION_SPACE_SIZE
        
        # track what products we currently have
        self.products = []
        
        # track how many bases a ring station has buffered
        self.rings_buf_bases = [0, 0]
                
        # current time
        self.time = 0
        
        
        # in case we use the identical recreation, we define like the RefBox
        if self.RefBox:
            machines, down_period, rings, ring_additional_bases, order_declarations, orders_full = self.RefBox.game_parametrize()
            self.machines = machines
            self.machines_down_period = down_period
            self.rings = rings
            self.ring_additional_bases = ring_additional_bases
            self.order_declarations = order_declarations
            self.orders_full = orders_full
        else:
            # assumes all parameters have already ben initialized
            self.generate_game_setting()
        
        # we sort the list to have the order of appearance
        self.order_declarations.sort(key=lambda x: x[0])
        # actually assign current orders
        self.update_orders()


        return self.get_observation()

    def get_normal(self, mean, var):
        var = np.random.normal(mean, var)
        
        # bound the lower end to greater 0 preventing errors
        if var < 0:
            var = 0
        
        return var

    def step(self, action):
        self.episode_step += 1
        done = False

        # TODO: make products appear randomly on time for specified order slots; account quantity & characteristics
        # TODO: update for multiple robots
        # TODO: currently assume no activity during the avoidable wait time
        
        ### we assume selecting from one of the orders and computing/returning real-world-like intermediate step
        assert action >= 0 and action <= self.ACTION_SPACE_SIZE - 1
        order = self.orders[action]
        stage = self.order_stage[action]
        # TODO: for the double orders

        
        # correct processing_order to actual next machine
        to_machine, ring_pos, ring_col, ring_num = self.stage_to_machine(stage, order)
        
        # positional targets
        current_pos = self.robots[0]
        next_pos = self.machines[to_machine]
        
        # now act probabilistic! later substitute with real data!
        # unavoidable time spend on moving to machine; adding (thus multiplying) up random varibles per meter
        distance = current_pos.distance(next_pos)
        time_driving = self.get_normal(distance * self.move_mean, distance * self.move_var)
        # unavoidable time consumption; here we only grap and thus half
        time_mechanical = self.get_normal(self.grap_and_place_mean / 2, self.grap_and_place_var / 2)
        
        # assume 1.5m travel distance around machine
        
        
        if stage == "BS":
            # avoidable time spent on machine internal processing; BS is gaussian physical time
            time_wait = self.get_normal(self.machine_times["BS"][0], self.machine_times["BS"][1])
            
            # no reward for getting a base
            reward = 0
            
            # tracking what partial products we own, adding the base
            self.products.append([order[0]])
            
        elif stage == "R1":
            pass
            
        elif stage == "R2":
            pass    
        
        elif stage == "R3":
            pass
                    
        elif stage == "CS":
            pass
            
        elif stage == "DS":
            pass
            
        elif stage == "SS":
            pass
        
        
        ### update the robots position, implying it drove there
        self.robots[0] = next_pos
        
        ### implying getting the base by transition to next stage
        self.order_stage[action] = self.processing_order[self.processing_order.index(stage) + 1]
        
        ### apply the real time passed
        self.time += time_driving + time_mechanical + time_wait
        
        ###
        
        # we are finished with the episode at the end of the game
        if self.time >= 1020:
            done = True
        
        # we do not award points for steps which would have finished too late
        if self.time > 1020:
            reward = 0
        
        return self.get_observation(), reward, done


if __name__ == "__main__":
    print("Please import the file.")
    
    
    #### for debug scenario
    r = RefBox_recreated()
    self = env_rcll(r)
    self.reset()
    
#    assert False
    self.orders_full = [[1, 0, 0, 0, 2, 0, 0, 0, 1020], # @ 0006
                   [2, 3, 0, 0, 2, 0, 0, 0, 1020], # @ 0006
                   [1, 4, 1, 0, 2, 0, 0, 847, 1008], # @ 0239
                   [1, 1, 3, 4, 2, 0, 0, 677, 833], # @ 0006
                   [2, 0, 0, 0, 1, 0, 0, 420, 571], # @ 0268
                   [3, 0, 0, 0, 2, 1, 0, 639, 747], # @ 0403
                   [2, 0, 0, 0, 2, 0, 1, 840, 1020], # @ 0661
                   [3, 2, 0, 0, 2, 0, 0, 709, 816], # @ 0209
                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    self.order_stage[5] = [self.order_stage[5], "BS"]
    self.orders = [[1, 0, 0, 0, 2, 0, 0, 0, 1020], # @ 0006
                   [2, 3, 0, 0, 2, 0, 0, 0, 1020], # @ 0006
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 3, 4, 2, 0, 0, 677, 833], # @ 0006
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [3, 2, 0, 0, 2, 0, 0, 709, 816], # @ 0209
                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]
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

    
    data = []
    for _ in range(5000):
        smp = self.get_normal(self.grap_and_place_mean, self.grap_and_place_var)
        data.append(smp)
    plt.grid(True)
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')
    
    
    
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
    o = 2
    
    plt.figure(figsize=(15,7))
    for y, l in zip(t_rewards[o:], labels[o:]):
        plt.plot(t, y, label=l, linewidth=3, alpha=0.9)
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure(figsize=(15,7))
    plt.grid(True)
    plt.plot(t_rewards[8])
    
    
    
