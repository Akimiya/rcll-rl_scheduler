#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
#import keras.backend.tensorflow_backend as backend
import tensorflow as tf

from collections import deque
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import random
import os

from env_orders import env_rcll


# Own Tensorboard class, as per default it updates on every .fit()
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self, simple=False):
        self.simple = simple
        # need TWO models: as we will start learning a lot or random actions
        # Main model / Training network; using dot fit (trained) every single step
        if self.simple:
            self.model = self.create_model_simple()
#            self.norm_factor = 9
            self.norm_factor = 1
        else:
            self.model = self.create_model()
            self.norm_factor = 255

        # prediction model / Target network; dot predict for every single step the agent takes
        if self.simple:
            self.target_model = self.create_model_simple()
        else:
            self.target_model = self.create_model()
        # set weights to same as above, will do every X steps later as well
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training; want agent prediction have some consistency
        # it is like the batches in NN to not learn on one sample at a time (or we fit to that sample too much)
        # random sampling of this is the batch we train on
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, datetime.now().strftime("%y%m%d_%H-%M-%S")))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    
    
    def create_model(self):
        model = Sequential()
        
        # 256 convolutions in window 3x3
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # this converts our 3D feature maps to 1D feature vectors so can use Dense
        model.add(Flatten())
        model.add(Dense(64))

        # the output llayer, ACTION_SPACE_SIZE = how many choices (9)
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        # adam optimizer with learning rate lr and track accuracy
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
    
    def create_model_simple(self):
        model = Sequential()
        
        # input is the 4 elements long touple
        # , kernel_initializer='uniform'
        model.add(Dense(50, activation='relu', input_shape=(19,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))

        # the output llayer, ACTION_SPACE_SIZE = how many choices (9)
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        
        # adam optimizer with learning rate lr and track accuracy
        optimizer = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=optimizer, metrics=['accuracy'])
        
        return model
    
    # Adds step's data to a memory replay array
    # transition = (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        # unpack state with the * Elements from this iterable are treated as if they were additional positional arguments
        s = np.array(state)
        if self.simple:
            ret = self.model.predict(s[np.newaxis,:] / self.norm_factor)
        else:
            ret = self.model.predict(s.reshape(-1, *s.shape) / self.norm_factor)[0]
        return ret
    
     # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        # check if we want to train; from replay memory we grab only a batch, as we dont want to overfit on same data (?)
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get *current states* from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / self.norm_factor # also scale image
        current_qs_list = self.model.predict(current_states) # the updated one

        # Get *future states* from minibatch, then query NN model for Q values; AFTER we take steps
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / self.norm_factor
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = [] # INPUT feature set e.g. images
        y = [] # OUTPUT labels e.g. action we decide

        # Now we need to enumerate our batches
        # the part of (reward + discount * max_a Q(s_{t+1}, a))
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            # Depending on what the NN predicted we want to update the NN to the more appropriate result
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        # we already did random sampling so no shuffle; callback to the custom one
        # ONLY FIT IF ON terminal_state, else nothing
        self.model.fit(np.array(X) / self.norm_factor, np.array(y), batch_size=MINIBATCH_SIZE, 
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        # Update target network counter every episode, so determine if we want update target_model
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights()) # copy weights
            self.target_update_counter = 0
        
if __name__ == "__main__":
    ### globals
    DISCOUNT = 0.99
    REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
    MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
    MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
    UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)
    MODEL_NAME = 'rcll_v8_lim-no_norm_upd10_hl5-n50'
    MIN_REWARD = -5  # For model save, as -300 for when an enemy hit
    MEMORY_FRACTION = 0.20
    
    # Environment settings
    EPISODES = 80000
    
    # Exploration settings
    epsilon = 0.5  # not a constant, going to be decayed
    EPSILON_DECAY = 0.99997
    MIN_EPSILON = 0.001
    
    #  Stats settings
    DEBUG_EVERY = 1  # episodes
    SHOW_PREVIEW = True
    
    
    ### Environment setup
    env = env_rcll(normalize=True)

    # For stats
    ep_rewards = [-200]
    
    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # Memory fraction, used mostly when trai8ning multiple agents; multiple models on same machine
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    #backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    
    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')


    ### DQN setup
    agent = DQNAgent(simple=True)
    
    order_selection = [0] * 3
    order_complexities = [0] * 4
    
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode
    
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1
    
        # Reset environment and get initial state; match OpenAI syntax with gym
        current_state = env.reset()
    
        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                a = "best   action"
                # Get action from Q table
                options = agent.get_qs(current_state)[0]
                agent.tensorboard.update_stats(discard=options[9], cap_grey=options[8], base_red=options[0])
                # TODO: does how does it influence ignoring some output? Find better approach, as some actions may be escalating?
                options[9] = options[0] - 1 # we dont use discard; make not max
                action = np.argmax(options)
                
                phase = env.order_stage
#                if phase == 0:
#                    action = np.argmax(options[0:3])
#                elif phase == 1:
#                    action = np.argmax(options[3:7]) + 3
#                elif phase == 2:
#                    doing = env.doing_order[0] # assumption that only one order remains here
#                    has_rings = 3 - env.orders[doing][1:4].count(0)
#                    done_rings = 3 - env.pipeline[1:4].count(0)
#                    if has_rings < done_rings:
#                        # if still missing some rings
#                        action = np.argmax(options[3:7]) + 3
#                    else:
#                        # finishing with cap
#                        action = np.argmax(options[7:9]) + 7
                
            else:
                order = env.orders
                a = "random action"# ({} => {})".format(order, env.doing_order)
                # Get random *viable* action (excluding not tracking order)
                phase = env.order_stage
                if phase == 0:
                    action = np.random.randint(0, 3)
                elif phase == 1:
                    action = np.random.randint(3, 7)
                elif phase == 2:
                    action = np.random.randint(3, 9) # currently includes more then needed, but no discard
#                    doing = env.doing_order[0]
#                    has_rings = env.orders[doing][1:4].count(0)
#                    done_rings = env.pipeline[1:4].count(0)
#                    if has_rings < done_rings:
#                        # if still missing some rings
#                        action = np.random.randint(3, 7)
#                    else:
#                        # finishing with cap
#                        action = np.random.randint(7, 9)
    
            new_state, reward, done = env.step(action)
#            assert reward != -20
    
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward
            
            # debug analysis on which oder finished
            if reward == 20:
                for doing in env.doing_order:
                    order_selection[doing] += 1
                    
                    complexity_took = 3 - env.orders[doing][1:4].count(0)
                    order_complexities[complexity_took - 1] += 1
                    if complexity_took < max(env.complexities):
                        order_complexities[3] -= 1
                
                # testing behavoir for duplicates
                if len(env.doing_order) != 1: 
                    print("taking {} {} giving {} reward with total {} and done={} || selected: {}{} || {}".format(a, action, reward, episode_reward, done, order_selection, order_complexities, epsilon))
                    env.render()
                    assert False
    
            if SHOW_PREVIEW and not episode % DEBUG_EVERY:
                print("taking {} {} (at {}) giving {} reward with total {} and done={} epsion={:.4f} || selected: {}{} || ".format(
                        a, action, phase, reward, episode_reward, done, epsilon, order_selection, order_complexities))
                env.render()
    
            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)
    
            current_state = new_state
            step += 1
    
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        write_every = 50
        if episode % write_every == 0 and episode >= 0:
            agent.tensorboard.update_stats(C1=order_complexities[0])
            agent.tensorboard.update_stats(C2=order_complexities[1])
            agent.tensorboard.update_stats(C3=order_complexities[2])
            
            order_complexities = [0] * 4 # reset
            
            average_reward = sum(ep_rewards[-write_every:])/len(ep_rewards[-write_every:])
            min_reward = min(ep_rewards[-write_every:])
            max_reward = max(ep_rewards[-write_every:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, 
                                           reward_max=max_reward, epsilon=epsilon)
    
            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                MIN_REWARD = min_reward + 1 # only save better models
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    
        # Decay epsilon, same as before
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    