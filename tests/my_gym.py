#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
#import keras.backend.tensorflow_backend as backend
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import deque
from tqdm import tqdm
from PIL import Image
import time
import numpy as np
import random
import os
#import cv2

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


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    # so one can check if two blobs are overlapping
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        # diagonal
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        # in x direction
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        # in y direction
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        
        # dont move
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = False
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 100
    FOOD_REWARD = 50
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == - self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self, p=0.5):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
#        cv2.imshow("image", np.array(img))  # show it!
#        cv2.waitKey(1)
#        img = Image.fromarray(np.array(img), "RGB")
#        img.show()
        plt.imshow(np.array(img))
        plt.show(block=False)
        plt.pause(p)
        plt.close()
#        time.sleep(1)
#        plt.close("all")
#        input("press key to continue..")

    # FOR CNN
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


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
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

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
        model.add(Dense(30, activation='relu', input_shape=(19,)))
        model.add(Dense(30, activation='relu'))
#        model.add(Dense(10, activation='relu'))

        # the output llayer, ACTION_SPACE_SIZE = how many choices (9)
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        
        # adam optimizer with learning rate lr and track accuracy
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
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
    UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
    MODEL_NAME = 'rcll_v3'
    MIN_REWARD = -5  # For model save, as -300 for when an enemy hit
    MEMORY_FRACTION = 0.20
    
    # Environment settings
    EPISODES = 50000
    
    # Exploration settings
    epsilon = 0.6  # not a constant, going to be decayed
    EPSILON_DECAY = 0.99995
    MIN_EPSILON = 0.001
    
    #  Stats settings
    AGGREGATE_STATS_EVERY = 5  # episodes
    SHOW_PREVIEW = True
    
    
    ### Environment setup
#    env = BlobEnv()
    env = env_rcll()

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
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
                a = "best   action"
            else:
                # Get random action
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)
                a = "random action"
    
            new_state, reward, done = env.step(action)
    
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward
    
            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:                
#                go = ""
#                if action == 0:
#                    go = "BOTTOM_RIGHT"
#                elif action == 1:
#                    go = "TOP_LEFT"
#                elif action == 2:
#                    go = "TOP_RIGHT"
#                elif action == 3:
#                    go = "BOTTOM_LEFT"
#                elif action == 4:
#                    go = "RIGHT"
#                elif action == 5:
#                    go = "LEFT"
#                elif action == 6:
#                    go = "TOP"
#                elif action == 7:
#                    go = "BOTTOM"
#                elif action == 8:
#                    go = "STAY"
#                print("moving {} by {} from {} to {}; got reward {}".format(go.ljust(12), a, current_state, new_state, reward))
                print("taking {} {} giving {} reward or total {} and done={} | {}".format(a, action, reward, episode_reward, done, epsilon))
                env.render()
    
            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)
    
            current_state = new_state
            step += 1
    
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
    
            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                MIN_REWARD += 1 # only save better models
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    
        # Decay epsilon, same as before
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    