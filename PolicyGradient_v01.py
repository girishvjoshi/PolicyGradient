import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque

# Initialize the model and learning Parameters
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Discount factor
GAMMA = 0.99
# Leanring rate
Lr = 1e-2

#Maximum Episodes
MAX_EPISODE = 10000
# Length of Each Episode
LEN_EPISODE = 999

#Sample Trajectories number for update
UPDATE_FREQ = 5

# Network Hidden Layer Size
n_hidden_layer = 10

def weight_variable(shape):
    initial_val = tf.truncated_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial_val)

def bias_variable(shape):
    initial_val = tf.constant(0.03,shape=shape)
    return tf.Variable(initial_val)

def discount_rewards(r):
    discounted_reward = np.zeros_like(r)
    running_reward = 0
    for t in reversed(range(0,r.size)):
        running_reward = running_reward*GAMMA + r[t]
        discounted_reward[t] = running_reward
    return discounted_reward

class PolicyGradient():
    
    def __init__(self,state_dim,action_dim,Lr):
        
        # Network Inputs
        self.inputs = tf.placeholder(dtype=tf.float32,shape=[None,state_dim], name='State_inputs')
        # Input Layer
        w1 = weight_variable([state_dim,n_hidden_layer])
        b1 = bias_variable([n_hidden_layer])

        # output layer
        w2 = weight_variable([n_hidden_layer,action_dim])
        b2 = bias_variable([action_dim])

        # 1st Hidden Layer
        h1 = tf.nn.relu(tf.matmul(self.inputs,w1)+b1)
        # Output Layer
        self.out = tf.nn.softmax(tf.matmul(h1,w2)+b2,name='Net_outputs')
        
        
        self.reward_holder = tf.placeholder(dtype=tf.float32,shape=[None])
        self.action_holder = tf.placeholder(dtype=tf.float32,shape=[None,action_dim])
        
        self.responsible_outs = tf.reduce_sum(self.out*self.action_holder,1)
        
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outs)*self.reward_holder)
        
        self.network_params = tf.trainable_variables()
        self.gradient_holder = []

        # Save File
        self.save_file = "PG/DQN_net"
        # Network Save
        self.saver = tf.train.Saver()
        
        for idx, grad in enumerate(self.network_params):
            grad_placeholder = tf.placeholder(tf.float32)
            self.gradient_holder.append(grad_placeholder)
            
        self.net_gradient = tf.gradients(self.loss,self.network_params)
        self.optimize = tf.train.AdamOptimizer(Lr).apply_gradients(zip(self.gradient_holder,self.network_params))


tf.reset_default_graph() #Clear the Tensorflow graph.
agent = PolicyGradient(state_dim, action_dim, Lr)

with tf.Session() as sess:

    # Set Random Seed for repeatability
    np.random.seed(1234)
    tf.set_random_seed(1234)
    env.seed(1234)

    sess.run(tf.global_variables_initializer())
    total_reward = []
    total_length = []
    Avg_Reward_History = []
    
    gradBuffer = sess.run(tf.trainable_variables())
    for idx, grad in enumerate(gradBuffer):
        gradBuffer[idx] = grad*0
    
    for epoch in range(MAX_EPISODE):
        s = env.reset()
        running_r = 0
        ep_history = deque()
        for ep_length in range(LEN_EPISODE):
            action_hot = np.zeros((1,2))
            action_prob = sess.run(agent.out, feed_dict={agent.inputs:np.reshape(s,(1,state_dim))})
            action_prob_selected = np.random.choice(action_prob[0], p=action_prob[0])
            action = np.argmax(action_prob == action_prob_selected)
            action_hot[0][action] = 1
            s1,r,done,info = env.step(action)
            ep_history.append((np.reshape(s,(state_dim,)),np.reshape(action_hot,(action_dim,)),r,np.reshape(s1,(state_dim,))))
            s = s1
            running_r += r
            if done:
                s_batch = np.array([_[0] for _ in ep_history])
                a_batch = np.array([_[1] for _ in ep_history])
                r_batch = np.array([_[2] for _ in ep_history])
                s1_batch = np.array([_[3] for _ in ep_history])
                r_batch = discount_rewards(r_batch)
                
                grads = sess.run(agent.net_gradient,feed_dict={agent.inputs:s_batch,
                                                                agent.reward_holder:r_batch,
                                                                agent.action_holder:a_batch})
                                                            
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad
                    
                if epoch % UPDATE_FREQ == 0 and epoch!=0:
                    feed_dict = dict(zip(agent.gradient_holder, gradBuffer))
                    sess.run(agent.optimize, feed_dict=feed_dict)
                    for idx, grad in enumerate(gradBuffer):
                        gradBuffer[idx] = grad*0
                total_reward.append(running_r)
                total_length.append(ep_length)
                break
        if epoch % 100 == 0:
            Avg_Reward_History.append(np.mean(total_reward[-100:]))
            print(np.mean(total_reward[-100:]))

    agent.saver.save(sess,agent.save_file)

plt.figure(1)
plt.plot(Avg_Reward_History)
plt.show()