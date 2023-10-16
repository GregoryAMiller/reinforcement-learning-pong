import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# %matplotlib inline



model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(210, 160, 3)),
    tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Only two actions: 2 and 3
])



optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def compute_loss(probabilities, actions, rewards):
    """
    Computes the loss for policy gradient methods.
    
    Args:
        probabilities: Tensor of shape (batch_size, num_actions). 
                       The probability of taking each action in each state.
        actions: Tensor of shape (batch_size,). The actions taken.
        rewards: Tensor of shape (batch_size,). The rewards obtained.
        
    Returns:
        loss: A scalar Tensor representing the loss.
    """
    # Ensure the tensors are of correct data type and shape
    actions = tf.cast(actions, tf.int32)
    rewards = tf.cast(rewards, tf.float32)
    
    # Compute the return G_t
    # Note: You might want to compute a discounted return
    # G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... 
    # where gamma is a discount factor in [0, 1]
    
    # For simplicity, let's consider G_t = r_t for this example
    G_t = rewards
    
    # Compute the log probabilities of the actions taken
    indices = tf.range(0, tf.shape(probabilities)[0]) * tf.shape(probabilities)[1] + actions
    chosen_probabilities = tf.gather(tf.reshape(probabilities, [-1]), indices)
    epsilon = 1e-10
    log_probabilities = tf.math.log(chosen_probabilities + epsilon)

    
    # Compute the loss
    loss = -tf.reduce_mean(log_probabilities * G_t)
    
    return loss



# Function to render environment and optionally display using matplotlib
def render(env, mode='human'):
    if mode == 'human':
        env.render()
    elif mode == 'rgb_array':
        img = env.render(mode='rgb_array')
        plt.imshow(img)
        plt.axis('off')
        plt.show()



# Define the environment
env = gym.make("ALE/Pong-v5", render_mode="human")


num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    # print(state)
    # Extract image data if state is a tuple
    actual_state = state[0] if isinstance(state, tuple) else state
    done = False
    
    states, actions, rewards, probs = [], [], [], []
    
    with tf.GradientTape() as tape:
        for t in range(1000):  # Limiting to 1000 timesteps per episode
            # if episode % 100 == 0:  # Render every 100 episodes
            render(env, mode='human')  
            # env.render()
            # Convert state to suitable input for network
            # Convert state to suitable input for network WITHOUT flattening
            state_input = tf.convert_to_tensor([actual_state], dtype=tf.float32) 
            
            # Predict action probabilities and choose action
            action_prob = model(state_input)

            initial_epsilon = 1.0
            final_epsilon = 0.01
            decay_rate = 0.995
            epsilon = initial_epsilon * (decay_rate ** episode)

            if np.random.rand() < epsilon:
                action = np.random.choice([2, 3])
            else:
                action = np.random.choice([2, 3], p=np.squeeze(action_prob.numpy()))

            # Take step in environment
            next_state, reward, done, info, additional_value = env.step(action)
            


            # print(state)
            # print(action)
            # if action == 2 or action == 3:  # if the paddle moves up or down
            #     reward += 0.05
            # if action != 2 and action != 3:  # if the paddle doesn't move
            #     reward -= 1
            # Amplify reward if the ball is hit
            if reward == 1:  # If the agent scores a point
                reward += 10  # Add an additional reward
            if reward == -1:  # If the agent misses the ball
                reward -= 2  # Penalize more heavily
            # Check if the ball was hit
            # if reward == 0:
            #     reward += 1
            
            
            # Store state, action, reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            probs.append(action_prob)
            
            state = next_state
            
            print(f"Episode: {episode}, Reward: {reward}, Total Reward: {np.sum(rewards)}")
            if done:
                break
        
        # Compute loss value
        print(f'Computing Loss')
        loss_value = compute_loss(probs, actions, rewards)
        print(f'loss_value: {loss_value}')
    # Compute gradient and perform optimization
    print('calulating gradient')
    grads = tape.gradient(loss_value, model.trainable_variables)
    print('gradient done')
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print('optimizer done')
    # Optionally, log metrics like episode reward, loss etc.
    
    if episode % 10 == 0:
        print(f"END: Episode: {episode}, Total Reward: {np.sum(rewards)}")

# Close the environment
print('Closing environment')
env.close()