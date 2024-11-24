import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ProgbarLogger
from sklearn.preprocessing import StandardScaler
from src.core import SampleSoundGenAI,FrameData,GameData
import gym
import os
from tensorflow.keras import layers
import librosa
from tqdm import tqdm

# load processed data
data = np.load(os.path.join('KirariAI', 'KiraAI', 'trained_model', 'processed_data2.npz'))
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Standardize feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a pretrained model
pretrained_model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation='relu', name='dense_pretrained1'),
    Dropout(0.5, name='dropout1'),
    Dense(128, activation='relu', name='dense_pretrained2'),
    Dropout(0.5, name='dropout2'),
    Dense(64, activation='relu', name='dense_pretrained3'),
    Dropout(0.5, name='dropout3'),
    Dense(y_train.shape[1], activation='softmax', name='output_pretrained')
])

# Adjust the learning rate of the optimizer
optimizer = Adam(learning_rate=0.02)

# Compile the pretrained model
pretrained_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set an early stop callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#  Train a pretrained model
pretrained_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate pretrained model
loss, accuracy = pretrained_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# save pretrained model
#pretrained_model.save("D:/FightingICE/KirariAI/my_model.keras")

# Load a pretrained supervised learning model
pretrained_model = tf.keras.models.load_model(os.path.join('KirariAI', 'KiraAI', 'trained_model', 'my_model.keras'))

# Create a new model and copy the schema and weights of the pretrained model
input_shape = (X_train.shape[1],)
num_actions = 16  # 

model = tf.keras.Sequential(name="dqn_model")

# Add fully connected layers
model.add(layers.Input(shape=input_shape, name='input'))
model.add(layers.Dense(256, activation='relu', name='dense_pretrained1'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu', name='dense_pretrained2'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu', name='dense_pretrained3'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_actions, activation='linear', name='output_layer'))

# Prints summary information for the new model
model.summary()

# Initialize the weights of the new model with the same weights as the pretrained model
for i, layer in enumerate(pretrained_model.layers[:-1]):  # Skip the last layer
    if isinstance(layer, Dense):  # Set weights only for the Dense layer
        if len(layer.get_weights()) > 0:
            model.layers[i].set_weights(layer.get_weights())
            print(f"Successfully set weights for layer {model.layers[i + 1].name}")
        else:
            print(f"No weights to set for layer {model.layers[i + 1].name}")
    else: continue
            


# Create a target model (for DQN) to ensure that the input layer and architecture are consistent
model_target = tf.keras.models.clone_model(model)
model_target.set_weights(model.get_weights())

class FightingICEEnv(gym.Env):
    def __init__(self):
        super(FightingICEEnv, self).__init__()
        self.sound_gen_ai = SampleSoundGenAI()

        self.action_space = gym.spaces.Discrete(16)  # 
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        self.action_dict = {
            0: "AIR_A",
            1: "DASH",
            2: "BACK_JUMP",
            3: "JUMP",
            4: "CROUCH",
            5: "THROW_A",
            6: "THROW_B",
            7: "STAND_A",
            8: "STAND_B",
            9: "STAND_FA",
            10: "STAND_FB",
            11: "STAND_GUARD",
            12: "AIR_GUARD",
            13: "THROW_HIT",
            14: "THROW_SUFFER",
            15: "STAND_A",
            16: "UNKNOWN"  # 
        }
        self.health = 400
        self.enemy_health = 400
        #self.enemy_attack_damage = np.random.randint(0, 5)  # Let's say the enemy deals 5 damage per attack
        # Define an index of aggressive actions
        self.attacks = [5,6,7,8,9,10,13,14,15]

    def reset(self):
        self.sound_gen_ai.init_round()
        self.health = 400
        self.enemy_health = 400
        return self._get_audio_features()

    def step(self, action):
        observation = self._get_audio_features()
        reward = 0.0
        done = False
        info = {}

        # Suppose 'get_game_state()' returns a dictionary of whether the action hit the target and the damage it caused
        game_state = self.get_game_state(action)
        # If it's an aggressive move, increase the base bonus
        if action in self.attacks:#Later added to hit
            reward += 2.0
            #print("+1")
            if game_state["hit"]:
              self.enemy_health -= game_state["damage"]
              reward += game_state["damage"]
              #print(f"+{reward}")

        #
        #self.health -= self.enemy_attack_damage
        enemy_action = np.random.randint(1, 15)
        game_state = self.get_game_state(enemy_action)

        if enemy_action in self.attacks:
            if game_state["hit"]:
              self.health -= game_state["damage"]
              reward -= game_state["damage"]

        # Check the enemy's HP
        if self.enemy_health <= 0:
            done = True
            reward += 50  # Defeat enemies to get a lot of rewards
            #print("+20")
        if self.health <= 0:
            done = True
            reward -= 50


    
        # Print actions and bonus values
        #print(f"Action: {action}, Reward: {reward}")

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _get_audio_features(self):
        audio_sample = np.random.uniform(-1.0, 1.0, 22050).astype(np.float32)
        mfcc = librosa.feature.mfcc(y=audio_sample, sr=22050, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc

    def get_game_state(self, action):
        hit = np.random.rand() > 0.5 # Suppose a 50% probability of hitting the target
        damage = np.random.randint(5, 10) if hit else 0
        return {"hit": hit, "damage": damage}

    
env = FightingICEEnv()

# Hyperparameters
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = (epsilon_max - epsilon_min)
batch_size = 32
gamma = 0.99
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.Huber()

# To train a DQN model, the ProgbarLogger callback function is added to display the training progress bar
def train():
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    epsilon = epsilon_max

    min_episodes = 2  # Minimum number of Episods
    reward_threshold = 250  # Reward thresholds

    while True:
        state = env.reset()
        episode_reward = 0

        for timestep in tqdm(range(1, 10000)):
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                state_input = np.expand_dims(state, axis=0)
                q_values = model.predict(state_input, verbose=0)
                action = np.argmax(q_values[0])

            state_next, reward, done, _ = env.step(action)
            episode_reward += reward

            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            rewards_history.append(reward)
            done_history.append(done)
            state = state_next

            if len(done_history) > batch_size:
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = np.array([rewards_history[i] for i in indices])
                action_sample = np.array([action_history[i] for i in indices])
                done_sample = np.array([done_history[i] for i in indices])

                future_rewards = model_target.predict(state_next_sample, verbose=0)
                future_rewards_max = np.max(future_rewards, axis=1).reshape(-1)

                rewards_sample = rewards_sample.reshape(-1, 1)
                done_sample = done_sample.reshape(-1, 1)

                updated_q_values = rewards_sample + gamma * future_rewards_max * (1 - done_sample)

                masks = tf.one_hot(action_sample, depth=num_actions)

                with tf.GradientTape() as tape:
                    q_values = model(state_sample, training=True)
                    q_action = tf.reduce_sum(q_values * masks, axis=1)
                    loss = loss_function(updated_q_values, q_action)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if done:
                #print("reward:",reward)
                break

        episode_reward_history.append(episode_reward)#reward与·episode_reward
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]

        running_reward = np.mean(episode_reward_history)
        print("episode_count:",{episode_count},"   ",{episode_reward})

        if  running_reward > reward_threshold and episode_count > min_episodes:
            print("running_reward",running_reward)
            print(f"solved in {episode_count} !")
            
            break
        
        if(episode_count>1000):
            print("1000times")
            break
        
        
        episode_count += 1
        
        
        epsilon -= epsilon_interval / 100000
        epsilon = max(epsilon, epsilon_min)


train()

model.save(os.path.join('KirariAI', 'KiraAI', 'trained_model', 'DQNsoundAdded_model.keras'))
print("success")