import random
import numpy as np

# Game settings
WIDTH = 600
segment_width = 50
NUM_BINS = WIDTH // 10
ACTIONS = [0, 1]  # 0 = don't shoot, 1 = shoot
TARGET_X = (WIDTH - segment_width * 5) // 2

# Q-table
Q = np.zeros((NUM_BINS, len(ACTIONS)))

# Learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return np.argmax(Q[state])

def update_q(state, action, reward, next_state):
    best_next = np.max(Q[next_state])
    Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

def get_score(x):
    rel_x = x - TARGET_X
    if 0 <= rel_x < segment_width * 5:
        segment = int(rel_x // segment_width)
        if segment == 2:
            return 5
        elif segment in [1, 3]:
            return 3
        elif segment in [0, 4]:
            return 1
    return -7

def simulate():
    global epsilon
    episodes = 1000
    max_shots = 100

    for episode in range(episodes):
        score = 0
        for _ in range(max_shots):
            x = random.randint(0, WIDTH - 1)
            state = min(x // 10, NUM_BINS - 1)
            action = choose_action(state)

            if action == 1:
                reward = get_score(x)
                next_state = min(x // 10, NUM_BINS - 1)
                update_q(state, action, reward, next_state)
                score += reward
            # else: do nothing

        print(f"Episode {episode+1}: Total Score = {score}")
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # print("\nFinal Q-table (rounded):")
    # print(np.round(Q, 2))

simulate()
