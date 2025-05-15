import pygame
import random
import numpy as np
import cv2
import os

pygame.init()
WIDTH, HEIGHT = 600, 400
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Shooter")

# Video writer үчүн mp4 codec (H264 же mp4v колдонсок болот)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' - mp4 форматы үчүн жакшы codec
video_out = cv2.VideoWriter('training_output.mp4', fourcc, 30.0, (WIDTH, HEIGHT))

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
FPS = 100
clock = pygame.time.Clock()

target_y = 100
segment_width = 50
target_x = (WIDTH - segment_width * 5) // 2

class Bullet:
    def __init__(self, x):
        self.x = x
        self.y = HEIGHT - 10
        self.radius = 3
        self.speed = 7
        self.hit = False

    def move(self):
        self.y -= self.speed

    def draw(self, win):
        pygame.draw.circle(win, WHITE, (self.x, self.y), self.radius)

class Shooter:
    def __init__(self):
        self.x = random.randint(0, WIDTH - 1)
        self.speed = 10

    def move(self):
        self.x += random.choice([-self.speed, 0, self.speed])
        self.x = max(0, min(WIDTH - 1, self.x))

    def draw(self, win):
        pygame.draw.rect(win, GREEN, (self.x - 5, HEIGHT - 15, 10, 10))

def draw_target(win):
    colors = [YELLOW, GREEN, RED, GREEN, YELLOW]
    for i in range(5):
        pygame.draw.rect(win, colors[i],
                         (target_x + i * segment_width, target_y, segment_width, 20))

def get_score(bullet):
    if target_y <= bullet.y <= target_y + 20:
        rel_x = bullet.x - target_x
        if 0 <= rel_x < segment_width * 5:
            segment = int(rel_x // segment_width)
            if segment == 2:
                return 5
            elif segment in [1, 3]:
                return 3
            elif segment in [0, 4]:
                return 1
    return -7

NUM_BINS = WIDTH // 10
ACTIONS = [0, 1]
Q = np.zeros((NUM_BINS, len(ACTIONS)))

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

def save_frame():
    raw_str = pygame.image.tostring(WIN, "RGB")
    image = np.frombuffer(raw_str, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    video_out.write(image_bgr)

def main():
    global epsilon
    font = pygame.font.SysFont("Arial", 20)
    episodes = 1000
    max_shots = 100

    for episode in range(episodes):
        shooter = Shooter()
        bullets = []
        total_shots = 0
        score = 0

        run = True
        while run and total_shots < max_shots:
            clock.tick(FPS)
            WIN.fill(BLACK)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    video_out.release()
                    pygame.quit()
                    return

            shooter.move()
            state = min(shooter.x // 10, NUM_BINS - 1)
            action = choose_action(state)

            if action == 1:
                bullets.append(Bullet(shooter.x))
                total_shots += 1

            for bullet in bullets[:]:
                bullet.move()
                bullet.draw(WIN)
                if not bullet.hit and bullet.y <= target_y + 20:
                    reward = get_score(bullet)
                    next_state = min(shooter.x // 10, NUM_BINS - 1)
                    update_q(state, action, reward, next_state)
                    score += reward
                    bullet.hit = True
                if bullet.y < 0:
                    bullets.remove(bullet)

            draw_target(WIN)
            shooter.draw(WIN)

            score_text = font.render(f"Episode: {episode+1} | Score: {score}", True, WHITE)
            shots_text = font.render(f"Shots: {total_shots}/{max_shots}", True, WHITE)
            WIN.blit(score_text, (10, 10))
            WIN.blit(shots_text, (10, 35))

            pygame.display.update()

            if 994 <= episode + 1 <= 999:
                save_frame()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    video_out.release()
    pygame.quit()

main()
