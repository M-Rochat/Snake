import gym
from gym import spaces
import numpy as np
import cv2
import random

SNAKE_LEN_GOAL = 15
SIZE = 50
DIM = 5


def new_apple(snake_position):
    remaining_position = [[i, j] for i in range(DIM) for j in range(DIM) if [i, j] not in snake_position]
    return random.choice(remaining_position)


def collision_with_boundaries(snake_head):
    return snake_head[0] >= DIM or snake_head[0] < 0 or snake_head[1] >= DIM or snake_head[1] < 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    return snake_head in snake_position[1:]


class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-5, high=5,shape=(5 + 2 * SNAKE_LEN_GOAL,), dtype=np.float32)

    def step(self, action):
        self.turn += 1
        # Change the head position based on the action
        if action == 1:
            self.snake_head[0] += 1
        elif action == 0:
            self.snake_head[0] -= 1
        elif action == 2:
            self.snake_head[1] += 1
        elif action == 3:
            self.snake_head[1] -= 1

        done = False
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.snake_position.insert(0, list(self.snake_head))
            self.score += 1
            if len(self.snake_position) < DIM ** 2:
                self.apple_position = new_apple(self.snake_position)
            else:
                done = True
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        new_reward = self._get_total_reward()
        reward = new_reward - self.total_reward
        self.total_reward = new_reward

        # On collision
        if collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_position):
            done = True
            reward = -1

        if self.score <= -1:
            done = True

        observation = self._get_observation()

        info = {}
        return observation, reward, done, info

    def reset(self):
        # Initial Snake and Apple position
        _mid = int(DIM / 2)
        self.snake_position = [[_mid, _mid], [_mid - 1, _mid], [_mid - 2, _mid]]
        self.apple_position = new_apple(self.snake_position)
        self.score = 0
        self.turn = 0
        self.snake_head = self.snake_position[0].copy()
        self.total_reward = self._get_total_reward()

        observation = self._get_observation()
        return observation  # reward, done, info can't be included

    def _get_total_reward(self):
        dist_to_apple = sum(abs(self.snake_head[i] - self.apple_position[i]) for i in range(2))
        return self.score - dist_to_apple / (2 * DIM) - self.turn / (6 * DIM)

    def _get_observation(self):
        # head_x, head_y, apple_x, apple_y, snake_length, previous_moves
        # -> x down y
        head_x = self.snake_head[0] / DIM
        head_y = self.snake_head[1] / DIM
        apple_x = self.apple_position[0] / DIM - head_x
        apple_y = self.apple_position[1] / DIM - head_y
        snake_length = len(self.snake_position) / SNAKE_LEN_GOAL

        previous_pos = []
        for i in range(SNAKE_LEN_GOAL):
            if i + 1 < len(self.snake_position):
                previous_pos += [(self.snake_position[i + 1][k] - self.snake_head[k]) / DIM for k in range(2)]
            else:
                previous_pos += [0, 0]

        observation = [head_x, head_y, apple_x, apple_y, snake_length] + previous_pos
        return np.array(observation)

    def render(self, mode='human', done = False):
        img = np.zeros((SIZE * DIM + 100, SIZE * DIM, 3), dtype='uint8')

        # Display Apple
        cv2.rectangle(img, (self.apple_position[0] * SIZE, self.apple_position[1] * SIZE),
                      (self.apple_position[0] * SIZE + SIZE, self.apple_position[1] * SIZE + SIZE),
                      (0, 0, 255), 4)
        # Display Snake
        length = len(self.snake_position)
        for i in range(length):
            position = self.snake_position[i]
            x, y = position[0], position[1]
            delta = i / 50
            cv2.rectangle(img, (int((x + delta) * SIZE), int((y + delta) * SIZE)),
                          (int((x + 1 - delta) * SIZE), int((y + 1 - delta) * SIZE)), (0, 255, 0), 4)

        # Display Score
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Score : {}'.format(self.score), (int(SIZE * DIM /5), SIZE * DIM + 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('snake', img)
        if not done:
            cv2.waitKey(100)
        else:
            cv2.waitKey(1500)

    def close(self):
        cv2.destroyAllWindows()
