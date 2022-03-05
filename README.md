# Snake
Example of reinforcement learning using gym environment.

![](snake.gif)

This project is inspired by Sentdex's tutorial https://www.youtube.com/watch?v=XbWhJdQgi7E.

To summarize, the major changes are:
- For the observation I removed the last moves and added the relative position of the body parts of the snake.
- For the reward I put l1 distance (instead of l2) to the apple and a time penalty.
- I improved the visualization and added a way to create a GIF.

The base code for the snake game comes from https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb.
