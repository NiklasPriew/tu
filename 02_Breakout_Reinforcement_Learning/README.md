# Breakout.ai

The following packages are required:
* `numpy` and
* `pygame` for watching the AI play the game.

The code in the present condition generates three models for the simple `single` brick layout within roughly 2 minutes by executing `python3 .\learn_parallel.py`. By executing `python3 .\breakout.py`, one of the models starts playing Breakout.

## HowTo
* Learning parameters and hyperparameters can bet set in `learn_parallel.py` and `ball.py`
* `ball.py/__init__` contains hyperparameters
  * `BRICK_REWARD` ... reward for hitting a brick
  * `ALL_BRICKS_REWARD` ... reward for hitting all bricks
  * `LOST_REWARD` ... reward/penalty for losing the game
  * `RESET_REWARD` ... reward/penalty for losing the ball
  * `TIME_LIMIT` ... time limit (maximum number of episodes)
* `learn_parallel.py/__main__` contains mostly learning parameters
  * `grid_height` ... Game grid height in cells
  * `grid_width` ... Game grid width in cells (must be a multiple of 3)
  * `grid_type` ... Layout of bricks (must be one of `single`, `double`, `bridge`, `towers`) 
  * `epsilons` ... probability of random actions
  * `gammas` ... discount rate for rewards
  * `n_runs` ... number of iterations for the learn-play cycle
  * `n_episodes_learn` ... number of training episodes
  * `n_episodes_play` ... number of playing episodes

### Learning
The pre-defined parameters should be good enough for most models. More difficult brick layouts require a higher `TIME_LIMIT`, but a limit between 100 and 500 should be sufficient. The number of training episodes `n_episodes_learn` should stay between 1.000 and 2.000 in order to balance runtime and training accuracy.

How to create a model:
1. Set the hyperparameters mentioned above in `__init__` from `ball.py`.   
2. Set the learning parameters mentioned above in `main` from `learn_parallel.py`. A separate process is started for each combination of `epsilons` and `gammas` in order to parallelize this step.
3. Start `learn_parallel.py` with a `python3` interpreter
4. Finally, verify that the directory `models` contains a `.pickle` file with the chosen parameters in its filename


### (Re)Playing
For this step, adaptations are only required in `breakout.py`.

How to run a model and watch it play Breakout:
1. Set the learning parameters (`grid_width`, `grid_height`, `grid_type`, `epsilon`, `gamma`) in `main` to chose a generated model
2. (Optional: Set `ball_x_speed_start` in line 18 to one of `[None,-2,-1,0,1,2]` to force a start direction for the ball, where `None` represents a random action)
3. (Optional: Set `efps` in line 26 to adapt the replay speed in terms of episodes/frames per second)
4. Verify that a model file in the `models` directory with matching parameters exist 
5. Start `breakout.py` with a `python3` interpreter