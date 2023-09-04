import numpy as np

from ball import Ball
from brick import Bricks
from paddle import Paddle


class Breakout:
    def __init__(self, layout,
                 ball_pos_x: int, ball_pos_y: int, ball_speed_x: int, ball_speed_y: int,
                 paddle_pos_x: int, paddle_speed_x: int,
                 grid_height: int, grid_width: int, grid_type:str):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_type = grid_type
        self.rng = np.random.default_rng(0)
        self.bricks = Bricks(layout)
        self.layout_copy = layout.copy()
        self.ball = Ball(ball_pos_x, ball_pos_y, ball_speed_x, ball_speed_y, grid_height, grid_width)
        self.paddle = Paddle(paddle_pos_x, paddle_speed_x, grid_width)

    def update(self, action: int):
        self.paddle.update(action)
        reset, reward, game_status, win = self.ball.update(self.paddle, self.bricks)
        if reset:
            np.copyto(self.bricks.layout, self.layout_copy)
            self.bricks.count_remaining_bricks = self.bricks.layout.sum()
            self.paddle.x_pos = self.grid_width // 2 - 2
            self.paddle.x_speed = 0
            self.ball.x_pos = self.grid_width // 2
            self.ball.y_pos = 1
            self.ball.x_speed = self.rng.integers(low=-2, high=2, size=1)[0] #if ball_x_speed_start is None else ball_x_speed_start
            self.ball.y_speed = 1
        state_bricks = self.bricks.layout.ravel()
        state_ball_paddle = np.asarray([self.ball.x_pos, self.ball.y_pos,
                                        self.ball.x_speed, self.ball.y_speed,
                                        self.paddle.x_pos, self.paddle.x_speed], dtype=np.int8)
        return reset, reward, state_ball_paddle, state_bricks, game_status, win
