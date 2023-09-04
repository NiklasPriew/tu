class Paddle:
    def __init__(self, x_pos: int, x_speed: int, grid_width: int):
        self.x_pos = x_pos
        self.x_speed = x_speed
        self.grid_width = grid_width

    def update(self, action: int):
        # When x_speed * action is 2 then either speed is -2 and action is left or speed is 2 and action is right,
        # both would create too high speed
        if self.x_speed * action != 2:
            self.x_speed += action

        # Collision with walls
        if self.x_pos + self.x_speed < 0:
            self.x_pos = 0
        elif self.x_pos + self.x_speed >= self.grid_width-4:
            self.x_pos = self.grid_width - 5
        else:
            self.x_pos += self.x_speed

    def collision(self, ball):
        tmp_diff = ball.x_pos - self.x_pos
        if 0 <= tmp_diff < 5:
            ball.y_speed = 1
            ball.x_speed = -2 + tmp_diff
