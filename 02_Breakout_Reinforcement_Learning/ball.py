class Ball:
    def __init__(self, x_pos: int, y_pos: int, x_speed: int, y_speed: int, grid_height: int, grid_width: int):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.BRICK_REWARD = 1000
        self.ALL_BRICKS_REWARD = 10000
        self.LOST_REWARD = -10000
        self.RESET_REWARD = -5000
        self.TIME_LIMIT = 250
        self.time = 0

    def update(self, paddle, bricks):
        self.time += 1
        win = False
        game_status = True  # Game still in progress
        reward = -1  # Basic reward per step to incentivize faster completions
        moved = False  # flag indicating that the ball was finally moved after not encountering any further collisions
        reset = False # reset after losing the ball

        while not moved:
            collided = False
            # Check collision with paddle to determine speed
            if self.y_pos == 1 and self.y_speed == -1:
                paddle.collision(
                    self)  # Tests for collision with the paddle, if collision occurs speed is set accordingly

            # Now we have three cases depending on the absolute horizontal speed of the ball
            if self.x_speed == 0:  # We only have vertical movement, therefore no need to check diagonals
                if self.y_pos + self.y_speed == self.grid_height:  # Collision with top of grid
                    self.y_speed = -1
                    collided = True
                if not collided and bricks.layout[
                    self.y_pos + self.y_speed, self.x_pos // 3] == 1:  # Vertical collision with brick
                    bricks.count_remaining_bricks -= 1
                    bricks.layout[self.y_pos + self.y_speed, self.x_pos // 3] = 0
                    self.y_speed *= -1
                    reward += self.BRICK_REWARD
                    collided = True
            elif abs(self.x_speed) == 1:  # Diagonal movement with 45-degree angle
                # First check collisions with top and side of grid
                if self.y_pos + self.y_speed == self.grid_height:  # Collision with top of grid
                    self.y_speed = -1
                    collided = True
                if -1 == self.x_pos + self.x_speed or self.x_pos + self.x_speed == self.grid_width:  # Collision with side of grid
                    self.x_speed *= -1
                    collided = True

                if not collided and bricks.layout[
                    self.y_pos + self.y_speed, self.x_pos // 3] == 1:  # Vertical collision with brick
                    bricks.count_remaining_bricks -= 1
                    bricks.layout[self.y_pos + self.y_speed, self.x_pos // 3] = 0
                    self.y_speed *= -1
                    reward += self.BRICK_REWARD
                    collided = True

                if not collided and bricks.layout[
                    self.y_pos, (self.x_pos + self.x_speed) // 3] == 1:  # horizontal collision with brick
                    bricks.count_remaining_bricks -= 1
                    bricks.layout[self.y_pos, (self.x_pos + self.x_speed) // 3] = 0
                    self.x_speed *= -1
                    reward += self.BRICK_REWARD
                    collided = True

                if not collided and bricks.layout[
                    self.y_pos + self.y_speed, (self.x_pos + self.x_speed) // 3] == 1:  # Diagonal collision if no other collision occurred before
                    bricks.count_remaining_bricks -= 1
                    bricks.layout[self.y_pos + self.y_speed, (self.x_pos + self.x_speed) // 3] = 0
                    self.x_speed *= -1
                    self.y_speed *= -1
                    reward += self.BRICK_REWARD
                    collided = True
            else:  # Diagonal movement with 30-degree angle, absolute value of horizontal speed is 2
                # At first we check for collisions one block away, this code is very similar to above, but the diagonal collision behaves differently
                # First check collisions with top and side of grid
                if self.y_pos + self.y_speed == self.grid_height:  # Collision with top of grid
                    self.y_speed = -1
                    collided = True
                if -1 == self.x_pos + self.x_speed//2 or self.x_pos + self.x_speed//2 == self.grid_width:  # Collision with side of grid
                    self.x_speed *= -1
                    collided = True

                try:
                    if not collided and bricks.layout[
                        self.y_pos + self.y_speed, self.x_pos // 3] == 1:  # Vertical collision with brick
                        bricks.count_remaining_bricks -= 1
                        bricks.layout[self.y_pos + self.y_speed, self.x_pos // 3] = 0
                        self.y_speed *= -1
                        reward += self.BRICK_REWARD
                        collided = True
                except IndexError:
                    pass

                if not collided and bricks.layout[
                    self.y_pos, (self.x_pos + self.x_speed//2) // 3] == 1:  # horizontal collision with brick
                    bricks.count_remaining_bricks -= 1
                    bricks.layout[self.y_pos, (self.x_pos + self.x_speed//2) // 3] = 0
                    self.x_speed *= -1
                    reward += self.BRICK_REWARD
                    collided = True

                if not collided and bricks.layout[
                    self.y_pos + self.y_speed, (
                                                       self.x_pos + self.x_speed//2) // 3] == 1:  # Diagonal collision if no other collision occurred before
                    bricks.count_remaining_bricks -= 1
                    bricks.layout[self.y_pos + self.y_speed, (self.x_pos + self.x_speed//2) // 3] = 0
                    # self.x_speed *= -1 We do not change x speed because of the slight angle we get when imagining a round ball
                    self.y_speed *= -1
                    reward += self.BRICK_REWARD
                    collided = True

                # Now we check for the collision two blocks away, we cannot get collision with top, because this has already been checked
                if not collided and (self.x_pos + self.x_speed == -1 or self.x_pos + self.x_speed == self.grid_width):
                    self.x_pos += self.x_speed
                    self.x_speed *= -1
                    collided = True
                if not collided and (bricks.layout[
                    self.y_pos, (
                                                       self.x_pos + self.x_speed) // 3] == 1 or bricks.layout[
                    self.y_pos + self.y_speed, (
                                                       self.x_pos + self.x_speed) // 3] == 1):
                    tmp = bricks.layout[
                    self.y_pos + self.y_speed, (
                                                       self.x_pos + self.x_speed) // 3] + bricks.layout[
                    self.y_pos + self.y_speed, (
                                                       self.x_pos + self.x_speed) // 3]

                    bricks.layout[
                        self.y_pos, (
                                self.x_pos + self.x_speed) // 3] = 0

                    bricks.layout[
                        self.y_pos + self.y_speed, (
                                self.x_pos + self.x_speed) // 3] = 0

                    bricks.count_remaining_bricks -= tmp
                    reward += tmp * self.BRICK_REWARD
                    self.x_pos += self.x_speed
                    self.x_speed *= -1
                    collided = True


            if not collided:
                moved = True
                self.x_pos += self.x_speed
                self.y_pos += self.y_speed

        if bricks.layout.sum() == 0: # Won the game
            reward += self.ALL_BRICKS_REWARD
            game_status = False
            win = True

        if self.y_pos == 0: # Lost the game
            reset = True
            reward += self.RESET_REWARD

        if  self.TIME_LIMIT <= self.time:
            game_status = False
            win = win if win else False
            reward += self.LOST_REWARD

        return reset, reward, game_status, win
