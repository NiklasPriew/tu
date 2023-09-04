import numpy as np
import pygame

from monte_carlo_control import MonteCarloControl, create_layout


def main():
    # play
    grid_width = 15
    grid_height = 10
    grid_type = 'double'
    epsilon = 0.10
    gamma = 0.95
    fn_model = f'models/breakout_model_e{epsilon:.2f}_g{gamma:.2f}_{grid_height}x{grid_width}-{grid_type}.pickle'
    line_layout = create_layout(grid_height, grid_width, grid_type)
    mc = MonteCarloControl(line_layout, grid_height, grid_width, grid_type, epsilon, gamma)
    mc.load_model(fn_model)
    episode_memory, won, n_steps = mc.play(mc.select_action_soft, ball_x_speed_start=-1)

    # set up pygame
    white = (255, 255, 255)
    red = (255, 0, 0)
    blue = (0, 0, 255)
    green = (0, 255, 0)
    zoom = 50
    efps = 10 # episodes/frames per second
    screen_width = grid_width * zoom
    screen_height = grid_height * zoom
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Breakout.AI")

    # replay
    ei = 0 # episode/frame number
    total_reward = 0
    for e in episode_memory:
        ei += 1
        (state, action, reward) = e
        s = np.frombuffer(state, dtype=np.int8)
        ball_x_pos = s[:4][0]
        assert 0 <=  ball_x_pos <= 15
        ball_y_pos = s[:4][1]
        assert  0 <= ball_y_pos <= 10
        ball_x_speed = s[:4][2]
        assert -2 <= ball_x_speed <= 2
        ball_y_speed = s[:4][3]
        assert -1 <= ball_y_speed <= 1
        paddle_x_pos = s[4:6][0]
        assert 0 <= paddle_x_pos <= 15-5
        paddle_x_speed = s[4:6][1]
        assert -2 <= paddle_x_speed <= 2
        state_bricks = s[6:].reshape(grid_height, grid_width // 3)
        total_reward += reward
        screen.fill(white)
        pygame.draw.rect(screen, red, (ball_x_pos*zoom, screen_height-ball_y_pos*zoom-zoom, 1*zoom, zoom))  # ball
        pygame.draw.rect(screen, blue, (paddle_x_pos*zoom, screen_height-zoom, 5*zoom, zoom))  # paddle
        print(f'episode/frame #{ei}\twon = {won}')
        print(f'paddle_x_pos = {paddle_x_pos}\tpaddle_x_speed = {paddle_x_speed}')
        print(f'ball_x_pos = {ball_x_pos}\t\tball_y_pos = {ball_y_pos}')
        print(f'ball_x_speed = {ball_x_speed}\tball_y_speed = {ball_y_speed}')
        print(f'remaining bricks = {state_bricks.sum()}')
        print(f'action = {action}')
        print(f'reward = {reward}')
        print(f'total_reward = {total_reward}')
        #print(state_bricks)
        print()
        for x in range(state_bricks.shape[1]):
            for y in range(state_bricks.shape[0]):
                if state_bricks[y,x] == 1:
                    pygame.draw.rect(screen, green, (x*3*zoom, screen_height - y*zoom-zoom, 3 * zoom, zoom))  # brick
        pygame.display.update()
        pygame.time.wait(1000//efps)


if __name__ == "__main__":
    main()
