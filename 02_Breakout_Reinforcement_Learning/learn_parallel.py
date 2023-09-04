import multiprocessing as mp
import os
import time

import numpy as np

from monte_carlo_control import MonteCarloControl, create_layout


def learn(pi, grid_height, grid_width, grid_type, epsilon, gamma, runs, n_episodes_learn, n_epsiodes_play):
    line_layout = create_layout(grid_height, grid_width, grid_type)
    mc = MonteCarloControl(line_layout, grid_height, grid_width, grid_type, epsilon, gamma)
    rts = []  # runtimes

    sr_best_model = 0.00  # success rate best model
    fn_prev_model = None
    for i in range(1, runs+1):
        t0 = time.perf_counter()
        mc.learn(n_episodes_learn, greedy=False)  # learn
        wins, steps = mc.learn(n_epsiodes_play, greedy=True)  # play
        sr_current_model = sum(wins) / n_epsiodes_play  # success rate current model
        if sr_best_model < sr_current_model:
            fn = f'models/breakout_model_e{epsilon:.2f}_g{gamma:.2f}_{grid_height}x{grid_width}-{grid_type}_w{sr_current_model:.2f}.pickle'  # filename
            mc.save_model(fn)
            if 1 < i and os.path.isfile(fn_prev_model):
                os.remove(fn_prev_model)
            fn_prev_model = fn
            sr_best_model = sr_current_model
            print(f'p{pi} ... best model yet ({sr_current_model:.2f} saved at {fn}, deleted {fn_prev_model}')
        t1 = time.perf_counter()
        rts.append(t1 - t0)
        print(f'p{pi}(e={epsilon:.2f},g={gamma:.2f})\trun {i + 1}\twins: {sum(wins) / n_epsiodes_play:.2f}\ttime: {(t1 - t0) * 1000.0:.1f}ms')
    print(f'p{pi}(e={epsilon:.2f},g={gamma:.2f}) median runtime: {np.median(rts) * 1000.0:.3f}ms after {runs} runs')


if __name__ == '__main__':
    grid_height = 10  # (0 <= grid_height <= 10)
    grid_width = 15  # (0 <= grid_width <= 15) and (grid_width % 3 == 0)
    grid_type = 'double'  # one of ['single', 'double', 'bridge', 'towers']
    epsilons = [0.1]
    gammas = [0.95]
    n_runs = 100
    n_episodes_learn = 1000
    n_episodes_play = 100

    assert len(epsilons) == len(gammas)
    n_processes = len(epsilons)

    for pi in range(n_processes):
        p = mp.Process(target=learn,
                       args=(pi, grid_height, grid_width, grid_type, epsilons[pi], gammas[pi],
                             n_runs, n_episodes_learn, n_episodes_play)) \
            .start()
