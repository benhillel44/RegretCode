import concurrent.futures
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from system import GridSystem
from visualizer import GridVisualizer
from policy import LookAheadPolicy, KRegretPolicy
from constants import *



def run(x_0, N_steps, _look_ahead_depth, load_memory=False, random_grid=False, visualise=False):
    start_time = datetime.now()
    gs = GridSystem()
    if random_grid:
        # gs.set_random_grid()
        gs.set_version2()
    look_ahead_policy = LookAheadPolicy(system=gs, look_ahead_depth=_look_ahead_depth, load_mem=load_memory)
    reg_policy = KRegretPolicy(system=gs, N=N_steps, look_ahead_depth=_look_ahead_depth-1, load_mem=load_memory)
    _vals, V = reg_policy.step(x_0, x_0, [], t=N_steps)
    reg_u, reg_w = _vals
    reg_cost = gs.J(x_0, reg_u, reg_w)
    look_ahead_u, look_ahead_cost = look_ahead_policy.step(x_0, reg_w)
    end_time = datetime.now()
    simulation_time = end_time - start_time
    print(f">>> Run time = {simulation_time} with parameters: \n>>> N={N_steps} | k={_look_ahead_depth} | sys id={gs.getId()}")
    print(f"adversary noise = {reg_w}")
    print(f"reg actions = {reg_u}")
    print(f"reg total cost = {reg_cost}")
    print(f"optimal actions = {look_ahead_u}")
    print(f"optimal total cost = {look_ahead_cost}")
    print(f"regret = {V}")

    if visualise:
        g_visualizer = GridVisualizer(GridSystem.COST_MAT, start_pos=x_0)
        g_visualizer.add_movement_path([reg_w[i] + reg_u[i] for i in range(len(reg_w))], color='red')
        g_visualizer.add_movement_path([reg_w[i] + look_ahead_u[i] for i in range(len(look_ahead_u))], color='green')
        g_visualizer.visualize()

    run_dat = {
        "system_id": gs.getId(),
        "memory_loaded": load_memory,
        "N": N_steps,
        "k": _look_ahead_depth,
        "x_0": x_0,
        "adversary_noise": reg_w,
        "reg_actions": reg_u,
        "look_ahead_actions": look_ahead_u,
        "reg_cost": reg_cost,
        "look_ahead_cost": look_ahead_cost,
        "regret": V,
        "simulation_time": simulation_time
    }
    return run_dat

def plot_results():
    # plot the regret vs N for each k using plolty
    df = pd.read_csv(RESULT_FILE_NAME)
    # sort the data frame by N
    df = df.sort_values(by="N")
    # for each system id, plot the regret vs N for each k
    for system_id in df["system_id"].unique():
        df_sys = df[df["system_id"] == system_id]
        fig = px.line(df_sys, x="N", y="regret", color="k", markers=True,
                      title="Regret vs N for Each Value of k",
                      labels={"N": "N (Horizon Length)", "regret": "Regret", "k": "Look-ahead Depth (k)"})
        fig.show()

def run_simulation(N_range, k, x_0, load_memory, random_grid, visualise=False):
    """Function to execute a single simulation"""
    print(f"Running for N range = {N_range}, k = {k} on PID {os.getpid()}")

    if type(N_range) == int:
        N_range = [N_range]
    res = pd.DataFrame()
    for N in N_range:
        run_dat = run(x_0, N, k, load_memory, random_grid=random_grid)
        res = res.append(run_dat, ignore_index=True)

    return res

def main():
    x_0 = np.array((3, 3))
    N_range = range(1, 10)
    k_range = range(1, 5)
    load_memory = True
    random_grid = True
    visualise = False
    # load the results from the previous run, if they exist
    try:
        results = pd.read_csv(RESULT_FILE_NAME)
    except FileNotFoundError:
        results = pd.DataFrame()


    # Use multiprocessing to parallelize the simulations
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_simulation, N_range, k, x_0, load_memory, random_grid, visualise): k for k in k_range}
        # Collect results
        results = pd.concat([results]+[future.result() for future in concurrent.futures.as_completed(futures)])

    # Save to CSV once at the end
    results.to_csv(RESULT_FILE_NAME, index=False)


if __name__ == "__main__":
    main()
    plot_results()
