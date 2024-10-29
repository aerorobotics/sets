
# standard
import numpy as np 
import time as timer
import os
import itertools as it
import multiprocessing as mp
import tqdm 
import glob 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# custom
import sys 
sys.path.append("../src")
import plotter 
from util import util 
from build.bindings import get_mdp, get_dots_mdp, get_uct, get_uct2, RNG, UCT, MDP, \
    run_uct, run_uct2, Trajectory, rollout_action_sequence, Tree
# from learning.feedforward import Feedforward


plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 1.0


def _rollout(args):
    return rollout(*args)


def rollout(process_count, config_dict, seed, parallel_on, initial_state=None):

    np.set_printoptions(precision=3)

    # set seeds 
    rng = RNG()
    rng.set_seed(seed)

    config_name = config_dict["config_name"]
    config_path = config_dict["config_path"]

    print("config_name: {}".format(config_name))
    
    # ground MDP 
    ground_mdp = get_mdp(config_dict["ground_mdp_name"], config_path)

    # print('np.array(config_dict["dots_rho"])',np.array(config_dict["dots_rho"]))
    # exit()

    # get dots 
    dots_mdp = get_dots_mdp()
    dots_mdp.set_param(ground_mdp, 
        config_dict["dots_expansion_mode"], 
        config_dict["dots_initialize_mode"],
        config_dict["dots_special_actions"], 
        config_dict["dots_num_branches"], 
        config_dict["dots_decision_making_horizon"], 
        config_dict["dots_dynamics_horizon"],
        config_dict["dots_spectral_branches_mode"], 
        config_dict["dots_control_mode"], 
        config_dict["dots_scale_mode"], 
        config_dict["dots_modal_damping_mode"], 
        config_dict["dots_modal_damping_gains"], 
        np.diag(np.array(config_dict["dots_rho"])), 
        config_dict["dots_greedy_gain"], 
        config_dict["dots_greedy_rate"], 
        config_dict["dots_greedy_min_dist"], 
        config_dict["dots_baseline_mode_on"],
        config_dict["dots_num_discrete_actions"],  
        config_dict["dots_verbose"])
    
    # get uct 
    if config_dict["uct_mode"] == "cbds":
        uct = get_uct() 
        # uct.set_param(
        #     config_dict["uct_N"],         
        #     config_dict["uct_max_depth"],         
        #     config_dict["uct_wct"],         
        #     config_dict["uct_c"],         
        #     config_dict["uct_export_topology"],         
        #     config_dict["uct_export_node_states"],         
        #     config_dict["uct_export_trajs"],         
        #     config_dict["uct_export_cbdsbds"],
        #     config_dict["uct_export_tree_statistics"],
        #     config_dict["uct_heuristic_mode"],         
        #     config_dict["uct_tree_exploration"],         
        #     config_dict["uct_verbose"],         
        #     )
        uct.set_param(
            config_dict["uct_N"],         
            config_dict["uct_max_depth"],         
            config_dict["uct_wct"],         
            config_dict["uct_c"],         
            config_dict["uct_export_topology"],         
            config_dict["uct_export_node_states"],         
            config_dict["uct_export_trajs"],         
            config_dict["uct_export_cbdsbds"],
            config_dict["uct_export_tree_statistics"],
            config_dict["uct_heuristic_mode"],         
            config_dict["uct_tree_exploration"],         
            config_dict["uct_downsample_traj_on"],         
            config_dict["uct_verbose"],         
            )
    elif config_dict["uct_mode"] == "no_cbds":
        uct = get_uct2() 
        uct.set_param(
            config_dict["uct_N"],         
            config_dict["uct_max_depth"],         
            config_dict["uct_wct"],         
            config_dict["uct_c"],         
            config_dict["uct_export_topology"],         
            config_dict["uct_export_node_states"],         
            config_dict["uct_export_trajs"],         
            config_dict["uct_export_cbdsbds"],
            config_dict["uct_export_tree_statistics"],
            config_dict["uct_heuristic_mode"],         
            config_dict["uct_tree_exploration"],         
            config_dict["uct_downsample_traj_on"],         
            config_dict["uct_verbose"],         
            )

    total_start_time = timer.time()

    # run once 
    if initial_state is None:
        initial_state = np.array(ground_mdp.initial_state())
    
    result = {
        "config_name" : config_name,
        "config_dict" : config_dict,
        "config_path" : config_path,
        "process_count" : process_count,
        "uct_trees" : [], 
        "uct_xss" : [], 
        "uct_uss" : [], 
        "uct_rss" : [], 
        "uct_vss" : [],
        "final_uct_xs": [],
        "final_uct_us": [],
        "final_uct_rs": [],
        "rollout_xs" : [], 
        "rollout_us" : [], 
        "rollout_rs" : [], 
        "success" : False
    }

    for trajs_fn in glob.glob(f"../data/test_{result['config_name']}_trajs_pc{process_count}_ii*"):
        if os.path.exists(trajs_fn):
            os.remove(trajs_fn)

    mode = config_dict["rollout_mode"]
    ii_trajs = 0
    print("mode: ", mode)

    # run!
    if mode == "uct":

        curr_state = initial_state

        # update environment 
        ground_mdp.clear_obstacles() 
        [ground_mdp.add_obstacle(obstacle) for obstacle in util.get_obstacles(config_dict, curr_state[ground_mdp.timestep_idx()])]
        ground_mdp.clear_thermals() 
        [ground_mdp.add_thermal(thermal_bound, thermal_force_moment) for thermal_bound, thermal_force_moment in util.get_thermals(config_dict, curr_state[ground_mdp.timestep_idx()])]

        print("k/H: {}/{}, curr_state: {}".format(curr_state[ground_mdp.timestep_idx()], ground_mdp.H(), curr_state))
        ground_mdp.set_dt(config_dict["uct_dt"])
        start_time = timer.time()
        if config_dict["uct_mode"] == "cbds":
            cpp_result_uct = run_uct(dots_mdp, uct, curr_state, rng)
        elif config_dict["uct_mode"] == "no_cbds":
            cpp_result_uct = run_uct2(dots_mdp, uct, curr_state, rng)
        print("seed: {}, wct uct: {} n: {}".format(seed, np.round(timer.time() - start_time, 3), len(cpp_result_uct.vs)))

        result["uct_trees"].append(cpp_result_uct.tree)
        result["uct_xss"].append(cpp_result_uct.planned_traj.xs)
        result["uct_uss"].append(cpp_result_uct.planned_traj.us)
        result["uct_rss"].append(cpp_result_uct.planned_traj.rs)
        result["uct_vss"].append(cpp_result_uct.vs)
        result["rollout_xs"] = cpp_result_uct.planned_traj.xs
        result["rollout_us"] = cpp_result_uct.planned_traj.us
        result["rollout_rs"] = cpp_result_uct.planned_traj.rs
        result["final_uct_xs"] = cpp_result_uct.planned_traj.xs
        result["final_uct_us"] = cpp_result_uct.planned_traj.us
        result["final_uct_rs"] = cpp_result_uct.planned_traj.rs
        result["final_scp_xs"] = cpp_result_uct.planned_traj.xs
        result["final_scp_us"] = cpp_result_uct.planned_traj.us
        result["final_scp_rs"] = cpp_result_uct.planned_traj.rs

    elif mode == "uct-mpc":
        
        curr_state = initial_state
        if config_dict["uct_downsample_traj_on"]:
            mpc_horizon = config_dict["uct_mpc_depth"]
        else:
            mpc_horizon = config_dict["dots_decision_making_horizon"] * config_dict["uct_mpc_depth"]


        while curr_state[ground_mdp.timestep_idx()] < ground_mdp.H():

            empty_action = np.zeros((4,1))
            print("k/H: {}/{}, curr_state: {}".format(curr_state[ground_mdp.timestep_idx()], ground_mdp.H(), curr_state))
            print("ground_mdp.R_verbose(curr_state, empty_action, True):", ground_mdp.R_verbose(curr_state, empty_action, True))

            # update environment 
            ground_mdp.clear_obstacles() 
            [ground_mdp.add_obstacle(obstacle) for obstacle in util.get_obstacles(config_dict, curr_state[ground_mdp.timestep_idx()])]
            ground_mdp.clear_thermals() 
            [ground_mdp.add_thermal(thermal_bound, thermal_force_moment) for thermal_bound, thermal_force_moment in util.get_thermals(config_dict, curr_state[ground_mdp.timestep_idx()])]

            # uct
            ground_mdp.set_dt(config_dict["uct_dt"])
            start_time = timer.time()
            if config_dict["uct_mode"] == "cbds":
                cpp_result_uct = run_uct(dots_mdp, uct, curr_state, rng)
            elif config_dict["uct_mode"] == "no_cbds":
                cpp_result_uct = run_uct2(dots_mdp, uct, curr_state, rng)
            print("seed: {}, wct uct: {} n: {}".format(seed, np.round(timer.time() - start_time, 3), len(cpp_result_uct.vs)))

            if cpp_result_uct.success:
                xs = np.array(cpp_result_uct.planned_traj.xs).tolist()
                us = np.array(cpp_result_uct.planned_traj.us).tolist()
                rs = np.array(cpp_result_uct.planned_traj.rs).tolist()
                result["uct_trees"].append(cpp_result_uct.tree)
                result["uct_xss"].append(xs)
                result["uct_uss"].append(us)
                result["uct_rss"].append(rs) 
                result["uct_vss"].append(cpp_result_uct.vs)
                result["final_uct_xs"].extend(xs[0:mpc_horizon])
                result["final_uct_us"].extend(us[0:mpc_horizon])
                result["final_uct_rs"].extend(rs[0:mpc_horizon])
                print("uct_ave_reward: {}".format(np.mean(rs[0:mpc_horizon])))
            else: 
                print("uct failed @ {} for seed: {} and config_name: {}".format(np.array(curr_state), seed, config_name))
                break

            curr_state = result["final_uct_xs"][-1]

            if config_dict["uct_export_trajs"] and len(cpp_result_uct.tree.trajs) != 0:
                from value_convergence import sparsify_trajs
                trajs_cpp = cpp_result_uct.tree.trajs
                # max_length = dots_H * tree_depth_H
                max_length = max([len(traj.xs) for traj in trajs_cpp])

                tree_memory_bytes = np.ones((1,)).itemsize * len(trajs_cpp) * max_length * ground_mdp.state_dim() 
                bens_laptop_mem = 30.0 * 10e9 
                print("tree_memory_bytes", tree_memory_bytes)
                if tree_memory_bytes < (bens_laptop_mem / 5.0):
                    trajs_np = np.nan * np.ones((len(trajs_cpp), max_length, ground_mdp.state_dim()))
                    for jj, traj in enumerate(trajs_cpp):
                        xs = np.array(traj.xs) # (timesteps, n)
                        if xs.shape[0] != 0:
                            trajs_np[jj,0:xs.shape[0],:] = xs
                    trajs_np = sparsify_trajs(trajs_np, 1000)
                    fn = "../data/test_{}_trajs_pc{}_ii{}".format(result["config_name"], result["process_count"], ii_trajs)
                    util.save_npy(trajs_np, fn)
                    ii_trajs += 1

        result["final_scp_xs"] = result["final_uct_xs"]
        result["final_scp_us"] = result["final_uct_us"]
        result["final_scp_rs"] = result["final_uct_rs"]
        result["rollout_xs"] = result["final_uct_xs"]
        result["rollout_us"] = result["final_uct_us"]
        result["rollout_rs"] = result["final_uct_rs"]

    if len(result["uct_trees"]) > 0:
        result_no_trees = result.copy()
        result_no_trees.pop("uct_trees")
        util.save_pickle(result_no_trees, "../data/rollout_result_{}_{}.pkl".format(config_name, process_count))
        if parallel_on: 
            return result_no_trees
    else:
        util.save_pickle(result, "../data/rollout_result_{}_{}.pkl".format(config_name, process_count))

    result["success"] = True
    # print_result(result)
    return result


def print_result(result):
    # print('result["config_dict"]', result["config_dict"])
    print('result["uct_trees"]', len(result["uct_trees"]))
    print('result["uct_xss"]', np.array(result["uct_xss"]).shape)
    print('result["uct_uss"]', np.array(result["uct_uss"]).shape)
    print('result["uct_rss"]', np.array(result["uct_rss"]).shape)
    print('result["uct_vss"]', np.array(result["uct_vss"]).shape)
    print('result["final_uct_xs"]', np.array(result["final_uct_xs"]).shape)
    print('result["final_uct_us"]', np.array(result["final_uct_us"]).shape)
    print('result["final_uct_rs"]', np.array(result["final_uct_rs"]).shape)
    print('result["rollout_xs"]', np.array(result["rollout_xs"]).shape)
    print('result["rollout_us"]', np.array(result["rollout_us"]).shape)
    print('result["rollout_rs"]', np.array(result["rollout_rs"]).shape)
    


def plot_result(result):
    
    render_branchdatas_on = False 
    plot_trajs_over_time_branchdata_on = False
    plot_trajs_over_time_result_on = False
    second_plot_on = False
    max_num_branchdatas = 3
    plot_tree_statistics = False
    plot_movie_on = False


    render_result_on = True
    plot_tree_trajs_on = True
    plot_trajectory_on = True

    if render_branchdatas_on:
        print("render_branchdata...")
        if "uct_trees" in result.keys():
            for tree in util.subsample_list(result["uct_trees"], max_num_branchdatas):
                render_branchdata(result["config_dict"], tree)

    if plot_trajs_over_time_branchdata_on:
        if "uct_trees" in result.keys():
            for tree in util.subsample_list(result["uct_trees"], max_num_branchdatas):
                print("plot_trajs_over_time_branchdata...")
                plot_trajs_over_time_branchdata(result["config_dict"], tree)        

    if render_result_on: 
        print("render_result...")
        render_result(result)

    # if len(result["uct_trees"]) > 0 and len(result["uct_trees"][0].trajs) > 0:
    #     for 

    if plot_trajs_over_time_result_on:
        print("plot_trajs_over_time_result...")
        plot_trajs_over_time_result(result)

    if second_plot_on and result["success"]:
        print("second_plot...")
        second_plot(result)

    if plot_tree_statistics:
        if "uct_trees" in result.keys():
            print("render_tree_statistics...")
            render_tree_statistics(result)
            # render_tree_statistics_only_vs(result)

    if plot_movie_on: 
        render_movie(result)

    if plot_tree_trajs_on:
        for trajs_fn in glob.glob(f"../data/test_{result['config_name']}_trajs_pc*_ii*"):
            trajs = util.load_npy(trajs_fn)
            from value_convergence import render_tree_glider
            ground_mdp = get_mdp(result["config_dict"]["ground_mdp_name"], result["config_path"])
            render_tree_glider(ground_mdp, trajs, result["config_dict"], 
                result["config_dict"]["dots_decision_making_horizon"], None, None, "black", 0.5)

    if plot_trajectory_on:
        xs = result["rollout_xs"]
        xs_np = np.array(xs) # (T, n)
        us = result["rollout_us"]
        us_np = np.array(us) # (T, m)

        state_lims = np.array(result["config_dict"]["ground_mdp_X"])
        state_lims = state_lims.reshape((state_lims.shape[0] // 2, 2), order="F")

        control_lims = np.array(result["config_dict"]["ground_mdp_U"])
        control_lims = control_lims.reshape((control_lims.shape[0] // 2, 2), order="F")

        for ii, state_label in enumerate(result["config_dict"]["ground_mdp_state_labels"]):
            fig, ax = plotter.make_fig()
            ax.plot(xs_np[:,ii])
            ax.set_title(state_label)
            ax.set_ylim([state_lims[ii,0], state_lims[ii,1]])

        for ii, control_label in enumerate(["delta_e", "delta_r", "delta_a"]):
            fig, ax = plotter.make_fig()
            ax.plot(us_np[:,ii])
            ax.set_title(control_label)
            ax.set_ylim([control_lims[ii,0], control_lims[ii,1]])




def render_movie(result):
    xs = result["rollout_xs"]
    xs_np = np.array(xs)
    labels_to_plot = ["rollout"]
    colors = plotter.get_n_colors(len(labels_to_plot))
    num_frames = 50
    idxs = util.subsample_list(list(range(len(xs))), num_frames)

    for idx in idxs:
        print("idx/idxs[-1]: {}/{}".format(idx,idxs[-1]))
        render_fig, render_ax = plotter.make_3d_fig()    
        xs_np_idx = xs_np[0:idx,:]
        if xs_np_idx.shape[0] == 0: continue
        # render initial state
        x0 = xs_np_idx[0,np.newaxis,:]
        plotter.render_xs_sixdofaircraft_game(x0, result["config_dict"]["ground_mdp_name"], result["config_dict"], 
            obstacles_on=True, color=colors[0], alpha=0.5, fig=render_fig, ax=render_ax)
        # render final state
        xf = xs_np_idx[-1,np.newaxis,:]
        plotter.render_xs_sixdofaircraft_game(xf, result["config_dict"]["ground_mdp_name"], result["config_dict"], 
            color=colors[0], alpha=0.5, fig=render_fig, ax=render_ax)
        # plot trajectory 
        render_ax.plot(xs_np[:,0], xs_np[:,1], -1 * xs_np[:,2], color=colors[0])

        tree_idx = int(idx / (result["config_dict"]["dots_decision_making_horizon"] * result["config_dict"]["uct_mpc_depth"]))
        if tree_idx < len(result["uct_trees"]):
            render_branchdata(result["config_dict"], result["uct_trees"][tree_idx])

    # render_ax.legend()

def render_tree_statistics(result):
    for kk, tree in enumerate(result["uct_trees"]):
        total_visit_counts_per_depth, visit_counts_per_depth = extract_from_tree_statistics(tree)
        fig = plt.figure()
        gs = GridSpec(max((2,len(total_visit_counts_per_depth))), 2)
        depth_axs = []
        for ii in range(len(total_visit_counts_per_depth)):
            depth_axs.append(fig.add_subplot(gs[ii,0]))
        survival_ax = fig.add_subplot(gs[0,1])
        v_over_n_ax = fig.add_subplot(gs[1:,1])
        for depth in range(len(total_visit_counts_per_depth)):
            survival_ax.bar(depth, total_visit_counts_per_depth[depth] / result["config_dict"]["uct_N"], alpha=0.5, label="depth {}".format(depth))
            survival_ax.set_title("Percent survival per depth")
        for depth in range(len(total_visit_counts_per_depth)):
            visit_counts_per_depth[depth].sort()
            if visit_counts_per_depth[depth] is not None:
                visit_counts_per_depth[depth].reverse()
            depth_axs[depth].bar(np.arange(len(visit_counts_per_depth[depth])), visit_counts_per_depth[depth], alpha=0.5, label="depth {}".format(depth))
        depth_axs[0].set_title("Sorted visit counts per node")
        vs = np.array(result["uct_vss"][kk])
        ns = np.arange(len(vs))
        v_over_n_ax.plot(ns, vs, alpha=1.0, label="v/n")
    pass

def render_tree_statistics_only_vs(result):
    fig, ax = plotter.make_fig()
    colors = plotter.get_n_colors(len(result["uct_trees"]))
    for kk, tree in enumerate(result["uct_trees"]):
        vs = np.array(result["uct_vss"][kk])
        ns = np.arange(len(vs))
        ax.plot(ns, vs, alpha=1.0, color=colors[kk])
    ax.set_title("v/n")


def extract_from_tree_statistics(tree):
    total_visit_counts_per_depth = [0]
    visit_counts_per_depth = [[] for _ in range(1)]
    for max_value, num_visits, depth in tree.node_visit_statistics:
        while depth >= len(total_visit_counts_per_depth):
            total_visit_counts_per_depth.append(0)
            visit_counts_per_depth.append([])
        if num_visits == 0:
            continue
        total_visit_counts_per_depth[depth] += num_visits
        visit_counts_per_depth[depth].append(num_visits)
    return total_visit_counts_per_depth, visit_counts_per_depth


def render_result(result):
    render_fig, render_ax = plotter.make_3d_fig()
    if result["config_dict"]["rollout_mode"] == "uct":
        # labels_to_plot = ["final_uct", "rollout"]
        labels_to_plot = ["rollout"]
    elif result["config_dict"]["rollout_mode"] == "uct-mpc":
        # labels_to_plot = ["final_uct", "rollout"]
        labels_to_plot = ["rollout"]
    elif result["config_dict"]["rollout_mode"] == "uct-scp-mpc":
        labels_to_plot = ["rollout", "final_uct", "final_scp"]
    elif result["config_dict"]["rollout_mode"] == "uct-mpc-scp":
        labels_to_plot = ["rollout", "final_uct", "final_scp"]
    colors = plotter.get_n_colors(len(labels_to_plot))
    for ii, label in enumerate(labels_to_plot):
        xs = result[label+"_xs"]
        xs_np = np.array(xs)
        if len(xs) == 0:
            continue
        # # render initial state
        x0 = xs_np[0,np.newaxis,:]
        plotter.render_xs_sixdofaircraft_game(x0, result["config_dict"]["ground_mdp_name"], result["config_dict"], 
            obstacles_on=True, thermals_on=True, color=colors[ii], alpha=0.5, fig=render_fig, ax=render_ax)
        # render final state
        xf = xs_np[-1,np.newaxis,:]
        plotter.render_xs_sixdofaircraft_game(xf, result["config_dict"]["ground_mdp_name"], result["config_dict"], 
            color=colors[ii], alpha=0.5, fig=render_fig, ax=render_ax)
        num_robots = 6
        for kk in np.linspace(0, xs_np.shape[0]-1, num_robots, dtype=int):
            plotter.render_xs_sixdofaircraft_game(xs_np[kk,np.newaxis,:], result["config_dict"]["ground_mdp_name"], result["config_dict"], 
                color=colors[ii], alpha=0.5, fig=render_fig, ax=render_ax)
        # plot trajectory 
        render_ax.plot(xs_np[:,0], xs_np[:,1], -1 * xs_np[:,2], alpha=0.5, color=colors[ii], label=label)
    # render_ax.legend()
    render_ax.axis("off")

    fig, ax = plotter.make_fig()
    xs_np = np.array(result["rollout_xs"])
    mass = 11.0 # kg 
    gravity = 9.8 # kg m / s^2
    KE = lambda x: 1 /2 * mass * np.linalg.norm(x[3:6]) ** 2.0
    PE = lambda x: mass * gravity * (-1 * x[2])
    kes = [KE(x) for x in xs_np]
    pes = [PE(x) for x in xs_np]
    totals = [ke+pe for (ke, pe) in zip(kes, pes)]
    ax.plot(kes, color="blue", label="Kinetic")
    ax.plot(pes, color="red", label="Potential")
    ax.plot(totals, color="purple", label="Total")
    # ax.plot(np.nan, np.nan, )
    # ax.set_title("Energy")
    
    ax.set_xlabel("Time")
    ax.set_xticklabels([])
    ax.set_ylabel("Energy")
    ax.set_yticklabels([])
    
    def in_cube(x, t):
        for ii in range(3):
            if not (x[ii] > t[ii,0] and x[ii] < t[ii,1]):
                return False
        return True

    thermal_intervals = []
    thermals = [np.array(t).reshape((13,2), order="F") for t in result["config_dict"]["Xs_thermal"]]
    print("thermals",thermals)
    ii_start = None
    ii_end = None
    for kk in range(xs_np.shape[0]):

        # if in thermal
        if any([in_cube(xs_np[kk,:], t) for t in thermals]):
            # if not currently in thermal
            if ii_start is None:
                ii_start = kk
            # if currently in thermal
            else:
                continue 
        # if not in thermal
        else: 
            # if currently in thermal 
            if ii_start is not None: 
                ii_end = kk 
                thermal_intervals.append((ii_start, ii_end))
                ii_start = None 
            # if not currently in thermal 
            else:
                continue 
    # if still in thermal 
    if ii_start is not None and any([in_cube(xs_np[-1,:], t) for t in thermals]):
        ii_end = xs_np.shape[0]-1
        thermal_intervals.append((ii_start, ii_end))

    print("thermal_intervals",thermal_intervals)
    for interval in thermal_intervals: 
        ax.axvspan(interval[0], interval[1], alpha=0.5, color="orange")

    # def in_vision_cone(state, target, config_dict):
    #     obs_cone_angle = config_dict["obs_cone_angle"]
    #     obs_cone_length = config_dict["obs_cone_length"]
    #     dist = 
    #     return abs(angle) < obs_cone_angle && dist < m_obs_cone_length;

    # reward intervals 
    ii_start = None
    ii_end = None
    rs_np = np.array(result["rollout_rs"])
    # target = 
    reward_intervals = []
    time_since_target_idx = -1
    # for kk in range(xs_np.shape[0]):
    #     if rs_np[kk] > 0.5:
    #         if ii_start is None:
    #             ii_start = kk
    #         else:
    #             continue
    #     else:
    #         if ii_start is not None:
    #             ii_end = kk 
    #             reward_intervals.append((ii_start, ii_end))
    #             ii_start = None
    #         else:
    #             continue

    if result["config_dict"]["uct_downsample_traj_on"]:
        for kk in range(xs_np.shape[0]):
            if xs_np[kk, time_since_target_idx] <= result["config_dict"]["dots_decision_making_horizon"]:
                if ii_start is None:
                    ii_start = kk
                else:
                    continue
            else:
                if ii_start is not None:
                    ii_end = kk 
                    reward_intervals.append((ii_start, ii_end))
                    ii_start = None
                else:
                    continue
    else:
        raise ValueError("reward interval logic not implemented")

    print("reward_intervals",reward_intervals)
    for interval in reward_intervals: 
        ax.axvspan(interval[0], interval[1], alpha=0.5, color="green")
    ax.plot(np.nan, np.nan, color="orange", alpha=0.5, label="Thermal")
    ax.plot(np.nan, np.nan, color="green", alpha=0.5, label="Observation")
    ax.legend()


    time_idx = 12
    fig, ax = plotter.make_fig()
    ax.plot(xs_np[:, time_idx], xs_np[:, time_since_target_idx])
    for interval in reward_intervals: 
        ax.axvspan(xs_np[interval[0], time_idx], xs_np[interval[1], time_idx], alpha=0.5, color="green")
    

    

def plot_trajs_over_time_result(result):
    print(result["config_dict"]["ground_mdp_name"])
    print(result["config_path"])
    ground_mdp = get_mdp(result["config_dict"]["ground_mdp_name"], result["config_path"])
    config_dict = result["config_dict"]
    uct_xs = np.array(result["final_uct_xs"])
    uct_us = np.array(result["final_uct_us"])
    uct_rs = np.array(result["final_uct_rs"])
    uct_times = uct_xs[:,ground_mdp.timestep_idx()] * config_dict["uct_dt"]
    scp_xs = np.array(result["final_scp_xs"])
    scp_us = np.array(result["final_scp_us"])
    scp_rs = np.array(result["final_scp_rs"])
    scp_times = scp_xs[:,ground_mdp.timestep_idx()] * config_dict["uct_dt"]
    rollout_xs = np.array(result["rollout_xs"])
    rollout_us = np.array(result["rollout_us"])
    rollout_rs = np.array(result["rollout_rs"])
    rollout_times = rollout_xs[:,ground_mdp.timestep_idx()] * config_dict["uct_dt"]
    colors = plotter.get_n_colors(3)
    # second plot 
    fig, ax = plotter.make_fig(nrows=3, ncols=6)
    if uct_xs.shape[0] > 0:
        plot_trajs_over_time(config_dict, uct_xs, uct_us, uct_rs, uct_times, fig, ax, colors[0])
    if scp_xs.shape[0] > 0:
        plot_trajs_over_time(config_dict, scp_xs, scp_us, scp_rs, scp_times, fig, ax, colors[1])
    if rollout_xs.shape[0] > 0:
        plot_trajs_over_time(config_dict, rollout_xs, rollout_us, rollout_rs, rollout_times, fig, ax, colors[2])


def second_plot(result):
    # plot all time-since-seen target states
    ground_mdp = get_mdp(result["config_dict"]["ground_mdp_name"], result["config_path"])
    config_dict = result["config_dict"]
    uct_xs = np.array(result["final_uct_xs"])
    uct_us = np.array(result["final_uct_us"])
    uct_rs = np.array(result["final_uct_rs"])
    uct_times = uct_xs[:,ground_mdp.timestep_idx()] * config_dict["uct_dt"]
    scp_xs = np.array(result["final_scp_xs"])
    scp_us = np.array(result["final_scp_us"])
    scp_rs = np.array(result["final_scp_rs"])
    scp_times = scp_xs[:,ground_mdp.timestep_idx()] * config_dict["uct_dt"]
    rollout_xs = np.array(result["rollout_xs"])
    rollout_us = np.array(result["rollout_us"])
    rollout_rs = np.array(result["rollout_rs"])
    rollout_times = rollout_xs[:,ground_mdp.timestep_idx()] * config_dict["uct_dt"]

    number_of_targets = len(rollout_xs[0,:]) - (ground_mdp.timestep_idx() + 1)
    print("number_of_targets", number_of_targets)
    for ii in range(number_of_targets):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(uct_times, uct_xs[:,ground_mdp.timestep_idx()+ii+1], alpha=0.5, label="UCT state for target {}".format(ii))
        ax.plot(rollout_times, rollout_xs[:,ground_mdp.timestep_idx()+ii+1], alpha=0.5, label="SCP state for target {}".format(ii))
        ax.legend()
        ax.set_xlabel("time")
        ax.set_title("time-since-seen targets")


def plot_trajs_over_time(config_dict, xs, us, rs, ts, fig, ax, color, marker=None):
    state_labels = config_dict["ground_mdp_state_labels"][0:12]
    state_lims = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
    control_labels = config_dict["ground_mdp_control_labels"]
    control_lims = np.reshape(np.array(config_dict["ground_mdp_U"]), (8, 2), order="F")
    for ii in range(6):
        for jj in range(2):
            idx = jj * 6 + ii
            ax[jj,ii].plot(ts, xs[:,idx], color=color, marker=marker)
            ax[jj,ii].axhline(state_lims[idx,0], color="gray")
            ax[jj,ii].axhline(state_lims[idx,1], color="gray")
            ax[jj,ii].set_title(state_labels[idx])
    for ii in range(us.shape[1]):
        idx = ii
        ax[2,ii].plot(ts, us[:,ii], color=color, marker=marker)
        ax[2,ii].axhline(control_lims[idx,0], color="gray")
        ax[2,ii].axhline(control_lims[idx,1], color="gray")
    ax[2,4].plot(ts, rs, color=color)

def add_to_line_collections(line_collections, colors, config_dict, xs, us, rs, ts, color, marker=None):
    for ii in range(6):
        for jj in range(2):
            idx = jj * 6 + ii
            line_collections[idx].append(np.array((ts, xs[:,idx])).T)
            colors[idx].append(color)
    for ii in range(us.shape[1]):
        idx = 12 + ii
        line_collections[idx].append(np.array((ts, us[:,ii])).T)
        colors[idx].append(color)
    line_collections[16].append(np.array((ts, rs)).T)
    colors[16].append(color)

def render_branchdata(config_dict, tree):
    if len(tree.cbds) == 0:
        return
    render_fig, render_ax = plotter.make_3d_fig()
    # plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode
    state_labels = config_dict["ground_mdp_state_labels"][0:12]
    state_lims_color = "black"
    state_lims_alpha = 0.1
    state_lims = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
    state_lims[2,:] = -1 * state_lims[2,:]
    labels = ["zbars", "zs_ref", "xs"]
    colors = plotter.get_n_colors(len(labels))

    zbar0s = []
    zbars_lc3d = []
    zs_ref_lc3d = []
    xs_lc3d = []

    num_traj_plot = 500
    step = max((1,int(len(tree.cbds) / num_traj_plot))) # only plot 
    print("step",step)
    count = 0
    zbarHs = []
    for (cbd, sbd) in zip(tree.cbds, tree.sbds):
        # convert to nice numpy
        if sbd.is_valid and (count % step == 0) and all([np.linalg.norm(sbd.zbars[-1] - zbarH) > 1e-5 for zbarH in zbarHs]):
        # if sbd.is_valid and (count % step == 0): 
            zbarHs.append(sbd.zbars[-1])
            zbars = np.array(sbd.zbars) # (horizon, state_dim-1,)
            ubars = np.array(sbd.ubars) # (horizon, action_dim,)
            zs_ref = np.array(sbd.zs_ref) 
            us_ref = np.array(sbd.us_ref) 
            xs = np.array(sbd.xs) 
            us = np.array(sbd.us) 
            rs = np.array(sbd.rs) 
            if zbars.shape[0] != 0:
                zbars[:,2] = -1 * zbars[:,2]
                zbar0s.append(zbars[0,:])
                zbars_lc3d.append(zbars[:,0:3])
            if zs_ref.shape[0] != 0:
                zs_ref[:,2] = -1 * zs_ref[:,2]
                zs_ref_lc3d.append(zs_ref[:,0:3])
            if xs.shape[0] != 0:
                xs[:,2] = -1 * xs[:,2]
                xs_lc3d.append(xs[:,0:3])
        count += 1
    
    if len(zbar0s) == 0:
        return 

    zbar0s = np.array(zbar0s)
    render_ax.scatter(zbar0s[:,0], zbar0s[:,1], zbar0s[:,2], color=colors[0], alpha=0.5)

    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    # line_segments = Line3DCollection(zbars_lc3d, colors=colors[0], alpha=0.5)
    # render_ax.add_collection3d(line_segments)

    # line_segments = Line3DCollection(zs_ref_lc3d, colors=colors[1], alpha=0.5)
    # render_ax.add_collection3d(line_segments)
    
    line_segments = Line3DCollection(xs_lc3d, colors=colors[2], alpha=0.5)
    render_ax.add_collection3d(line_segments)

    render_ax.plot(np.nan, np.nan, np.nan, color=colors[0], label=labels[0])
    render_ax.plot(np.nan, np.nan, np.nan, color=colors[1], label=labels[1])
    render_ax.plot(np.nan, np.nan, np.nan, color=colors[2], label=labels[2])
    render_ax.legend()

    # xlim_ptp = np.ptp(render_ax.get_xlim())
    # ylim_ptp = np.ptp(render_ax.get_ylim())
    # zlim_ptp = np.ptp(render_ax.get_zlim())
    # render_ax.set_box_aspect((xlim_ptp, ylim_ptp, zlim_ptp))  # aspect ratio is 1:1:1 in data space
    render_ax.set_xlabel("x")
    render_ax.set_ylabel("y")
    render_ax.set_zlabel("z")

    plotter.render_xs_sixdofaircraft_game(np.array(tree.root)[np.newaxis,:], config_dict["ground_mdp_name"], config_dict, 
        obstacles_on=True, thermals_on=True, color="blue", alpha=0.5, fig=render_fig, ax=render_ax)


def plot_trajs_over_time_branchdata(config_dict, tree):
    if len(tree.cbds) == 0:
        return
    fig, ax = plotter.make_fig(nrows=3, ncols=6)
    # plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode
    if len(tree.cbds) == 0:
        return
    state_labels = config_dict["ground_mdp_state_labels"][0:12]
    state_lims_color = "black"
    state_lims_alpha = 0.5
    state_lims = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
    # state_lims[2,:] = -1 * state_lims[2,:]
    labels = ["zbars", "zs_ref", "xs", "zbar_H"]
    colors = plotter.get_n_colors_rgb(len(labels))
    line_collections = [[] for _ in range(17)]
    colors_for_collection = [[] for _ in range(17)]
    max_t = 0.0
    min_t = np.inf
    num_traj_plot = 500
    zbarHs = []
    step = max((1,int(len(tree.cbds) / num_traj_plot))) # only plot 
    count = 0
    for (cbd, sbd) in tqdm.tqdm(zip(tree.cbds, tree.sbds), desc="plot_trajs_over_time_branchdata"):
        # convert to nice numpy
        # if sbd.is_valid and (count % step == 0):
        if sbd.is_valid and (count % step == 0) and all([np.linalg.norm(sbd.zbars[-1] - zbarH) > 1e-5 for zbarH in zbarHs]):
            zbarHs.append(sbd.zbars[-1])

            timestep0 = np.array(cbd.timestep0) # double
            # convert to nice numpy
            zbars = np.array(sbd.zbars) # (horizon, state_dim-1,)
            ubars = np.array(sbd.ubars) # (horizon, action_dim,)
            zs_ref = np.array(sbd.zs_ref) 
            us_ref = np.array(sbd.us_ref) 
            xs = np.array(sbd.xs) 
            us = np.array(sbd.us) 

            if zbars.shape[0] != 0:
                ts = config_dict["ground_mdp_dt"] * (timestep0 + np.arange(zbars.shape[0]))
                if ts[-1] > max_t: max_t = ts[-1]
                if ts[0] < min_t: min_t = ts[0]
                rs = np.zeros(zbars.shape[0])
                add_to_line_collections(line_collections, colors_for_collection, config_dict, zbars, ubars, rs, ts, colors[0])

            if xs.shape[0] != 0:
                ts = config_dict["ground_mdp_dt"] * (np.arange(xs.shape[0]) + timestep0)
                if ts[-1] > max_t: max_t = ts[-1]
                if ts[0] < min_t: min_t = ts[0]
                rs = np.array(sbd.rs) 
                add_to_line_collections(line_collections, colors_for_collection, config_dict, xs, us, rs, ts, colors[2])

            if zs_ref.shape[0] != 0:
                ts = config_dict["ground_mdp_dt"] * (np.arange(zs_ref.shape[0]) + timestep0)
                if ts[-1] > max_t: max_t = ts[-1]
                if ts[0] < min_t: min_t = ts[0]
                rs = np.zeros(zs_ref.shape[0])
                add_to_line_collections(line_collections, colors_for_collection, config_dict, zs_ref, us_ref, rs, ts, colors[1])
        count += 1
    
    if len(line_collections[0]) == 0:
        return 

    from matplotlib.collections import LineCollection
    for ii in range(6):
        for jj in range(2):
            idx = jj * 6 + ii
            line_segments = LineCollection(line_collections[idx], colors = colors_for_collection[idx], alpha=0.5)
            ax[jj,ii].add_collection(line_segments)
            ax[jj,ii].set_title(state_labels[idx])
            ax[jj,ii].axhline(state_lims[idx,0], color=state_lims_color, alpha=state_lims_alpha)
            ax[jj,ii].axhline(state_lims[idx,1], color=state_lims_color, alpha=state_lims_alpha)
            ax[jj,ii].set_xlim([min_t, max_t])
    control_lims = np.reshape(np.array(config_dict["ground_mdp_U"]), (8, 2), order="F")
    control_labels = config_dict["ground_mdp_control_labels"]
    # control_lims = control_lims[3:7,:]
    # control_labels = control_labels[3:7]
    control_lims = control_lims[0:4,:]
    control_labels = control_labels[0:4]
    for ii in range(4):
        idx = 12 + ii
        line_segments = LineCollection(line_collections[idx], colors = colors_for_collection[idx], alpha=0.5)
        ax[2,ii].add_collection(line_segments)
        ax[2,ii].set_title(control_labels[ii])
        ax[2,ii].axhline(control_lims[ii,0], color=state_lims_color, alpha=state_lims_alpha)
        ax[2,ii].axhline(control_lims[ii,1], color=state_lims_color, alpha=state_lims_alpha)
        ax[2,ii].set_xlim([min_t, max_t])
    idx = 16
    line_segments = LineCollection(line_collections[idx], colors = colors_for_collection[idx], alpha=0.5)
    ax[2,4].add_collection(line_segments)
    ax[2,4].set_title("r")
    ax[2,4].axhline(0.0, color=state_lims_color, alpha=state_lims_alpha)

    ax[2,4].plot(np.nan, np.nan, color=colors[0], label=labels[0])
    ax[2,4].plot(np.nan, np.nan, color=colors[1], label=labels[1])
    ax[2,4].plot(np.nan, np.nan, color=colors[2], label=labels[2])
    ax[2,4].legend()


        
def main():

    num_seeds = 4
    parallel_on = True
    only_plot = False

    # config_path = util.get_config_path("fixed_wing")
    config_path = util.get_config_path("rollout")

    if only_plot:
        fns = glob.glob("../data/rollout_result_rollout_*.pkl")
        results = [util.load_pickle(fn) for fn in fns]
    else:
        start_time = timer.time()
        args = list(it.product([util.load_yaml(config_path)], range(num_seeds)))
        args = [[ii, *arg, parallel_on] for ii, arg in enumerate(args)] 
        if parallel_on:
            num_workers = mp.cpu_count() - 1
            pool = mp.Pool(num_workers)
            results = [x for x in tqdm.tqdm(pool.imap(_rollout, args), total=len(args))]
        else:
            results = [_rollout(arg) for arg in tqdm.tqdm(args)]
        results = [result for result in results if result is not None]   
        print("total time: {}s".format(timer.time() - start_time))

    for result in results:
        if result is not None:
            plot_result(result)
    plotter.save_figs("../plots/rollout.pdf")
    plotter.open_figs("../plots/rollout.pdf")
    # plotter.show_figs()

    print("done!")



if __name__ == '__main__':
    main()
