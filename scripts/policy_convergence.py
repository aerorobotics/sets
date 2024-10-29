
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
import plotter 
from util import util 
from build.bindings import get_mdp, get_dots_mdp, get_uct, RNG, UCT, MDP, \
    run_uct, Trajectory, rollout_action_sequence, Tree
# from learning.feedforward import Feedforward

from rollout import rollout, extract_from_tree_statistics

plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 1.0


def _policy_convergence(args):
    return policy_convergence(*args)

def policy_convergence(process_count, config_dict, seed, N, parallel_on, initial_state=None):
    config_dict["rollout_mode"] = "uct"
    config_dict["uct_N"] = N
    config_dict["uct_wct"] = 10000.0
    config_dict["uct_export_tree_statistics"] = True
    config_dict["uct_heuristic_mode"] = "shuffled"
    config_dict["uct_max_depth"] = 20
    config_dict["uct_c"] = 3.0
    rollout_result = rollout(process_count, config_dict, seed, parallel_on, initial_state=None)
    total_visit_counts_per_depth, visit_counts_per_depth = extract_from_tree_statistics(rollout_result["uct_trees"][0])
    max_frac_visit_counts_per_depth = []
    for depth in range(len(total_visit_counts_per_depth)):
        if len(visit_counts_per_depth[depth]) > 0:
            max_frac_visit_counts_per_depth.append(max(visit_counts_per_depth[depth]) / total_visit_counts_per_depth[0])
        else:
            max_frac_visit_counts_per_depth.append(0)
    # print("max_frac_visit_counts_per_depth",max_frac_visit_counts_per_depth)
    policy_convergence_result = {
        "N": N, 
        "max_frac_visit_counts_per_depth" : max_frac_visit_counts_per_depth
    }
    # todo: save data
    util.save_pickle(policy_convergence_result, "../data/policy_convergence_result_cast_game2_{}.pkl".format(process_count))
    return policy_convergence_result


# My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01
def log_tick_formatter(val, pos=None):
    print("val",val)
    return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation


def plot_policy_convergence_results(results):
    fig, ax = plotter.make_3d_fig()

    Ns = set()
    Ds = set()
    top = dict()
    for result in results:
        Ns.add(result["N"])
        for depth, max_frac_visit in enumerate(result["max_frac_visit_counts_per_depth"]):
            Ds.add(depth)
            if (result["N"], depth) not in top.keys():
                top[(result["N"], depth)] = []
            top[(result["N"], depth)].append(max_frac_visit)
    Ns = list(Ns)
    Ds = list(Ds)
    Ns.sort()
    Ds.sort()

    top_np = np.zeros((len(Ds), len(Ns)))
    for (N, D), val in top.items():
        ii_N = Ns.index(N)
        ii_D = Ds.index(D)
        top_np[ii_D, ii_N] = np.mean(val)

    print("Ns",Ns)
    print("Ds",Ds)
    print("top_np",top_np)

    plot_type = "surf"

    if plot_type == "bar":
        _y = np.arange(len(Ns))
        _xx, _yy = np.meshgrid(np.array(Ds), _y)
        x, y = _xx.ravel(order="F"), _yy.ravel(order="F")
        top_np = top_np.ravel()
        bottom = np.zeros_like(top_np)
        width = 1 
        depth = 1
        from matplotlib.colors import LightSource
        lightsource = LightSource(azdeg=60, altdeg=45)
        ax.bar3d(x, y, bottom, width, depth, top_np, shade=True, alpha=1.0, color=(0,0,1,0), lightsource=lightsource, edgecolor='black')
        ax.set_yticks(_y, fontsize=8)
        ax.set_yticklabels(Ns)

    elif plot_type == "surf":

        y = np.arange(len(Ns))
        # y = np.array(Ns)
        X, Y = np.meshgrid(np.array(Ds), y)

        from matplotlib import cm

        # Plot the surface.
        print("X.shape", X.shape)
        print("Y.shape", Y.shape)
        print("top_np.shape", top_np.shape)
        surf = ax.plot_surface(X, Y, top_np.T, cmap=cm.coolwarm, linewidth=1, antialiased=True)
        ax.set_yticks(y[::2])

    ax.set_title("Policy Convergence")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Number Simulations")
    ax.set_zlabel("Most Visited Node's \n Fraction of Visits")

    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.set_yticklabels([f"$10^{{{int(np.log10(val))}}}$" for val in Ns[::2]])

    # ax.set_yscale('log')

    [item.set_fontsize(8) for item in ax.get_xticklabels()]
    [item.set_fontsize(8) for item in ax.get_yticklabels()]
    [item.set_fontsize(8) for item in ax.get_zticklabels()]
    ax.zaxis.label.set_rotation(180)




def main():

    num_seeds = 1
    parallel_on = False
    only_plot = False

    # config_path = util.get_config_path("fixed_wing")
    # config_path = util.get_config_path("value_convergence")
    config_path = util.get_config_path("policy_convergence")

    # Ns = [100, 1000, 10000]
    # Ns = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000] # this is as far as I got on 64
    # Ns = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
    Ns = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000] # this is as far as I got on 32 gb
    # Ns = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000] 
    # Ns = [50, 100, 500, 1000, 5000, 10000, 50000] 
    # Ns = [50, 100, 500]

    if only_plot:
        fns = glob.glob("../data/policy_convergence_*.pkl")
        print("fns",fns)
        results = [util.load_pickle(fn) for fn in fns]
    else:
        start_time = timer.time()
        args = list(it.product([util.load_yaml(config_path)], range(num_seeds), Ns))
        args = [[ii, *arg, parallel_on] for ii, arg in enumerate(args)] 
        if parallel_on:
            num_workers = mp.cpu_count() - 1
            pool = mp.Pool(num_workers)
            results = [x for x in tqdm.tqdm(pool.imap(_policy_convergence, args), total=len(args))]
        else:
            results = [_policy_convergence(arg) for arg in tqdm.tqdm(args)]
        results = [result for result in results if result is not None]   
        print("total time: {}s".format(timer.time() - start_time))

    plot_policy_convergence_results(results)

    plotter.save_figs("../plots/policy_convergence.pdf")
    plotter.open_figs("../plots/policy_convergence.pdf")

    print("done!")



if __name__ == '__main__':
    main()
