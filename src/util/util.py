

import pathlib
import yaml
import tqdm
import glob
import re
import os
import pickle 
import numpy as np 
import scipy 
from queue import Queue, Empty


def smooth_xs(xs_np, window_size, skip_idxs=[]):
    # xs is (num_timesteps, state_dim) (hopefully)
    half_width = int(np.floor(window_size/2.0))
    x0 = xs_np[0,:]
    x0_padding = np.vstack([x0] * half_width)
    xf = xs_np[-1,:]
    xf_padding = np.vstack([xf] * half_width)
    xs_np_padded = np.vstack((x0_padding, xs_np, xf_padding))
    xs_np_smoothed = xs_np.copy()
    for ii_col in range(xs_np.shape[1]):
        if ii_col in skip_idxs:
            continue
        xs_np_smoothed[:, ii_col] = np.convolve(xs_np_padded[:, ii_col], np.ones(window_size)/float(window_size), mode='valid')[:-1]
    return xs_np_smoothed


def subsample_list(xs, num_samples):
    assert type(xs) == list
    assert type(num_samples) == int
    if num_samples > len(xs) - 1:
        return xs
    step = len(xs) // num_samples
    idxs = np.arange(0, len(xs)-1, step, dtype=int)
    return [xs[idx] for idx in idxs]


# ---------- UTIL ----------
def split_list(my_list, len_split_list):
    split_list = []
    for ii in range(len_split_list):
        split_list.append([])
    for ii, item in enumerate(my_list):
        split_list[ii % len_split_list].append(item)
    return split_list

# minor adjustment to tqdm to work with multiprocessing
class MyProgressBar: 

    def __init__(self, rank, queue, total, desc=None):
        self.rank = rank 
        self.queue = queue 
        self.total = total
        self.desc = desc

    def init(self):
        pbar = None 
        if self.rank == 0:
            if self.desc is not None:
                pbar = tqdm.tqdm(total=self.total, desc=self.desc)
            else:
                pbar = tqdm.tqdm(total=self.total)
        return pbar

    def update(self, to_add, pbar):
        if self.rank == 0:
            count = to_add
            try:
                while True:
                    count += self.queue.get_nowait()
            except Empty:
                pass
            pbar.update(count)
        else:
            self.queue.put_nowait(to_add)

def save_npy(np_arr, fn):
    np.save(fn, np_arr)

def load_npy(fn):
    return np.load(fn)

def save_pickle(obj, fn):
    with open(fn, 'wb') as h:
        pickle.dump(obj, h, pickle.HIGHEST_PROTOCOL)

def load_pickle(fn):
    try:
        with open(fn, 'rb') as h:
            obj = pickle.load(h)
        return obj
    except:
        import pickle5
        with open(fn, 'rb') as h:
            obj = pickle5.load(h)
        return obj 

def load_yaml(path):
    py_dict = yaml.safe_load(pathlib.Path(path).read_text())
    return py_dict

class DataHelper: 

    def __init__(self):
        self.idx_dict = None
        self.ncols = None

    def make_idx_dict(self, state_dim, control_dim, horizon, num_branches):
        shape_dict = {
            "state_dim"         : 1, 
            "control_dim"       : 1,
            "num_horizon"       : 1, 
            "num_branches"      : 1, 
            "x0"  : state_dim, 
            "tss" : num_branches * horizon,                
            "xss" : num_branches * horizon * state_dim, 
            "uss" : num_branches * horizon * control_dim, 
        }
        curr_idx = 0 
        self.idx_dict = {}
        for key, shape in shape_dict.items():
            self.idx_dict[key] = np.arange(curr_idx, curr_idx + shape)
            curr_idx = curr_idx + shape 
        self.ncols = curr_idx

    def convert_mp_data_dict_to_arr(self, data_dict): 
        my_arr = np.empty((self.ncols,), dtype=float)
        for key, data_value in data_dict.items():
            if type(data_value) is np.ndarray: data_value = data_value.ravel()
            my_arr[self.idx_dict[key]] = data_value
        return my_arr

    def convert_mp_arr_to_data_dict(self, arr):
        my_dict = {}
        for key, idx in self.idx_dict.items():
            my_dict[key] = arr[idx]
        # convert types 
        my_dict["state_dim"] = int(my_dict["state_dim"])
        my_dict["control_dim"] = int(my_dict["control_dim"])
        my_dict["num_horizon"] = int(my_dict["num_horizon"])
        my_dict["num_branches"] = int(my_dict["num_branches"])
        # convert shapes 
        my_dict["tss"] = np.reshape(my_dict["tss"], (my_dict["num_branches"], my_dict["num_horizon"], 1))
        my_dict["xss"] = np.reshape(my_dict["xss"], (my_dict["num_branches"], my_dict["num_horizon"], my_dict["state_dim"]))
        my_dict["uss"] = np.reshape(my_dict["uss"], (my_dict["num_branches"], my_dict["num_horizon"], my_dict["control_dim"]))
        return my_dict

    def get_arr_val_from_key(self, key, arr):
        return arr[self.idx_dict[key]]


def get_config_path(full_config_path):
    examples_folder_path = os.path.join(os.path.dirname(__file__), '../../configs')
    # strip directory
    config_name = full_config_path.split("/")[-1].split(".")[0]
    matches = list(pathlib.Path(examples_folder_path).rglob(config_name + ".yaml"))
    if len(matches) == 0:
        raise Exception("Config file {} not found!".format(config_name))
    elif len(matches) > 1:
        print("Matches: ", matches)
        raise Exception("Multiple named config files! Rename configs to be unambiguous.")
    else:
        # print("running: ", matches[0])
        return str(matches[0])

# ---------- OPERATIONS ----------
def load_motion_primitives_data(model_config_dict):
    motion_primitives_data_fns = glob.glob(re.sub("IDX", "*", model_config_dict["model_motion_primitives_data_path"]))
    motion_primitives_data = []
    for fn in motion_primitives_data_fns:
        motion_primitives_data.extend(load_npy(fn))
    motion_primitives_data = np.array(motion_primitives_data)
    return motion_primitives_data

def clean_motion_primitives_data(model_config_dict):
    print("cleaning dispersion solns")
    motion_primitives_data_fns = glob.glob(re.sub("IDX", "*", model_config_dict["model_motion_primitives_data_path"]))
    for fn in motion_primitives_data_fns:
        print("removing fn: {}".format(fn))
        os.remove(fn)
    return None

# random 
def print_tree_stats(tree):
    num_states = 0
    for traj in tree.trajs:
        xs_np = np.array(traj.xs)
        # print("xs_np.shape", xs_np.shape)
        num_states += xs_np.shape[0]
    num_nodes = len(tree.node_states)
    print("num_states: ", num_states)
    print("num_nodes: ", num_nodes)
    return num_states, num_nodes

# https://math.stackexchange.com/questions/2431159/\
# how-to-obtain-the-equation-of-the-projection-shadow-of-an-ellipsoid-into-2d-plan
def project_nd_ellipse_to_2d(D, w, psi0, psi1):
    debug = True

    D = D.copy()
    w = w.copy()

    if not (psi0 == 0 and psi1 == 1):
        D[:,[0,psi0]] = D[:,[psi0,0]]
        D[:,[1,psi1]] = D[:,[psi1,1]]
        D[[0,psi0],:] = D[[psi0,0],:]
        D[[1,psi1],:] = D[[psi1,1],:]
        w[[0,psi0],:] = w[[psi0,0],:]
        w[[1,psi1],:] = w[[psi1,1],:]
        # exit("project_nd_ellipse_to_2d not implemented") 

    mode_project = 1

    if mode_project == 0:
        D_2d = D[0:2,0:2]
        w_2d = w[0:2,:]

    elif mode_project == 1:
        try:
            D_inv = np.linalg.pinv(D) 
            A = D_inv.T @ D_inv
            J = A[0:2, 0:2]
            L = A[2:, 0:2] 
            K = A[2:, 2:]
            K_inv = np.linalg.pinv(K)
            D_2d_inv = np.linalg.cholesky(J - L.T @ K_inv @ L).T
            D_2d = np.linalg.pinv(D_2d_inv)
            w_2d = w[0:2,:]
        except np.linalg.LinAlgError as e: 
            if debug: print("Projection linalg error, D=",D,"; w=",w,"; A=",A," error=",e)
            return D[0:2,0:2], w[0:2,:]
            #return None, None

    return D_2d, w_2d


def convert_ellipse_parameterization(mode, A=None, b=None, c=None, D=None, E=None, f=None, w=None):
    # convert from quadratic form to affine transformation of unit ball
    if mode == 1:
        # from: \{ x | x^T A x + b^T x + c = 0 \}
        # to: \{ D v + w | \forall \|v\|_2 = 1 \}
        if (A is None or b is None or c is None):
            return None, None
        n = A.shape[0]
        # prepare decomp
        L, U = np.linalg.eig(A)
        A_inv = np.linalg.pinv(A)
        S = np.zeros((n,n))
        S_inv = np.zeros((n,n))
        for i in range(n):
            S[i,i] = (-1 * L[i]) ** (1/2)
            S_inv[i,i] = 1 / S[i,i]
        # convert to intermediate parameterization
        h = - 1/2 * A_inv @ b
        k = c - 1/4 * b.T @ A_inv @ b
        # convert to affine transformation of unit ball 
        D = np.sqrt(k) * U @ S_inv
        w = h
        return D, w

    # convert from intermediate form to affine transformation of unit ball
    elif mode == 2:
        # from: \{ x | \| E x + f \|_2 = 1 \}
        # to: \{ D v + w | \forall \|v\|_2 = 1 \}
        if (E is None or f is None):
            return None, None
        # E_inv = np.linalg.pinv(E)
        E_inv = scipy.linalg.fractional_matrix_power(E, -1.0)
        D = E_inv
        w = - 1 * E_inv @ f
        return D, w

    # convert from affine transformation of unit ball to intermediate form
    elif mode == 3:
        # from: \{ D v + w | \forall \|v\|_2 = 1 \}
        # to: \{ x | \| E x + f \|_2 = 1 \}
        if (D is None or w is None):
            return None, None
        D_inv = np.linalg.pinv(D)
        E = D_inv
        f = - 1 * E @ w
        return E, f

    else:
        exit("mode not implemented")


def get_random_obstacles(config_dict, seed, num_obstacles, obstacle_length):

    rng = np.random.default_rng(seed)

    state_lims = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
    x0 = np.array(config_dict["ground_mdp_x0"])

    obstacles = []
    for ii in range(num_obstacles):
        try_count = 0
        while (try_count < 1000):
            center = rng.uniform(size=(3,)) * (state_lims[0:3,1] - state_lims[0:3,0]) + state_lims[0:3,0]
            cube = np.array([
                [center[0] - obstacle_length / 2.0, center[0] + obstacle_length / 2.0],
                [center[1] - obstacle_length / 2.0, center[1] + obstacle_length / 2.0],
                # [center[2] - obstacle_length / 2.0, center[2] + obstacle_length / 2.0],
                [-2.2, 0.0],
                ])
            if not vec_in_cube(x0[0:3], cube):
                break 
        obstacle = state_lims.copy()
        obstacle[0:3,:] = cube
        obstacles.append(obstacle)
    return obstacles

def vec_in_cube(vec, cube):
    for ii in range(vec.shape[0]):
        if vec[ii] > cube[ii,1] or vec[ii] < cube[ii,0]:
            return False
        else:
            return True


def get_obstacles(config_dict, timestep):

    if "config_name" in config_dict.keys() and config_dict["config_name"] == "cast_game3_exp3":
        return get_random_obstacles(config_dict, 1, 7, 0.75)

    obstacles = []
    for ii, obstacle in enumerate(config_dict["ground_mdp_obstacles"]):
        if ii not in config_dict["ground_mdp_special_obstacle_idxs"]: 
            obstacles.append(np.reshape(np.array(obstacle), (13,2), order="F"))
        else:
            if config_dict["ground_mdp_name"] == "GameSixDOFAircraft":
                obstacle = np.reshape(np.array(obstacle), (13,2), order="F")
                dynamic_obstacle = obstacle.copy()

                obstacle_center = (obstacle[0:3,0] + obstacle[0:3,1]) / 2 + timestep * obstacle[3:6,0]

                dynamic_obstacle[0,0] = obstacle_center[0] - (obstacle[0,1] - obstacle[0,0]) / 2
                dynamic_obstacle[0,1] = obstacle_center[0] + (obstacle[0,1] - obstacle[0,0]) / 2

                dynamic_obstacle[1,0] = obstacle_center[1] - (obstacle[1,1] - obstacle[1,0]) / 2
                dynamic_obstacle[1,1] = obstacle_center[1] + (obstacle[1,1] - obstacle[1,0]) / 2

                dynamic_obstacle[2,0] = obstacle_center[2] - (obstacle[2,1] - obstacle[2,0]) / 2
                dynamic_obstacle[2,1] = obstacle_center[2] + (obstacle[2,1] - obstacle[2,0]) / 2

                obstacles.append(dynamic_obstacle)

            else:
                # here is our hacked human dynamic obstacle, alternative is to read in from vicon data 
                obstacle = np.reshape(np.array(obstacle), (13,2), order="F")

                # start at bottom left
                # initial_center = np.array([-1.6, -2.8])
                z_center = (obstacle[2,0] + obstacle[2,1]) / 2
                initial_center = np.array([2.5, 1.0, z_center])
                intermediate_center = np.array([1.4, 1.0, z_center])
                terminal_center = np.array([1.4, -1.0, z_center])

                initial_obstacle = np.zeros((13,2))
                intermediate_obstacle = np.zeros((13,2))
                terminal_obstacle = np.zeros((13,2))

                initial_obstacle[2,:] = obstacle[2,:]
                intermediate_obstacle[2,:] = obstacle[2,:]
                terminal_obstacle[2,:] = obstacle[2,:]
                
                initial_obstacle[0,0] = initial_center[0] - (obstacle[0,1] - obstacle[0,0]) / 2
                initial_obstacle[0,1] = initial_center[0] + (obstacle[0,1] - obstacle[0,0]) / 2
                initial_obstacle[1,0] = initial_center[1] - (obstacle[1,1] - obstacle[1,0]) / 2
                initial_obstacle[1,1] = initial_center[1] + (obstacle[1,1] - obstacle[1,0]) / 2

                intermediate_obstacle[0,0] = intermediate_center[0] - (obstacle[0,1] - obstacle[0,0]) / 2
                intermediate_obstacle[0,1] = intermediate_center[0] + (obstacle[0,1] - obstacle[0,0]) / 2
                intermediate_obstacle[1,0] = intermediate_center[1] - (obstacle[1,1] - obstacle[1,0]) / 2
                intermediate_obstacle[1,1] = intermediate_center[1] + (obstacle[1,1] - obstacle[1,0]) / 2

                terminal_obstacle[0,0] = terminal_center[0] - (obstacle[0,1] - obstacle[0,0]) / 2
                terminal_obstacle[0,1] = terminal_center[0] + (obstacle[0,1] - obstacle[0,0]) / 2
                terminal_obstacle[1,0] = terminal_center[1] - (obstacle[1,1] - obstacle[1,0]) / 2
                terminal_obstacle[1,1] = terminal_center[1] + (obstacle[1,1] - obstacle[1,0]) / 2
                
                H = config_dict["ground_mdp_H"]

                if config_dict["wind_mode"] == "thermal":
                    switching_times = [H/8, H/4, 3*H/8, H/2]
                else:
                    switching_times = [H/4, H/2, 3*H/4, H]

                if timestep < switching_times[0]:
                    obstacles.append(initial_obstacle)
                    obstacles[-1][-1,0] = timestep
                elif timestep < switching_times[1]:
                    alpha = (timestep - switching_times[0]) / (switching_times[1] - switching_times[0]) # in [0,1]
                    obstacles.append((1 - alpha) * initial_obstacle + alpha * intermediate_obstacle)
                    displacement = intermediate_center - initial_center
                    obstacles[-1][3:6,0] = displacement / ((switching_times[2] - switching_times[1]) * config_dict["ground_mdp_dt"])
                    obstacles[-1][-1,0] = timestep
                elif timestep < switching_times[2]:
                    alpha = (timestep - switching_times[1]) / (switching_times[2] - switching_times[1]) # in [0,1]
                    obstacles.append((1 - alpha) * intermediate_obstacle + alpha * terminal_obstacle)
                    displacement = terminal_center - intermediate_center
                    obstacles[-1][3:6,0] = displacement / ((switching_times[3] - switching_times[2]) * config_dict["ground_mdp_dt"])
                    obstacles[-1][-1,0] = timestep
                else:
                    obstacles.append(terminal_obstacle)
                    obstacles[-1][-1,0] = timestep
        
    return obstacles


def get_thermals(config_dict, timestep):

    observation_with_dynamic_obstacle_and_thermal = \
        config_dict["ground_mdp_name"] == "SixDOFAircraft" and \
        config_dict["reward_mode"] == "observation" and \
        (config_dict["wind_mode"] == "thermal" or config_dict["aero_mode"] == "neural_thermal_moment") and \
        len(config_dict["ground_mdp_obstacles"]) in [4,5]
    observation_with_dynamic_obstacle_and_thermal = False

    if observation_with_dynamic_obstacle_and_thermal:

        thermals = []

        if "Xs_thermal" in config_dict.keys():
            Xs_thermal = config_dict["Xs_thermal"]
            Vs_thermal = config_dict["Vs_thermal"]
        
            H = config_dict["ground_mdp_H"]
            switching_times = [H/2]

            if timestep < switching_times[0]:
                pass 
            # elif timestep < switching_times[1]:
            else:
                for ii in range(len(Xs_thermal)):
                    X_thermal = np.reshape(Xs_thermal[ii], (13,2), order="F")
                    V_thermal = np.reshape(Vs_thermal[ii], (3,))
                    thermals.append((X_thermal, V_thermal))

    else:
        thermals = []
        if "Xs_thermal" in config_dict.keys():
            
            Xs_thermal = config_dict["Xs_thermal"]
            Vs_thermal = config_dict["Vs_thermal"]

            for ii in range(len(Xs_thermal)):
                X_thermal = np.reshape(Xs_thermal[ii], (13,2), order="F")
                V_thermal = np.reshape(Vs_thermal[ii], (6,))
                thermals.append((X_thermal, V_thermal))

    return thermals