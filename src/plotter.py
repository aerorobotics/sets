
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
# import pandas as pd
import argparse
import pickle 
import glob 
import os

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_pdf import PdfPages 
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.figure import figaspect
from matplotlib.gridspec import GridSpec

from tqdm import tqdm
from os.path import split, isdir, join, exists
from os import makedirs, getcwd
from subprocess import call
import sys

# custom 
from util import util 
# from build.bindings import get_mdp

# defaults
plt.rcParams.update({'font.size': 4})
plt.rcParams['lines.linewidth'] = 0.5


def has_figs():
    if len(plt.get_fignums()) > 0:
        return True
    else:
        return False


def show_figs():
    plt.show()


def make_only_fig():
    fig = plt.figure()
    return fig 


def make_fig(nrows=None,ncols=None):
    if nrows is None or ncols is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(nrows=nrows,ncols=ncols,squeeze=False)
    return fig, ax


def make_fig_3d_2d_ax():
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2)
    return fig, [ax0,ax1]

def make_fig_3d_1x2_ax():
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    return fig, [ax0,ax1]

def make_fig_3d_2x2_ax():
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax0 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3 = fig.add_subplot(2, 2, 4, projection='3d')
    return fig, [ax0,ax1,ax2,ax3]

def make_fig_3d_1x3_ax():
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    return fig, [ax0,ax1,ax2]

def make_3d_fig():
    fig = plt.figure() 
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    return fig, ax


def make_n_colors(n):
    fig = plt.figure()
    colors = [] 
    for i in range(n):
        line = plt.plot(np.nan, np.nan)
        colors.append(line[0].get_color())
    plt.close(fig)
    return colors


def save_figs(filename):
    file_dir, file_name = split(filename)
    if len(file_dir) > 0 and not (isdir(file_dir)):
        makedirs(file_dir)
    fn = join(getcwd(), filename)
    pp = PdfPages(fn)
    for i in tqdm(plt.get_fignums(), desc="save_figs"):
        pp.savefig(plt.figure(i))
        # plt.close(plt.figure(i))
    pp.close()

def close_figs():
    for i in tqdm(plt.get_fignums(), desc="save_figs"):
        plt.close(plt.figure(i))

def save_one_fig(filename):
    file_dir, file_name = split(filename)
    if len(file_dir) > 0 and not (isdir(file_dir)):
        makedirs(file_dir)
    fn = join(getcwd(), filename)
    plt.savefig(fn, dpi=50)


def open_figs(filename):
    pdf_path = join(getcwd(), filename)
    if exists(pdf_path):
        if "linux" in sys.platform:
            call(["xdg-open", pdf_path])
        elif "darwin" in sys.platform:
            call(["open", pdf_path])


def get_n_colors(n):
    # colors = []
    # for _ in range(n):
    #     line = ax.plot(np.nan, np.nan)
    #     colors.append(line[0].get_color())
    # plt.close(fig)
    if n<5:
        return ["blue", "orange", "green", "magenta", "cyan"][0:n]

    colors = []
    fig, ax = plt.subplots()
    # cm = plt.get_cmap('gist_rainbow')
    cm = plt.get_cmap('tab20')
    ax.set_prop_cycle(color=[cm(1.*i/n) for i in range(n)])
    for _ in range(n):
        line = ax.plot(np.nan, np.nan)
        colors.append(line[0].get_color())
    plt.close(fig)
    return colors

def get_n_colors_rgb(n):
    return [mpl.colors.to_rgb(color) for color in get_n_colors(n)]

# https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def get_cmap(name):
    return mpl.cm.get_cmap(name)


def add_2d_segments(segments, fig, ax, linewidth, colors, alpha):
    ln_coll = LineCollection(segments, linewidth=linewidth, colors=colors, alpha=alpha)
    ax.add_collection(ln_coll)


def add_3d_segments(segments, fig, ax, linewidth, colors, alpha):
    ln_coll = Line3DCollection(segments, linewidth=linewidth, colors=colors, alpha=alpha, rasterized=True)
    ax.add_collection(ln_coll)


def plot_2d_cube(cube, fig, ax, color, alpha):
    ax.plot([cube[0,0], cube[0,1]], [cube[1,0], cube[1,0]], color=color, alpha=alpha)
    ax.plot([cube[0,1], cube[0,1]], [cube[1,0], cube[1,1]], color=color, alpha=alpha)
    ax.plot([cube[0,1], cube[0,0]], [cube[1,1], cube[1,1]], color=color, alpha=alpha)
    ax.plot([cube[0,0], cube[0,0]], [cube[1,1], cube[1,0]], color=color, alpha=alpha)


def plot_3d_cube(points, fig, ax, color, alpha):
    # points in 3 x 2
    xlims = points[0,:]
    ylims = points[1,:]
    zlims = points[2,:]

    # surface 1
    xx, yy = np.meshgrid(xlims, ylims)
    zz = zlims[0] * np.ones((2,2))
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color)

    # surface 2
    xx, yy = np.meshgrid(xlims, ylims)
    zz = zlims[1] * np.ones((2,2))
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color)

    # surface 3
    xx, zz = np.meshgrid(xlims, zlims)
    yy = ylims[0] * np.ones((2,2))
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color)

    # surface 4
    xx, zz = np.meshgrid(xlims, zlims)
    yy = ylims[1] * np.ones((2,2))
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color)

    # surface 5
    yy, zz = np.meshgrid(ylims, zlims)
    xx = xlims[0] * np.ones((2,2))
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color)

    # surface 6
    yy, zz = np.meshgrid(ylims, zlims)
    xx = xlims[1] * np.ones((2,2))
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color)

    return fig, ax


def plot_cylinder(base_center, radius, height, fig, ax, color, alpha):
    # draw edge
    x = np.linspace(base_center[0] - radius, base_center[0] + radius, 100)
    z = np.linspace(base_center[2], base_center[2] + height, 100)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(radius**2 - (Xc - base_center[0])**2) + base_center[1]
    rstride = 20
    cstride = 10
    ax.plot_surface(Xc, Yc, Zc, alpha=alpha, color=color, rstride=rstride, cstride=cstride)
    # draw faces
    theta = np.linspace(0, 2*np.pi, 100)
    radii = np.linspace(0, radius, 100)
    Theta, Radii = np.meshgrid(theta, radii)
    Xs = Radii * np.cos(Theta) + base_center[0]
    Ys = Radii * np.sin(Theta) + base_center[1]
    Zs = base_center[2] * np.ones(Xs.shape)
    ax.plot_surface(Xs, Ys, Zs, alpha=alpha, color=color, rstride=rstride, cstride=cstride)
    Zs = (base_center[2] + height) * np.ones(Xs.shape)
    ax.plot_surface(Xs, Ys, Zs, alpha=alpha, color=color, rstride=rstride, cstride=cstride)
    return fig, ax


def plot_cone(cone_center, cone_radius, cone_length, roll, pitch, yaw, fig, ax, color, alpha):

    # from body from to world frame 
    R = np.array([
        [np.cos(pitch) * np.cos(yaw), 
         np.cos(pitch) * np.sin(yaw), 
         -1 * np.sin(pitch)],
        [-1 * np.cos(pitch) * np.sin(yaw) + np.sin(yaw) * np.sin(pitch) * np.cos(yaw), 
         np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw), 
         np.sin(roll) * np.cos(pitch)],
        [np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw),
         -1 * np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw),
         np.cos(roll) * np.cos(pitch)
        ]])
    R = R.T 

    num_thetas = 20
    r1 = cone_radius
    h1 = 0.2
    h2 = 1.0

    for jj in range(num_thetas):

        th_jj = jj / num_thetas * 2 * np.pi 
        th_jjp1 = (jj+1) / num_thetas * 2 * np.pi 

        xs = cone_length * np.array([h2, h2, h1, h1])

        ys = np.array([
            r1 * h2 * np.sin(th_jjp1), 
            r1 * h2 * np.sin(th_jj), 
            r1 * h1 * np.sin(th_jj), 
            r1 * h1 * np.sin(th_jjp1)
            ])

        zs = np.array([
            r1 * h2 * np.cos(th_jjp1), 
            r1 * h2 * np.cos(th_jj), 
            r1 * h1 * np.cos(th_jj), 
            r1 * h1 * np.cos(th_jjp1)
            ])

        XYZ = np.vstack([xs, ys, zs])

        XYZ_rot = R @ XYZ

        xps = XYZ_rot[0,:]
        yps = XYZ_rot[1,:]
        zps = XYZ_rot[2,:]

        xps = [xp + cone_center[0,0] for xp in xps]
        yps = [yp + cone_center[1,0] for yp in yps]
        zps = [zp + cone_center[2,0] for zp in zps]

        verts = [list(zip(xps, yps, zps))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=color, alpha=alpha))

    return fig, ax


# render xs 
def render_xs(xs, mdp_name, config_dict, fig=None, ax=None, color=None, alpha=1.0):
    # xs is np array in (t, n)
    if "SingleIntegrator2d" == config_dict["ground_mdp_name"]:
        return render_xs_singleintegrator2d(xs, mdp_name, config_dict, fig, ax, color)
    elif "Cartpole" == config_dict["ground_mdp_name"]:
        return render_xs_cartpole(xs, mdp_name, config_dict, fig, ax, color, alpha)
    elif "MountainCar" == config_dict["ground_mdp_name"]:
        return render_xs_mountaincar(xs, mdp_name, config_dict, fig, ax, color, alpha)
    elif "SixDOFAircraft" == config_dict["ground_mdp_name"]:
        return render_xs_sixdofaircraft(xs, mdp_name, config_dict, fig, ax, color, alpha)
    elif "GameSixDOFAircraft" == config_dict["ground_mdp_name"]:
        return render_xs_sixdofaircraft_game(xs, mdp_name, config_dict, fig, ax, color, alpha)
    else:
        raise NotImplementedError("render_xs not implemented for ground_mdp_name: {}".format(config_dict["ground_mdp_name"]))


def render_xs_mountaincar(xs, mdp_name, config_dict, fig, ax, color, alpha):
    mode = 0
    if mode== 0:
        if fig is None or ax is None:
            fig, ax = make_fig()
        xs = np.array(xs)
        if color is not None:
            ax.plot(xs[:,0], xs[:,1], marker="o", color=color, alpha=alpha)
        else:
            ax.plot(xs[:,0], xs[:,1], "k-", alpha=alpha)
        ax.set_xlabel("x")
        ax.set_ylabel("vx")
        ax.axvline(config_dict["ground_mdp_xd"][0], color="green")
    elif mode== 1:
        if fig is None or ax is None:
            fig, ax = make_fig()
        xs = np.array(xs)
        ax.plot(xs[:,0], np.sin(3 * xs[:,0])/3 + 0.02, "green")
        ax.plot(xs[-1,0], np.sin(3 * xs[-1,0])/3 + 0.02, "green", marker="o")
        # plot surface
        surface_xs = np.linspace(-1.2, 0.6, 100)
        surface_ys = np.sin(3 * surface_xs)/3
        ax.plot(surface_xs, surface_ys)
    ax.set_title("render_xs")
    return fig, ax


def render_xs_singleintegrator2d(xs, mdp_name, config_dict, fig, ax, color):
    if fig is None or ax is None:
        fig, ax = make_fig()
    if color is None:
        color = "blue"
    ax.plot(xs[:,0], xs[:,1], color=color)
    ax.plot(xs[0,0], xs[0,1], marker="o", color=color)
    ax.plot(xs[-1,0], xs[-1,1], marker="s", color=color)
    xg = config_dict["ground_mdp_xd"]
    goal_eps = config_dict["ground_mdp_goal_eps"]
    ths = np.linspace(0, 2*np.pi, 100)
    ax.plot(xg[0] + goal_eps * np.cos(ths), xg[1] + goal_eps * np.sin(ths), color="green")
    ax.plot(xg[0], xg[1], marker="*", color="green")
    print(config_dict["ground_mdp_obstacles"])
    for obstacle in config_dict["ground_mdp_obstacles"]:
        obstacle_np = np.reshape(np.array(obstacle), (3,2), order="F")
        rect = Rectangle((obstacle_np[0,0], obstacle_np[1,0]), np.ptp(obstacle_np[0,:]), np.ptp(obstacle_np[1,:]), 
            facecolor='gray', alpha=0.5)
        ax.add_patch(rect)
    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (3, 2), order="F")
    ax.set_xlim(X[0,:])
    ax.set_ylim(X[1,:])
    return fig, ax


def render_xs_cartpole(xs, mdp_name, config_dict, fig, ax, color, alpha):
    if fig is None or ax is None:
        fig, ax = make_fig()
    if color is None:
        color = "blue"

    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (5, 2), order="F")
    length_scale = np.ptp(X[0,:])

    cart_height = length_scale / 10.0
    cart_width = length_scale / 5.0
    pole_length = length_scale / 5.0
    pole_radius = length_scale / 50.0
    # alpha = 0.5
    
    pole_xys = cartpole_xs_to_pole_xys(xs, pole_length)
    
    render_cartpole(xs[-1,:], cart_height, cart_width, pole_length, pole_radius, fig, ax, color, alpha)
    ax.plot(pole_xys[:,0], pole_xys[:,1], color=color, alpha=alpha)
    ax.plot(pole_xys[0,0], pole_xys[0,1], marker="o", color=color, alpha=alpha)
    ax.plot(pole_xys[-1,0], pole_xys[-1,1], marker="s", color=color, alpha=alpha)
    if mdp_name in ["Regulator", "Discrete", "DOTS"]:
        xg = config_dict["ground_mdp_xd"]
        pole_xgs = cartpole_xs_to_pole_xys(np.array(xg)[np.newaxis,:], pole_length)
        ax.plot(pole_xgs[0,0], pole_xgs[0,1], marker="*", color="green")
        # if config_dict["ground_mdp_obstacles_on"]:
        # 	for obstacle in config_dict["ground_mdp_obstacles"]:
        # 		obstacle_np = np.reshape(np.array(obstacle), (3,2), order="F")
        # 		rect = Rectangle((obstacle_np[0,0], obstacle_np[1,0]), np.ptp(obstacle_np[0,:]), np.ptp(obstacle_np[1,:]), 
        # 			facecolor='gray', alpha=0.5)
        # 		ax.add_patch(rect)
    # ax.set_aspect('square')
    ax.set_aspect(1.0)
    ax.set_xlim(X[0,:])
    ax.set_ylim(X[0,:])
    return fig, ax	


def render_cartpole(x, cart_height, cart_width, pole_length, pole_radius, fig, ax, color, alpha):
    # plot cart
    cart = Rectangle((x[0] - cart_width / 2.0, - cart_height / 2.0), 
        cart_width, cart_height, facecolor=color, alpha=alpha)
    ax.add_patch(cart)
    # plot pole 
    ax.plot([x[0], x[0] + pole_length * np.sin(x[1])], [0, -pole_length * np.cos(x[1])], color=color, alpha=alpha)
    circ = Circle((x[0] + pole_length * np.sin(x[1]), -pole_length * np.cos(x[1])), radius=pole_radius, color=color, alpha=alpha)
    ax.add_patch(circ)
    return fig, ax


def render_xs_sixdofaircraft(xs, mdp_name, config_dict, fig, ax, color, alpha, obstacles=None, view=None, goal_on=False, observation_cone_on=False):

    robot_on = True
    obstacles_on = True
    thermals_on = config_dict["aero_mode"] in ["neural_thermal", "neural_thermal_moment"] or config_dict["wind_mode"] == "thermal"
    bounds_on = False
    # goal_on = config_dict["reward_mode"] == "regulation"
    observation_cone_on = config_dict["reward_mode"] == "observation" or config_dict["ground_mdp_name"] == "GameSixDOFAircraft"

    if fig is None or ax is None:
        fig, ax = make_3d_fig()

    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    if color is None:
        color = "blue"

    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
    
    # always plot trajectory 
    ax.plot(xs[:,0], xs[:,1], -1 * xs[:,2], color=color, alpha=alpha)
    ax.scatter(xs[0,0], xs[0,1], -1 * xs[0,2], marker="o", color=color, alpha=alpha)
    ax.scatter(xs[-1,0], xs[-1,1], -1 * xs[-1,2], marker="s", color=color, alpha=alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if robot_on:
        robot_alpha = 0.125
        length_scale = np.ptp(X[0,:])
        body_radius = length_scale / 5.0
        num_robots = 2
        idxs = np.linspace(0, xs.shape[0]-1, num=num_robots, endpoint=True, dtype=int)
        for idx in idxs:
            render_sixdofaircraft(xs[idx,:], config_dict, body_radius, fig, ax, color, robot_alpha)

    if obstacles_on:
        from util.util import get_obstacles, get_thermals
        if obstacles is None:
            obstacles = get_obstacles(config_dict, xs[-1][-1])

        state_lims_color = "black"
        state_lims_alpha = 0.1
        state_lims = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
        state_lims[2,:] = -state_lims[2,:]
        state_lims = state_lims[0:3,:]
        # plot_3d_cube(state_lims, fig, ax, state_lims_color, state_lims_alpha)

        obstacle_color = "black"
        obstacle_alpha = 0.1
        for obstacle in obstacles:
            obstacle = np.reshape(np.array(obstacle), (13, 2), order="F")
            obstacle[2,:] = -obstacle[2,:]
            plot_3d_cube(obstacle, fig, ax, obstacle_color, obstacle_alpha)

    if thermals_on:
        from util.util import get_obstacles, get_thermals
        thermals = get_thermals(config_dict, xs[-1][-1])

        thermal_color = "orange"
        thermal_alpha = 0.1
        for X_thermal, V_thermal in thermals:
            X_thermal = X_thermal[0:3,:]
            X_thermal[2,:] = -X_thermal[2,:]
            plot_3d_cube(X_thermal, fig, ax, thermal_color, thermal_alpha)

    if observation_cone_on:
        idx = -1
        cone_x = xs[idx]
        cone_center = np.array([cone_x[0], cone_x[1], -1 * cone_x[2]])[:,np.newaxis]
        cone_radius = config_dict["obs_cone_length"] * np.tan(config_dict["obs_cone_angle"])
        cone_color = color
        cone_alpha = 0.125
        plot_cone(cone_center, cone_radius, config_dict["obs_cone_length"], cone_x[6], -1 * cone_x[7], 
            cone_x[8], fig, ax, cone_color, cone_alpha)
        # plot target 
        xg = config_dict["ground_mdp_xd"]
        ax.scatter(xg[0], xg[1], -1 * xg[2], color="green")

        # human danger zone 
        for idx in config_dict["ground_mdp_special_obstacle_idxs"]:
            obstacle_color = "red"
            obstacle_alpha = 0.1
            unsafe_zone = np.reshape(np.copy(np.array(obstacles[idx])), (13, 2), order="F")[0:3,:]
            unsafe_radius = config_dict["ground_mdp_special_obstacle_radius"]
            unsafe_x = (unsafe_zone[0,0] + unsafe_zone[0,1])/2
            unsafe_y = (unsafe_zone[1,0] + unsafe_zone[1,1])/2      
            plot_cylinder(np.array([unsafe_x, unsafe_y, -unsafe_zone[2,1]])[:,np.newaxis], 
                          unsafe_radius, unsafe_zone[2,1] - unsafe_zone[2,0], fig, ax, obstacle_color, obstacle_alpha)

    if goal_on:
        xg = config_dict["ground_mdp_xd"]
        render_sixdofaircraft(xg, config_dict, body_radius, fig, ax, "green", robot_alpha)

    if bounds_on:
        ax.set_xlim(X[0,:])
        ax.set_ylim(X[1,:])
        ax.set_zlim([-1 * X[2,1], -1 * X[2,0]])
        set_axes_equal(ax)
    else:
        set_axes_equal(ax)

    return fig, ax


def render_xs_sixdofaircraft_game(xs, mdp_name, config_dict, fig, ax, color, alpha, obstacles_on=False, thermals_on=True, view=None):

    robot_on = True
    thermals_on = thermals_on and (config_dict["aero_mode"] in ["neural_thermal", "neural_thermal_moment"] \
        or config_dict["wind_mode"] in ["thermal", "analytical_thermal"])
    bounds_on = False
    plot_bounds_on = True
    observation_cone_on = True

    if fig is None or ax is None:
        fig, ax = make_3d_fig()

    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    if color is None:
        color = "blue"

    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
    
    # always plot trajectory 
    ax.plot(xs[:,0], xs[:,1], -1 * xs[:,2], color=color, alpha=alpha)
    # ax.scatter(xs[0,0], xs[0,1], -1 * xs[0,2], marker="o", color=color, alpha=alpha)
    # ax.scatter(xs[-1,0], xs[-1,1], -1 * xs[-1,2], marker="s", color=color, alpha=alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    if robot_on:
        robot_alpha = 0.125
        length_scale = np.ptp(X[0,:])
        body_radius = length_scale / 2.0
        num_robots = 1
        idxs = np.linspace(0, xs.shape[0]-1, num=num_robots, endpoint=True, dtype=int)
        for idx in idxs:
            render_sixdofaircraft(xs[idx,:], config_dict, body_radius, fig, ax, color, robot_alpha)

    state_lims_color = "black"
    state_lims_alpha = 0.1
    state_lims = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
    state_lims[2,:] = -state_lims[2,:]
    state_lims = state_lims[0:3,:]
    if plot_bounds_on:
        # plot_3d_cube(state_lims, fig, ax, state_lims_color, state_lims_alpha)
        ax.set_xlim([state_lims[0,0], state_lims[0,1]])
        ax.set_ylim([state_lims[1,0], state_lims[1,1]])
        ax.set_zlim([state_lims[2,0], state_lims[2,1]])
        
    if obstacles_on:
        from util.util import get_obstacles, get_thermals
        obstacles = get_obstacles(config_dict, xs[-1][-1])


        obstacle_color = "black"
        obstacle_alpha = 0.1
        for obstacle in obstacles:
            obstacle = np.reshape(np.array(obstacle), (13, 2), order="F")
            obstacle[2,:] = -obstacle[2,:]
            plot_3d_cube(obstacle, fig, ax, obstacle_color, obstacle_alpha)

    if thermals_on:
        from util.util import get_obstacles, get_thermals
        thermals = get_thermals(config_dict, xs[-1][-1])

        thermal_color = "orange"
        thermal_alpha = 0.1
        for X_thermal, V_thermal in thermals:
            X_thermal = X_thermal[0:3,:]
            X_thermal[2,:] = -X_thermal[2,:]
            plot_3d_cube(X_thermal, fig, ax, thermal_color, thermal_alpha)

    if observation_cone_on:
        idx = -1
        cone_x = xs[idx]
        cone_center = np.array([cone_x[0], cone_x[1], -1 * cone_x[2]])[:,np.newaxis]
        cone_radius = config_dict["obs_cone_length"] * np.tan(config_dict["obs_cone_angle"])
        cone_color = color
        cone_alpha = 0.125
        plot_cone(cone_center, cone_radius, config_dict["obs_cone_length"], cone_x[6], -1 * cone_x[7], 
            cone_x[8], fig, ax, cone_color, cone_alpha)

        # human danger zone 
        for idx in config_dict["ground_mdp_special_obstacle_idxs"]:
            obstacle_color = "red"
            obstacle_alpha = 0.1
            unsafe_zone = np.reshape(np.copy(np.array(obstacles[idx])), (13, 2), order="F")[0:3,:]
            unsafe_radius = config_dict["ground_mdp_special_obstacle_radius"]
            unsafe_x = (unsafe_zone[0,0] + unsafe_zone[0,1])/2
            unsafe_y = (unsafe_zone[1,0] + unsafe_zone[1,1])/2      
            plot_cylinder(np.array([unsafe_x, unsafe_y, -unsafe_zone[2,1]])[:,np.newaxis], 
                          unsafe_radius, unsafe_zone[2,1] - unsafe_zone[2,0], fig, ax, obstacle_color, obstacle_alpha)

    for target in config_dict["targets"]:
        ax.scatter(target[0], target[1], -1 * target[2], color="green")

    if bounds_on:
        ax.set_xlim(X[0,:])
        ax.set_ylim(X[1,:])
        ax.set_zlim([-1 * X[2,1], -1 * X[2,0]])
        set_axes_equal(ax)
    else:
        set_axes_equal(ax)

    return fig, ax


def get_fixedwing_vertss(scale):
    vertss = []

    # hi 
    configuration = "vtail"
    in_plane_angle = 5 * np.pi / 180 
    fuselage_length = 1.0 * scale
    fuselage_diameter = 0.1 * scale
    nose_to_wing_dist = 0.2 * scale
    wing_span = 2.0 * scale
    wing_chord = 0.2 * scale
    tailplane_span = 0.4 * scale
    rudder_span = 0.4 * scale
    tailplane_chord = 0.1 * scale
    rudder_chord = 0.1 * scale
    rotor_length = 0.1 * scale
    rotor_height = 0.05 * scale
    ruddervator_angle = 30 * np.pi / 180
    ruddervator_length = 0.2 * scale

    # # fuselage
    fuselage_1_xs = [fuselage_length/2, -fuselage_length/2, -fuselage_length/2, fuselage_length/2]
    fuselage_1_ys = [-fuselage_diameter/2, -fuselage_diameter/2, fuselage_diameter/2, fuselage_diameter/2]
    fuselage_1_zs = [0, 0, 0, 0]
    fuselage_1_verts = [list(zip(fuselage_1_xs, fuselage_1_ys, fuselage_1_zs))]
    vertss.append(fuselage_1_verts)

    # wings 
    starboard_wing_xs = [
        fuselage_length / 2 - nose_to_wing_dist, 
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle),
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle) - wing_chord,
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle) - wing_chord,
    ]
    starboard_wing_ys = [0, wing_span / 2, wing_span / 2, 0]
    starboard_wing_zs = [0, 0, 0, 0]
    starboard_wing_verts = [list(zip(starboard_wing_xs, starboard_wing_ys, starboard_wing_zs))]
    vertss.append(starboard_wing_verts) 

    port_wing_xs = [
        fuselage_length / 2 - nose_to_wing_dist, 
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle),
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle) - wing_chord,
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle) - wing_chord,
    ]
    port_wing_ys = [0, -wing_span / 2, -wing_span / 2, 0]
    port_wing_zs = [0, 0, 0, 0]
    port_wing_verts = [list(zip(port_wing_xs, port_wing_ys, port_wing_zs))]
    vertss.append(port_wing_verts)

    if configuration == "standard":
        
        tailplane_xs = [
            -fuselage_length/2, 
            -fuselage_length/2+tailplane_chord, 
            -fuselage_length/2+tailplane_chord, 
            -fuselage_length/2
            ]
        tailplane_ys = [-tailplane_span/2, -tailplane_span/2, tailplane_span/2, tailplane_span/2]
        tailplane_zs = [0, 0, 0, 0]
        tailplane_verts = [list(zip(tailplane_xs, tailplane_ys, tailplane_zs))]
        vertss.append(tailplane_verts)

        rudder_xs = [-fuselage_length/2, -fuselage_length/2 + rudder_chord, -fuselage_length/2 + rudder_chord, -fuselage_length/2]
        rudder_ys = [0, 0, 0, 0]
        rudder_zs = [0, 0, rudder_span/2, rudder_span/2]
        rudder_verts = [list(zip(rudder_xs, rudder_ys, rudder_zs))]
        vertss.append(rudder_verts)

    elif configuration == "vtail":
        
        # Ruddervators
        port_ruddervators_xs = [
            -fuselage_length/2, 
            -fuselage_length/2+ruddervator_length, 
            -fuselage_length/2+ruddervator_length, 
            -fuselage_length/2
            ]
        port_ruddervators_ys = [
            0.0, 
            0.0, 
            ruddervator_length * np.cos(ruddervator_angle), 
            ruddervator_length * np.cos(ruddervator_angle)
            ]
        port_ruddervators_zs = [
            0.0, 
            0.0, 
            ruddervator_length * np.sin(ruddervator_angle), 
            ruddervator_length * np.sin(ruddervator_angle)
            ]
        port_ruddervators_verts = [list(zip(port_ruddervators_xs, port_ruddervators_ys, port_ruddervators_zs))]
        vertss.append(port_ruddervators_verts)

        starboard_ruddervators_xs = [
            -fuselage_length/2, 
            -fuselage_length/2+ruddervator_length, 
            -fuselage_length/2+ruddervator_length, 
            -fuselage_length/2
            ]
        starboard_ruddervators_ys = [
            0.0, 
            0.0, 
            -ruddervator_length * np.cos(ruddervator_angle), 
            -ruddervator_length * np.cos(ruddervator_angle)
            ]
        starboard_ruddervators_zs = [
            0.0, 
            0.0, 
            ruddervator_length * np.sin(ruddervator_angle), 
            ruddervator_length * np.sin(ruddervator_angle)
            ]
        starboard_ruddervators_verts = [list(zip(starboard_ruddervators_xs, starboard_ruddervators_ys, starboard_ruddervators_zs))]
        vertss.append(starboard_ruddervators_verts)

    return vertss


def get_3d_quadrotor_vertss(scale):
    vertss = []

    moment_arm = 0.4 * scale
    rotor_length = 0.2 * scale
    rotor_height = 0.05 * scale

    # center of rotor 
    rotor_centers = np.array([
        (moment_arm, moment_arm),
        (-1 * moment_arm, moment_arm),
        (-1 * moment_arm, -1 * moment_arm),
        (moment_arm, -1 * moment_arm)
    ])

    for (x,y) in rotor_centers:
        rotor_xs = [
            x + rotor_length, 
            x, 
            x - rotor_length, 
            x
            ]
        rotor_ys = [
            y, 
            y + rotor_length, 
            y, 
            y - rotor_length]
        rotor_zs = [rotor_height, rotor_height, rotor_height, rotor_height]
        rotor_verts = [list(zip(rotor_xs, rotor_ys, rotor_zs))]
        vertss.append(rotor_verts)

        rotor_mount_xs = [x, x, x, x]
        rotor_mount_ys = [y, y, y, y]
        rotor_mount_zs = [rotor_height, rotor_height, 0, 0]
        rotor_mount_verts = [list(zip(rotor_mount_xs, rotor_mount_ys, rotor_mount_zs))]
        vertss.append(rotor_mount_verts)

    cross_frame1_xs = [
        rotor_centers[0,0],
        rotor_centers[0,0],
        rotor_centers[1,0],
        rotor_centers[1,0],
    ]
    cross_frame1_ys = [
        rotor_centers[0,1],
        rotor_centers[0,1],
        rotor_centers[2,1],
        rotor_centers[2,1],
    ]
    cross_frame1_zs = [0,0,0,0]
    cross_frame1_verts = [list(zip(cross_frame1_xs, cross_frame1_ys, cross_frame1_zs))]
    vertss.append(cross_frame1_verts)

    cross_frame2_xs = [
        rotor_centers[1,0],
        rotor_centers[1,0],
        rotor_centers[0,0],
        rotor_centers[0,0],
    ]
    cross_frame2_ys = [
        rotor_centers[0,1],
        rotor_centers[0,1],
        rotor_centers[2,1],
        rotor_centers[2,1],
    ]
    cross_frame2_zs = [0,0,0,0]
    cross_frame2_verts = [list(zip(cross_frame2_xs, cross_frame2_ys, cross_frame2_zs))]
    vertss.append(cross_frame2_verts)

    return vertss


def get_front_rotor_vertss(scale):

    # hi 
    configuration = "vtail"
    in_plane_angle = 5 * np.pi / 180 
    fuselage_length = 1.0 * scale
    fuselage_diameter = 0.1 * scale
    nose_to_wing_dist = 0.2 * scale
    wing_span = 2.0 * scale
    wing_chord = 0.2 * scale
    tailplane_span = 0.4 * scale
    rudder_span = 0.4 * scale
    tailplane_chord = 0.1 * scale
    rudder_chord = 0.1 * scale
    rotor_length = 0.1 * scale
    rotor_height = 0.05 * scale
    ruddervator_angle = 30 * np.pi / 180
    ruddervator_length = 0.2 * scale

    # wings 
    starboard_wing_xs = [
        fuselage_length / 2 - nose_to_wing_dist, 
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle),
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle) - wing_chord,
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle) - wing_chord,
    ]
    starboard_wing_ys = [0, wing_span / 2, wing_span / 2, 0]
    starboard_wing_zs = [0, 0, 0, 0]
    port_wing_xs = [
        fuselage_length / 2 - nose_to_wing_dist, 
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle),
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle) - wing_chord,
        fuselage_length / 2 - nose_to_wing_dist - wing_span / 2 * np.tan(in_plane_angle) - wing_chord,
    ]
    port_wing_ys = [0, -wing_span / 2, -wing_span / 2, 0]
    port_wing_zs = [0, 0, 0, 0]

    vertss = []
    # rotors 
    forward_thrust_rotor_xs = [fuselage_length/2, fuselage_length/2, fuselage_length/2, fuselage_length/2]
    # forward_thrust_rotor_ys = [rotor_length, -rotor_length, -rotor_length, rotor_length]
    # forward_thrust_rotor_zs = [-rotor_length, -rotor_length, rotor_length, rotor_length]
    forward_thrust_rotor_ys = [rotor_length, 0, -rotor_length, 0]
    forward_thrust_rotor_zs = [0, -rotor_length, 0, rotor_length]
    forward_thrust_rotor_verts = [list(zip(forward_thrust_rotor_xs, forward_thrust_rotor_ys, forward_thrust_rotor_zs))]
    vertss.append(forward_thrust_rotor_verts)
    return vertss


def render_sixdofaircraft(x, config_dict, body_radius, fig, ax, color, alpha):

    # shape parameters
    scale = body_radius

    vertss = []
    if config_dict["flight_mode"] == "quadrotor":
        vertss.extend(get_3d_quadrotor_vertss(scale))
    if config_dict["aero_mode"] in ["linear", "nonlinear", "neural", "linear_with_thermal"]:
        scale = scale / 5
        vertss.extend(get_fixedwing_vertss(scale))
    elif config_dict["flight_mode"] == "transition":
        vertss.extend(get_3d_quadrotor_vertss(scale))
        vertss.extend(get_fixedwing_vertss(scale))
        vertss.extend(get_front_rotor_vertss(scale))

    transformed_vertss = [transform_sixdofaircraft_verts(x, verts) for verts in vertss]
    add_sixdofaircraft_vertss(transformed_vertss, ax, color, alpha)
    return None


def transform_sixdofaircraft_verts(x, verts):
    p = (x[0], x[1], -1 * x[2])
    phi, theta, psi = x[6], -1 * x[7], x[8] 

    # rot_mat_body_to_inertial
    R = np.array([
        [np.cos(theta) * np.cos(psi), 
        np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
        np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
        [np.cos(theta) * np.sin(psi),
        np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
        np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
        [-np.sin(theta), 
        np.sin(phi) * np.cos(theta),
        np.cos(phi) * np.cos(theta)] ])

    transformed_verts = []
    for vert in verts: 
        transformed_vert = []
        for point in vert:
            transformed_point = (R @ np.array(point) + p).tolist()
            transformed_vert.append(transformed_point)
        transformed_verts.append(transformed_vert)
    return transformed_verts


def add_sixdofaircraft_vertss(vertss, ax, color, alpha):
    # transform
    for verts in vertss:
        ax.add_collection3d(Poly3DCollection(verts, color=color, alpha=alpha)) 


# render tree 
def render_tree(tree, mdp_name, config_dict, fig=None, ax=None, color=None, alpha=None):
    render_trajs(tree.trajs, mdp_name, config_dict, fig=None, ax=None, color=None, alpha=None)


def render_trajs(trajs, mdp_name, config_dict, fig=None, ax=None, color=None, alpha=None):

    # num_states, num_nodes = util.print_tree_stats(tree)
    # if num_states > 50000: 
    # 	print("skipping tree render")
    # 	return None, None

    # tree is np array in (num_traj, H, n)
    if "SixDOFAircraft" in config_dict["ground_mdp_name"]:
        return render_tree_sixdofaircraft(trajs, mdp_name, config_dict, fig, ax, color, alpha)
    else:
        raise NotImplementedError("render_tree not implemented for ground_mdp_name: {}".format(config_dict["ground_mdp_name"]))




def render_tree_singleintegrator2d(tree, mdp_name, config_dict, fig, ax):
    # tree is list of trajectory objects 

    if fig is None or ax is None:
        fig, ax = make_fig()
    linewidth = 0.5
    color = "black"
    alpha = 0.25
    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (3, 2), order="F")
    ax.set_xlim(X[0,:])
    ax.set_ylim(X[1,:])

    x0 = tree.root 
    segments = [] 
    for traj in tree.trajs:
        xs = np.array(traj.xs) # (H, n)
        segments.append([x0[0:2], xs[0,0:2]])
        segments.extend([[xs[k,0:2], xs[k+1,0:2]] for k in range(xs.shape[0]-1)])
    add_2d_segments(segments, fig, ax, linewidth, color, alpha)

    render_xs(x0[np.newaxis,:], mdp_name, config_dict, fig=fig, ax=ax)
    
    return fig, ax


def render_tree_cartpole(tree, mdp_name, config_dict, fig, ax):
    # tree is list of trajectory objects 

    if fig is None or ax is None:
        fig, ax = make_fig()

    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (5, 2), order="F")
    length_scale = np.ptp(X[0,:])

    pole_length = length_scale / 5.0
    linewidth = 0.25
    color = "black"
    alpha = 0.1257

    x0 = tree.root 
    pole_x0s = cartpole_xs_to_pole_xys(np.array(x0)[np.newaxis,:], pole_length)
    segments = [] 
    for traj in tree.trajs:
        xs = np.array(traj.xs) # (H, n)
        pole_xys = cartpole_xs_to_pole_xys(xs, pole_length)
        segments.append([pole_x0s[0,0:2], pole_xys[0,0:2]])
        segments.extend([[pole_xys[k,0:2], pole_xys[k+1,0:2]] for k in range(pole_xys.shape[0]-1)])
    add_2d_segments(segments, fig, ax, linewidth, color, alpha)

    render_xs(x0[np.newaxis,:], mdp_name, config_dict, fig=fig, ax=ax)
    
    return fig, ax


def render_tree_planarquadrotor(tree, mdp_name, config_dict, fig, ax, color, alpha):
    # tree is list of trajectory objects 

    if fig is None or ax is None:
        fig, ax = make_fig()

    if color is None:
        color = "black"

    if alpha is None:
        alpha = 0.125

    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (7, 2), order="F")
    linewidth = 0.25

    # extract nodes
    if mdp_name in ["DOTS"]:
        node_xs = [node_state[0] for node_state in tree.node_states]
        node_ys = [node_state[1] for node_state in tree.node_states]
        ax.scatter(node_xs, node_ys, marker="o", s=2, color=color, alpha=alpha)

    x0 = tree.root 
    segments = [] 
    for traj in tree.trajs:
        # segments
        xs = np.array(traj.xs) # (H, n)
        segments.append([x0[0:2], xs[0,0:2]])
        segments.extend([[xs[k,0:2], xs[k+1,0:2]] for k in range(xs.shape[0]-1)])
    add_2d_segments(segments, fig, ax, linewidth, color, alpha)
    # render_xs(x0[np.newaxis,:], mdp_name, config_dict, fig=fig, ax=ax)
    return fig, ax


def render_tree_sixdofaircraft(trajs, mdp_name, config_dict, fig, ax, color, alpha):
    # tree is list of trajectory objects 

    if fig is None or ax is None:
        fig, ax = make_fig()

    if color is None:
        color = "black"

    if alpha is None:
        # alpha = 0.15
        alpha = 0.25

    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")
    # linewidth = 0.5
    linewidth = 1.0
    nodesize = 1.0

    segments = [] 
    nodes = []
    for traj in trajs:
        # segments
        xs = np.array(traj) # (H, n)
        # print("xs.shape",xs.shape)
        if len(xs.shape) > 0:
            xs[:,2] = -1 * xs[:,2]
            nodes.append(xs[0,:])
            segments.extend([[xs[k,0:3], xs[k+1,0:3]] for k in range(xs.shape[0]-1) if not np.isnan(np.sum(xs[k]))])
    add_3d_segments(segments, fig, ax, linewidth, color, alpha)

    nodes = np.array(nodes)
    # ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2], color=color, alpha=alpha, s=nodesize)
    ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2], color=color, alpha=alpha, s=nodesize)

    return fig, ax


def render_node_states(config_dict, node_states, fig, ax, color, alpha):
    if (len(node_states) == 0):
        return 
    if config_dict["ground_mdp_name"] in ["SingleIntegrator2d"]:
        node_xs = [node_state[0] for node_state in node_states]
        node_ys = [node_state[1] for node_state in node_states]
        ax.scatter(node_xs, node_ys, marker="o", s=2, color=color, alpha=alpha)
    elif config_dict["ground_mdp_name"] in ["SixDOFAircraft"]:
        node_xs = [node_state[0] for node_state in node_states]
        node_ys = [node_state[1] for node_state in node_states]
        node_zs = [-1 * node_state[2] for node_state in node_states]
        ax.scatter(node_xs, node_ys, node_zs, marker="o", s=2, color=color, alpha=alpha)


# some util 
def cartpole_xs_to_pole_xys(xs, pole_length):
    # xs is np in (t, n)
    # pole_xys is np in (t, 2)
    pole_xys = np.vstack(( xs[:,0] + pole_length * np.sin(xs[:,1]), -pole_length * np.cos(xs[:,1]) )).T
    return pole_xys


def rot_2d_matrix(th):
    # rotates about +z direction 
    R = np.array([
        [np.cos(th), -np.sin(th)],
        [np.sin(th),  np.cos(th)]
        ])
    return R


# plot modes 
def plot_modes(xs, us, config_path, config_dict, fig=None, ax=None, color=None, alpha=None):
    if "PlanarQuadrotor" in config_dict["ground_mdp_name"]:
        return plot_modes_planarquadrotor(xs, us, config_dict, fig, ax, color, alpha)
    elif "SixDOFAircraft" in config_dict["ground_mdp_name"]:
        return plot_modes_sixdofaircraft(xs, us, config_path, fig, ax, color, alpha)
    else:
        raise NotImplementedError("render_tree not implemented for ground_mdp_name: {}".format(config_dict["ground_mdp_name"]))


def plot_modes_sixdofaircraft(xs, us, config_path, fig, ax, color, alpha):
    # plots the spectrum of the grammian 

    import time as timer

    # subsample xs if many xs 
    idxs = np.linspace(0, xs.shape[0]-1, 5, dtype=int)

    for idx in idxs:

        x = xs[idx]

        start_time = timer.time()
        W = compute_reachability_gramian(x, config_path)
        print("wct compute_reachability_gramian: {}".format(timer.time()-start_time))

        fig, ax = make_fig(nrows=1,ncols=2)

        fig.suptitle("idx: {}".format(idx))

        eigenValues, eigenVectors = np.linalg.eigh(W)
        sort_idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[sort_idx]
        eigenVectors = eigenVectors[:,sort_idx]

        ax[0,0].plot(eigenValues)
        ax[0,0].set_xlabel("Mode Index")
        ax[0,0].set_ylabel(r"$\lambda$")

        im = ax[0,1].imshow(eigenVectors)
        ax[0,1].set_xlabel("Mode Index")
        ax[0,1].set_ylabel("State Dimension")

        fig.colorbar(im)


def plot_modes_planarquadrotor(xs, us, config_dict, fig=None, ax=None, color=None, alpha=None):

    # plot settings 
    actual_color = "blue"
    particular_color = "red"
    cumulative_color = "green"

    n = xs.shape[1]
    H = xs.shape[0]

    X = xs[0:-1,:].T # (n,H-1)
    Xp = xs[1:,:].T # (n,H-1)
    pinv_X = np.linalg.pinv(X) # (H-1, n)
    A = Xp @ pinv_X

    # tmp 
    # A = (A + A.T)/2

    eigenValues, eigenVectors = np.linalg.eig(A)
    # eigenValues, eigenVectors = np.linalg.eigh(A)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    pinv_eigenVectors = np.linalg.pinv(eigenVectors)

    np.set_printoptions(precision=3)
    for ii in range(n):
        print("ii: ", ii)
        print("eigenValues[ii]: ",eigenValues[ii])
        print("eigenVectors[:,ii]: ",eigenVectors[:,ii])

    def projected_A(mode_number):
        projected_A = np.zeros((n,n), dtype=complex)
        projected_A = eigenVectors @ np.diag(np.concatenate((eigenValues[0:mode_number+1],np.zeros((n-mode_number-1))))) @ pinv_eigenVectors	
        return projected_A

    # columns are: modes 
    # rows are: trajectory, eigenvalue, and eigenvector  
    fig, ax = make_fig(ncols=n,nrows=3)
    for ii in range(n):

        Wii_nan = np.nan * np.zeros((n,n))
        Wii_nan[:,ii] = np.real(eigenVectors[:,ii])

        cumulative_xs = np.array([np.linalg.matrix_power(projected_A(ii), kk) @ xs[0,:] for kk in range(5)])
        # print("cumulative_xs.shape", cumulative_xs.shape)

        # actual trajectory 
        render_xs(xs, config_dict["ground_mdp_name"], config_dict, fig=fig, ax=ax[0,ii], color=actual_color)
        render_xs(cumulative_xs, config_dict["ground_mdp_name"], config_dict, fig=fig, ax=ax[0,ii], color=cumulative_color)
        ax[0,ii].set_xticklabels([])
        ax[0,ii].set_yticklabels([])

        # eigen vectors 
        im = ax[1,ii].imshow(Wii_nan, cmap="bwr") 
        im.set_clim(-1.0,1.0)
        ax[1,ii].set_xticklabels([])
        ax[1,ii].set_yticklabels([])

        # eigen values 
        th = np.arange(0, 2*np.pi, 0.1)
        ax[2,ii].plot(np.cos(th), np.sin(th), color="blue", alpha=0.2)
        ax[2,ii].plot(np.real(eigenValues[ii]), np.imag(eigenValues[ii]), marker="o", color="blue")
        ax[2,ii].plot(np.real(eigenValues[[jj for jj in range(n) if jj != ii]]), 
            np.imag(eigenValues[[jj for jj in range(n) if jj != ii]]), marker="x", alpha=0.2, color="blue", linestyle="None")
        ax[2,ii].set_aspect(1.0)
        ax[2,ii].grid()
        ax[2,ii].set_xticklabels([])
        ax[2,ii].set_yticklabels([])

    w, h = figaspect(3 / n)
    fig.set_figwidth(w)
    fig.set_figheight(h)
    fig.tight_layout()


def plot_rectangle(xy, width, height, fig, ax, color="gray", alpha=0.125):
    rect = Rectangle(xy, width, height, facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    return rect


def plot_state_report(xs, config_dict, fig=None, ax=None, normalized=True, legend_on=True, alpha=1.0):
    if fig is None or ax is None:
        fig, ax = make_fig()
    state_dim = np.array(config_dict["ground_mdp_X"]).shape[0] // 2
    X = np.reshape(np.array(config_dict["ground_mdp_X"]), (state_dim, 2), order="F")
    colors = get_n_colors(X.shape[0])
    label = None
    for ii in range(state_dim-1): # dont plot time 
        if legend_on: label = config_dict["ground_mdp_state_labels"][ii]
        if normalized: ax.plot((xs[:,ii]-X[ii,0])/(X[ii,1]-X[ii,0]), color=colors[ii], label=label, alpha=alpha)
        else: ax.plot(xs[:,ii], color=colors[ii], label=label, alpha=alpha)
    if normalized: ax.set_ylim([-0.1,1.1])
    if legend_on: ax.legend()
    title = "Normalized State Report" if normalized else "State Report"
    ax.set_title(title)
    ax.grid(True)


def plot_control_report(us, config_dict, fig=None, ax=None, normalized=True, legend_on=True, alpha=1.0):
    if fig is None or ax is None:
        fig, ax = make_fig()

    if config_dict["ground_mdp_name"] == "SixDOFAircraft":
        U = np.reshape(np.array(config_dict["ground_mdp_U"]), (8, 2), order="F")
        labels = config_dict["ground_mdp_control_labels"]
        if config_dict["flight_mode"] == "fixed_wing":
            control_dim = 3
            U = U[0:control_dim,:]
            labels = labels[0:control_dim]
        if config_dict["flight_mode"] == "quadrotor":
            control_dim = 4
            U = U[3:7,:]
            labels = labels[3:7]
        if config_dict["flight_mode"] == "transition":
            control_dim = 8
            U = U[:,:]
            labels = labels[:]
    else:
        control_dim = np.array(config_dict["ground_mdp_U"]).shape[0] // 2
        U = np.reshape(np.array(config_dict["ground_mdp_U"]), (control_dim, 2), order="F")
        labels = config_dict["ground_mdp_control_labels"]

    colors = get_n_colors(U.shape[0])
    label = None
    for ii in range(control_dim):
        if legend_on: label = labels[ii]
        if normalized: 
            if U[ii,1]-U[ii,0] > 0:
                ax.plot((us[:,ii]-U[ii,0])/(U[ii,1]-U[ii,0]), color=colors[ii], label=label, alpha=alpha)
            else:
                ax.plot(0.5*np.ones_like(us[:,ii]), color=colors[ii], label=label, alpha=alpha)
        else: ax.plot(us[:,ii], color=colors[ii], label=label, alpha=alpha)
    # if normalized: ax.set_ylim([0,1])
    if normalized: ax.set_ylim([-0.1,1.1])
    if legend_on: ax.legend()
    title = "Normalized Control Report" if normalized else "Control Report"
    ax.set_title(title)
    ax.grid(True)


def plot_histogram(Xs, labels):
    # Xs is np in num_points x vector_dim
    for ii in range(Xs.shape[1]):
        fig, ax = make_fig()
        ax.hist(Xs[:,ii])
        ax.set_title(labels[ii])
        ax.set_xlabel("value")
        ax.set_ylabel("count")


def plot_2d_ellipse(D, w, fig, ax, color, alpha=1.0):

    # parameterize unit ball
    ths = np.arange(-np.pi, np.pi, 0.01) # n,
    vs = [np.array([[np.cos(th)],[np.sin(th)]]) for th in ths]
    vs_arr = np.array(vs)

    # write ellipse coordinates
    ellipse = [D @ v + w for v in vs]
    ellipse_arr = np.array(ellipse).squeeze(axis=2) # n x 2
    
    # plot
    ax.plot(ellipse_arr[:,0], ellipse_arr[:,1], color=color, alpha=alpha)
    return fig, ax


def unit_scalar_to_rgb(scalar, cmap):
    return cmap(scalar)


def plot_tree_topology(rng, tree_topology, config_dict, fig=None, ax=None):
    if fig is None or ax is None: 
        fig, ax = make_fig()

    import time as timer
    import networkx as nx

    # tree topology is matrix 
    # row = [node_idx, parent_idx, num_visits, max_value, is_valid, branch_idx, depth, time_of_expansion]

    cmap = get_cmap("Greens")

    G = nx.DiGraph()

    # large trees are hard to visualzie, so we subsample 
    subsample_modes = [1]
    nlist_subsample, elist_subsample, node_color_subsample = [0], [], [unit_scalar_to_rgb(1.0,cmap)]

    # subsample strategy 1: sample "n" nodes and take their ancestors to root.
    if 0 in subsample_modes:
        subsample_num_nodes = 20 
        idxs = rng.choice(np.arange(tree_topology.shape[0]), size=subsample_num_nodes)
        for idx in idxs:
            curr_idx = idx
            while tree_topology[curr_idx,1] != -1:
                row = tree_topology[curr_idx,:]
                # always add edge
                elist_subsample.append((row[1], row[0]))
                if (row[0] not in nlist_subsample):
                    # add node
                    nlist_subsample.append(row[0])
                    if row[4]:
                        node_color_subsample.append(unit_scalar_to_rgb(row[2] / tree_topology[0,2], cmap))
                    else:
                        node_color_subsample.append("gray")
                curr_idx = int(row[1])

    # take all the 0,1,2nd generations .. limited to |U|^2 nodes 
    elif 1 in subsample_modes:
        for row in tree_topology:
            if row[6] in [0,1,2]:
                elist_subsample.append((row[1], row[0]))
                nlist_subsample.append(row[0])
                if row[4]:
                    node_color_subsample.append(unit_scalar_to_rgb(row[2] / tree_topology[0,2], cmap))
                else:
                    node_color_subsample.append("gray")                

    G.add_nodes_from(nlist_subsample)
    G.add_edges_from(elist_subsample)

    # print("len(nlist_subsample)", len(nlist_subsample))
    start_time = timer.time()
    pos = nx.kamada_kawai_layout(G)
    print("layout time: {}s".format(timer.time() - start_time))

    # default node_size=300
    # default (edge) width=1.0
    start_time = timer.time()
    nx.draw(G, pos, ax=ax, arrows=False, node_color=node_color_subsample, node_size=50, width=0.5, edge_color="gray")
    # add node outlines
    ax.collections[0].set_edgecolor("black") 
    print("draw time: {}s".format(timer.time() - start_time))

    num_unique_nodes = tree_topology.shape[0]
    frac_valid_nodes = tree_topology[tree_topology[:,4]==1,:].shape[0] / tree_topology.shape[0]
    title = "num unique nodes: {} \n frac valid nodes: {}".format(num_unique_nodes, frac_valid_nodes)
    ax.set_title(title)



def plot_nodal_visit_distribution(tree_topology, config_dict, fig=None, ax=None):
    if fig is None or ax is None: 
        fig, ax = make_fig()

    # tree topology is matrix 
    # row = [node_idx, parent_idx, num_visits, max_value, is_valid, branch_idx, depth, time_of_expansion]

    if tree_topology.shape[0] == 0:
        return 

    depths = tree_topology[:,6].astype(np.int)
    max_depth = max(depths)
    max_num_idxs = 10

    data_normalized = np.nan * np.ones((max_depth+1, max_num_idxs))
    data = np.nan * np.ones((max_depth+1, max_num_idxs))
    for ii_depth in range(max_depth+1):
        num_visits = tree_topology[depths==ii_depth,2]
        sorted_num_visits = np.sort(num_visits)[::-1] # in descending order 
        idxs = np.min((max_num_idxs, sorted_num_visits.shape[0]))
        data_normalized[ii_depth,0:idxs] = sorted_num_visits[0:idxs] / sorted_num_visits[0]
        data[ii_depth,0:idxs] = sorted_num_visits[0:idxs]
    cbar = ax.imshow(data_normalized, interpolation="nearest")
    ax.set_aspect("auto")
    ax.set_yticks(np.arange(0,max_depth+1))
    # fig.colorbar(cbar, ax)

    for jj in range(data.shape[0]):
        for ii in range(data.shape[1]):
            if not np.isnan(data[jj,ii]):
                ax.text(ii,jj,int(data[jj,ii]))

    # # Minor ticks
    ax.set_xticks(np.arange(-.5, max_num_idxs, 1), minor=True)
    ax.set_yticks(np.arange(-.5, max_depth+1, 1), minor=True)

    # # Gridlines based on minor ticks
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)


def plot_nodal_branch_idx_distribution(tree_topology, config_dict, fig=None, ax=None):
    if fig is None or ax is None: 
        fig, ax = make_fig()

    # tree topology is matrix 
    # row = [node_idx, parent_idx, num_visits, max_value, is_valid, branch_idx, depth, time_of_expansion]

    if tree_topology.shape[0] == 0:
        return 

    depths = tree_topology[:,6].astype(np.int)
    branch_idxs = tree_topology[:,5].astype(np.int)
    max_depth = max(depths)
    max_branch_idxs = max(branch_idxs)

    data_normalized = np.nan * np.ones((max_depth+1, max_branch_idxs))
    data = np.nan * np.ones((max_depth+1, max_branch_idxs))
    for ii_depth in range(max_depth+1):
        # num_visits = tree_topology[depths==ii_depth,2]
        # sorted_num_visits = np.sort(num_visits)[::-1] # in descending order 

        nodes_h = [idx for idx in range(tree_topology.shape[0]) if tree_topology[idx,2] == ii_depth]
        # print("nodes_h",nodes_h)

        num_visits = np.zeros((max_branch_idxs))
        for ii_branch_idx in range(max_branch_idxs):
            nodes_h_br = [idx for idx in nodes_h if tree_topology[idx,5] == ii_branch_idx]
            num_visits[ii_branch_idx] = len(nodes_h_br)

        for ii_branch_idx in range(max_branch_idxs):
            data_normalized[ii_depth,ii_branch_idx] = num_visits[ii_branch_idx] / np.sum(num_visits)
            data[ii_depth,ii_branch_idx] = num_visits[ii_branch_idx]

    cbar = ax.imshow(data_normalized, interpolation="nearest")
    ax.set_aspect("auto")
    ax.set_yticks(np.arange(0,max_depth+1))
    # fig.colorbar(cbar, ax)

    for jj in range(data.shape[0]):
        for ii in range(data.shape[1]):
            if not np.isnan(data[jj,ii]):
                ax.text(ii,jj,int(data[jj,ii]))

    # # Minor ticks
    ax.set_xticks(np.arange(-.5, max_branch_idxs, 1), minor=True)
    ax.set_yticks(np.arange(-.5, max_depth+1, 1), minor=True)

    # # Gridlines based on minor ticks
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)


def plot_nodal_valid_distribution(tree_topology, config_dict, fig=None, ax=None):
    if fig is None or ax is None: 
        fig, ax = make_fig()

    # tree topology is matrix 
    # row = [node_idx, parent_idx, num_visits, max_value, is_valid, branch_idx, depth, time_of_expansion]

    if tree_topology.shape[0] == 0:
        return 

    depths = tree_topology[:,6].astype(np.int)
    # max_depth = max(depths)
    max_depth = config_dict["uct_max_depth"]

    data = np.zeros((max_depth))
    for ii_depth in range(max_depth):
        idxs = (depths==ii_depth) 
        if len(idxs) > 0:
            data[ii_depth] = np.sum(tree_topology[idxs,4]) / len(idxs)
    ax.plot(data)
    ax.grid()
    # fig.colorbar(cbar, ax)



def plot_simple_result(result, obstacles = None):

    config_dict = util.load_yaml(result["config_path"])

    if config_dict["ground_mdp_name"] in ["SixDOFAircraft", "GameSixDOFAircraft"]:
        return plot_simple_result_sixdof(result, obstacles=obstacles)

    np_rng = np.random.default_rng(result["seed"])

    fig = plt.figure()

    gs = GridSpec(3, 4)
    render_ax = fig.add_subplot(gs[0:3,0:3])
    xs_ax = fig.add_subplot(gs[0,3])
    us_ax = fig.add_subplot(gs[1,3])
    vs_ax = fig.add_subplot(gs[2,3])
        
    xs_np = np.array(result["final_xs"]) # (horizon, state_dim)
    us_np = np.array(result["final_us"]) # (horizon, state_dim)
    rs_np = np.array(result["final_rs"]) # (horizon, state_dim)

    if xs_np.shape[0] > 0:
        render_xs(xs_np, result["config_dict"]["ground_mdp_name"], result["config_dict"], 
            color="blue", alpha=0.5, fig=fig, ax=render_ax)

        if "sixdofaircraft" in result["config_path"]:
            xy_ax.plot(xs_np[:,0], xs_np[:,1])
            xz_ax.plot(xs_np[:,0], -1 * xs_np[:,2])
            yz_ax.plot(xs_np[:,1], -1 * xs_np[:,2])
    
    # failed first expansion
    else:
        x0 = np.array(result["initial_state"])[np.newaxis,:]
        render_xs(x0, result["config_dict"]["ground_mdp_name"], result["config_dict"], 
            color="blue", alpha=0.5, fig=fig, ax=render_ax)

        if "sixdofaircraft" in result["config_path"]:
            xy_ax.plot(x0[:,0], x0[:,1])
            xz_ax.plot(x0[:,0], -1 * x0[:,2])
            yz_ax.plot(x0[:,1], -1 * x0[:,2])


    if "uct" in result["config_dict"]["solver_mode"] and len(result["result_uct"]["xs"])>0:
        # render_node_states(result["config_dict"], result["result_uct"]["node_states"], fig, render_ax, "blue", 0.05)
        # render_xs(np.array(result["result_uct"]["xs"]), result["config_dict"]["ground_mdp_name"], result["config_dict"], 
        #     color="blue", alpha=0.5, fig=fig, ax=render_ax)
        # render_ax.plot(np.nan,np.nan,color="blue",alpha=0.5,label="uct")

        vs_ax.plot(result["result_uct"]["ns"], result["result_uct"]["vs"])
        vs_ax.set_ylabel("V")
        vs_ax.set_xlabel("n")

    if "uct_mpc" in result["config_dict"]["solver_mode"] and len(result["result_uct"]["trajss"])>0:
        # subsampling is done earlier 
        # num = min(5, len(result["result_uct"]["trajss"]))
        # idxs = np.linspace(0, len(result["result_uct"]["trajss"])-1, num=num, endpoint=True, dtype=int)
        
        idxs = list(range(0, len(result["result_uct"]["trajss"])))
        mpc_colors = get_n_colors(len(idxs))
        
        for ii, idx in enumerate(idxs):
            print('result["result_uct"]["trajss"][idx].shape', result["result_uct"]["trajss"][idx].shape)
            render_trajs(result["result_uct"]["trajss"][idx], result["config_dict"]["ground_mdp_name"], result["config_dict"], 
                fig=fig, ax=render_ax, color=mpc_colors[ii], alpha=0.05)

    if result["config_dict"]["solver_mode"] in ["scp", "uct_then_scp"]:
        
        idxs = np.linspace(1, len(result["result_scp"]["tree_xss"])-1, num=10, endpoint=True, dtype=int)
        colors = get_n_colors(len(idxs))
        for jj, idx in enumerate(idxs):
            render_xs(np.array(result["result_scp"]["tree_xss"][idx]), 
                result["config_dict"]["ground_mdp_name"], result["config_dict"], 
            color=colors[jj], alpha=0.5, fig=fig, ax=render_ax)
            render_ax.plot(np.nan,np.nan,color=colors[jj],label="SCP iter: {}".format(idx))
        render_xs(xs_np, result["config_dict"]["ground_mdp_name"], result["config_dict"], fig=fig, ax=render_ax, color="gray", alpha=0.9)
        render_ax.plot(np.nan, np.nan, color="gray", alpha=0.9, label="rollout")

        for jj, idx in enumerate(idxs):
            vs_ax.plot(np.cumsum(result["result_scp"]["tree_rss"][idx]), color=colors[jj])
            vs_ax.plot(np.nan,np.nan,color=colors[jj],label="SCP iter: {}".format(idx))
        vs_ax.set_ylabel("V")
        vs_ax.set_xlabel("t")

    render_ax.legend()

    if "scp_mpc" in result["config_dict"]["solver_mode"]:
        render_xs(np.array(result["result_scp"]["xs"]), result["config_dict"]["ground_mdp_name"], result["config_dict"], fig=fig, ax=render_ax, color="gray", alpha=0.9)

    if xs_np.shape[0] > 0:
        plot_state_report(xs_np, result["config_dict"], fig=fig, ax=xs_ax, normalized=True, legend_on=True, alpha=0.5)
        plot_control_report(us_np, result["config_dict"], fig=fig, ax=us_ax, normalized=True, legend_on=True, alpha=0.5)
        # plot_control_report(us_np, result["config_dict"], fig=fig, ax=us_ax, normalized=False, legend_on=True, alpha=1.0)

    if ("obstacles" in result["result_uct"].keys()):
        last_tstep_obstacles = result["result_uct"]["obstacles"][-1]
        obstacle_color = "black"
        obstacle_alpha = 0.1
        for obstacle in last_tstep_obstacles:
            obstacle = np.reshape(np.array(obstacle), (13, 2), order="F")
            obstacle[2,:] = -obstacle[2,:]
            plot_3d_cube(obstacle, fig, render_ax, obstacle_color, obstacle_alpha)
        for idx in result["config_dict"]["ground_mdp_special_obstacle_idxs"]:
            obstacle_color = "red"
            obstacle_alpha = 0.1
            unsafe_zone = np.reshape(np.copy(np.array(last_tstep_obstacles[idx])), (13, 2), order="F")[0:3,:]
            unsafe_radius = result["config_dict"]["ground_mdp_special_obstacle_radius"]
            unsafe_x = (unsafe_zone[0,0] + unsafe_zone[0,1])/2
            unsafe_y = (unsafe_zone[1,0] + unsafe_zone[1,1])/2 
            plot_cylinder(np.array([unsafe_x, unsafe_y, -unsafe_zone[2,1]])[:,np.newaxis], 
                            unsafe_radius, unsafe_zone[2,1] - unsafe_zone[2,0], fig, render_ax, obstacle_color, obstacle_alpha)
    
    if len(xs_np.shape) > 1:
        total_physical_time = xs_np[-1,-1] * result["config_dict"]["ground_mdp_dt"]
    else:
        total_physical_time = 0
    config_name = os.path.splitext(os.path.basename(result["config_path"]))[0]
    # fig.suptitle("{}, {}".format(config_name, result["seed"]))
    fig.suptitle("{}, {}, WCT: {}, Time: {}".format(config_name, result["seed"], result["total_wct"], total_physical_time))
    figsize = fig.get_size_inches()
    scale_fig(fig, 1)
    gs.update(wspace=0.4, hspace=0.4)


def plot_simple_result_sixdof(result, obstacles=None, second_plot_on=True, stop_idx=None):

    config_dict = util.load_yaml(result["config_path"])
    if "SixDOFAircraft" == config_dict["ground_mdp_name"]:
        render_xs_fn = render_xs_sixdofaircraft
    else:
        render_xs_fn = render_xs_sixdofaircraft_game


    np_rng = np.random.default_rng(result["seed"])

    fig = plt.figure()

    gs = GridSpec(2, 2)

    views = [[30, -60], [90, -90], [0, -90], [0, 0]]    
    ax = fig.add_subplot(gs[0,0], projection="3d")
    xy_ax = fig.add_subplot(gs[0,1], projection="3d")
    xz_ax = fig.add_subplot(gs[1,0], projection="3d")
    yz_ax = fig.add_subplot(gs[1,1], projection="3d")

    ax.set_xlabel("x")
    ax.set_ylabel("x")
    ax.set_zlabel("x")
    xy_ax.set_xlabel("x")
    xy_ax.set_ylabel("y")
    xz_ax.set_xlabel("x")
    xz_ax.set_ylabel("z")
    yz_ax.set_xlabel("y")
    yz_ax.set_ylabel("z")

    xs_np = np.array(result["final_xs"]) # (horizon, state_dim)
    us_np = np.array(result["final_us"]) # (horizon, state_dim)
    rs_np = np.array(result["final_rs"]) # (horizon, state_dim)
    if stop_idx is not None:        
        xs_np = xs_np[0:stop_idx,:] # (horizon, state_dim)
        us_np = us_np[0:stop_idx,:] # (horizon, state_dim)
        rs_np = rs_np[0:stop_idx] # (horizon, state_dim)


    # render initial state
    x0 = np.array(result["initial_state"])[np.newaxis,:]
    render_xs_fn(x0, result["config_dict"]["ground_mdp_name"], result["config_dict"], color="blue", alpha=0.5, fig=fig, ax=ax, view=views[0])
    render_xs_fn(x0, result["config_dict"]["ground_mdp_name"], result["config_dict"], color="blue", alpha=0.5, fig=fig, ax=xy_ax, view=views[1])
    render_xs_fn(x0, result["config_dict"]["ground_mdp_name"], result["config_dict"], color="blue", alpha=0.5, fig=fig, ax=xz_ax, view=views[2])
    render_xs_fn(x0, result["config_dict"]["ground_mdp_name"], result["config_dict"], color="blue", alpha=0.5, fig=fig, ax=yz_ax, view=views[3])


    if xs_np.shape[0] > 0:

        # render initial state
        xf = xs_np[-1,np.newaxis,:]
        render_xs_fn(xf, result["config_dict"]["ground_mdp_name"], result["config_dict"], color="blue", alpha=0.5, fig=fig, ax=ax, view=views[0])
        render_xs_fn(xf, result["config_dict"]["ground_mdp_name"], result["config_dict"], color="blue", alpha=0.5, fig=fig, ax=xy_ax, view=views[1])
        render_xs_fn(xf, result["config_dict"]["ground_mdp_name"], result["config_dict"], color="blue", alpha=0.5, fig=fig, ax=xz_ax, view=views[2])
        render_xs_fn(xf, result["config_dict"]["ground_mdp_name"], result["config_dict"], color="blue", alpha=0.5, fig=fig, ax=yz_ax, view=views[3])

        # plot traj
        ax.plot(xs_np[:,0], xs_np[:,1], -1 * xs_np[:,2])
        xy_ax.plot(xs_np[:,0], xs_np[:,1], -1 * xs_np[:,2])
        xz_ax.plot(xs_np[:,0], xs_np[:,1], -1 * xs_np[:,2])
        yz_ax.plot(xs_np[:,1], xs_np[:,1], -1 * xs_np[:,2])

        if result["config_dict"]["solver_mode"] in ["uct_mpc", "uct_mpc_then_scp"]:
            if len(result["result_uct"]["trajss"]) > 0 :
                idxs = list(range(0, len(result["result_uct"]["trajss"])))
                mpc_colors = get_n_colors(len(idxs))
                for ii, idx in enumerate(idxs):
                    print('result["result_uct"]["trajss"][idx].shape', result["result_uct"]["trajss"][idx].shape)
                    render_trajs(result["result_uct"]["trajss"][idx], result["config_dict"]["ground_mdp_name"], result["config_dict"], 
                        fig=fig, ax=ax, color=mpc_colors[ii], alpha=0.05)

        if result["config_dict"]["solver_mode"] in ["scp", "uct_then_scp", "uct_mpc_then_scp"]:
            if "tree_xss" in result["result_scp"].keys():
                num = np.min((len(result["result_scp"]["tree_xss"])-1, 5))
                idxs = np.linspace(0, len(result["result_scp"]["tree_xss"])-1, num=num, endpoint=True, dtype=int)
                colors = get_n_colors(len(idxs))
                for jj, idx in enumerate(idxs):
                    this_traj_xs = np.array(result["result_scp"]["tree_xss"][idx])
                    ax.plot(this_traj_xs[:,0], this_traj_xs[:,1], -1 * this_traj_xs[:,2], color=colors[jj], label="SCP_iter: {}".format(idx))
                    xy_ax.plot(this_traj_xs[:,0], this_traj_xs[:,1], -1 * this_traj_xs[:,2], color=colors[jj], label="SCP_iter: {}".format(idx))
                    xz_ax.plot(this_traj_xs[:,0], this_traj_xs[:,1], -1 * this_traj_xs[:,2], color=colors[jj], label="SCP_iter: {}".format(idx))
                    yz_ax.plot(this_traj_xs[:,1], this_traj_xs[:,1], -1 * this_traj_xs[:,2], color=colors[jj], label="SCP_iter: {}".format(idx))

        ax.legend()

    else: 
        fig.suptitle("solver failed")

    if ("obstacles" in result["result_uct"].keys()):
        last_tstep_obstacles = result["result_uct"]["obstacles"][-1]
        obstacle_color = "black"
        obstacle_alpha = 0.1
        for obstacle in last_tstep_obstacles:
            obstacle = np.reshape(np.array(obstacle), (13, 2), order="F")
            obstacle[2,:] = -obstacle[2,:]
            plot_3d_cube(obstacle, fig, ax, obstacle_color, obstacle_alpha)
        for idx in result["config_dict"]["ground_mdp_special_obstacle_idxs"]:
            obstacle_color = "red"
            obstacle_alpha = 0.1
            unsafe_zone = np.reshape(np.copy(np.array(last_tstep_obstacles[idx])), (13, 2), order="F")[0:3,:]
            unsafe_radius = result["config_dict"]["ground_mdp_special_obstacle_radius"]
            unsafe_x = (unsafe_zone[0,0] + unsafe_zone[0,1])/2
            unsafe_y = (unsafe_zone[1,0] + unsafe_zone[1,1])/2 
            plot_cylinder(np.array([unsafe_x, unsafe_y, -unsafe_zone[2,1]])[:,np.newaxis], 
                            unsafe_radius, unsafe_zone[2,1] - unsafe_zone[2,0], fig, ax, obstacle_color, obstacle_alpha)
    
    if second_plot_on and xs_np.shape[0] > 0:

        # make another figure with useful things 
        fig = plt.figure()
        gs = GridSpec(4, 6)

        aero_force_ax = fig.add_subplot(gs[0,0])
        aero_moment_ax = fig.add_subplot(gs[0,1])
        vs_ax = fig.add_subplot(gs[0,2])
        rewards_ax = fig.add_subplot(gs[0,3])

        control_dt = result["config_dict"]["ground_mdp_dt"] * result["config_dict"]["ground_mdp_control_hold"]
        # time = control_dt * np.arange(0,xs_np.shape[0])

        # plot aero forces
        ground_mdp = get_mdp(result["config_dict"]["ground_mdp_name"], result["config_path"])
        time = control_dt * xs_np[:,ground_mdp.timestep_idx()] * 10

        if result["config_dict"]["aero_mode"] in ["neural", "neural_thermal", "neural_thermal_moment"]:
            from learning.feedforward import Feedforward
            model = Feedforward(result["config_dict"]["model_device"],
                result["config_dict"]["model_overfit_mode"],
                result["config_dict"]["model_training_mode"],
                result["config_dict"]["model_lipshitz_const"],
                result["config_dict"]["model_train_test_ratio"],
                result["config_dict"]["model_batch_size"],
                result["config_dict"]["model_initial_learning_rate"],
                result["config_dict"]["model_num_hidden_layers"],
                result["config_dict"]["model_num_epochs"],
                result["config_dict"]["model_path"],
                result["config_dict"]["model_input_dim"],
                result["config_dict"]["model_hidden_dim"],
                result["config_dict"]["model_output_dim"])
            weightss, biass = model.extract_ff_layers()
            ground_mdp.set_weights(weightss, biass)
        
        if result["config_dict"]["ground_mdp_name"] == "SixDOFAircraft":
            ground_mdp.clear_thermals() 
            [ground_mdp.add_thermal(X_thermal, V_thermal) for X_thermal, V_thermal in util.get_thermals(result["config_dict"], x0[0,-1])]

            aeros = []
            for kk in range(xs_np.shape[0]):
                aeros.append(wrapper_aero_model(xs_np[kk], us_np[kk], ground_mdp))
            aeros_np = np.array(aeros) # (len_traj, 6)

            aero_force_labels = ["f_x", "f_y", "f_z"] 
            aeros_np[:,2] = -1 * aeros_np[:,2] # flip "z" coordinate
            for ii in range(3):
                aero_force_ax.plot(time, aeros_np[:,ii], label=aero_force_labels[ii])
            aero_force_ax.legend()

            aero_moment_labels = ["m_x", "m_y", "m_z"]
            for ii in range(3,6):
                aero_moment_ax.plot(time, aeros_np[:,ii], label=aero_moment_labels[ii-3])
            aero_moment_ax.legend()
        
        # plot rs over t
        rewards_ax.plot(time, rs_np)
        rewards_ax.legend()
        rewards_ax.set_xlabel("time")
        rewards_ax.set_ylabel("Reward")

        if "uct" in result["config_dict"]["solver_mode"]:
            uct_time = ground_mdp.dt() * np.arange(len(result["result_uct"]["rs"]))
            rewards_ax.plot(uct_time, result["result_uct"]["rs"])

        # plot vs over n
        if len(result["result_uct"]):
            vs_ax.plot(result["result_uct"]["ns"], result["result_uct"]["vs"])

        # next two rows are states with physical units 
        state_axs = []
        state_labels = result["config_dict"]["ground_mdp_state_labels"]
        state_lims = np.reshape(np.array(result["config_dict"]["ground_mdp_X"]), (13, 2), order="F")

        for ii in range(6):
            for jj in range(2):
                idx = jj * 6 + ii
                ax = fig.add_subplot(gs[jj+1,ii])
                ax.plot(time, xs_np[:,idx])
                ax.axhline(state_lims[idx,0], color="gray")
                ax.axhline(state_lims[idx,1], color="gray")
                ax.set_title(state_labels[idx])

        # third row are control inputs with physical units 
        control_axs = []
        control_labels = result["config_dict"]["ground_mdp_control_labels"]
        control_lims = np.reshape(np.array(result["config_dict"]["ground_mdp_U"]), (8, 2), order="F")
        for ii in range(4):
            idx = ii + 3
            ax = fig.add_subplot(gs[3,ii])
            ax.plot(time, us_np[:,ii])
            ax.axhline(control_lims[idx,0], color="gray")
            ax.axhline(control_lims[idx,1], color="gray")
            ax.set_title(control_labels[idx])

        config_name = os.path.splitext(os.path.basename(result["config_path"]))[0]
        fig.suptitle("{}, {}".format(config_name, result["seed"]))
        scale_fig(fig, 1)
        gs.update(wspace=0.4, hspace=0.4)

    return fig, ax


def scale_fig(fig, scale):
    figsize = fig.get_size_inches()
    # print("figsize",figsize)
    fig.set_size_inches( (figsize[0]*scale, figsize[1]*scale) )


def multi_axs_state_report(xs_np, config_dict):

    fig = plt.figure()
    gs = GridSpec(2, 6)        

    if config_dict["ground_mdp_name"] != "SixDOFAircraft":
        return

    control_dt = config_dict["ground_mdp_dt"] * config_dict["ground_mdp_control_hold"]
    time = control_dt * np.arange(0,xs_np.shape[0])

    # last two rows are states with physical units 
    state_axs = []
    state_labels = config_dict["ground_mdp_state_labels"]
    state_lims = np.reshape(np.array(config_dict["ground_mdp_X"]), (13, 2), order="F")

    for ii in range(6):
        for jj in range(2):
            idx = jj * 6 + ii
            ax = fig.add_subplot(gs[jj,ii])
            ax.plot(time, xs_np[:,idx])
            ax.axhline(state_lims[idx,0], color="gray")
            ax.axhline(state_lims[idx,1], color="gray")
            ax.set_title(state_labels[idx])

    return fig, ax


def plot_branchdata_results(results):

    if results[0]["config_dict"]["ground_mdp_name"] == "SixDOFAircraft":
        plot_branchdata_sixdofaircraft_results(results)
    else:
        print("plot_results not implemented for system")
        return 


def plot_branchdata_sixdofaircraft_results(results):

    # sort by initial condition 
    x0s = np.unique([result["xbar"] for result in results], axis=0)
    num_x0s = x0s.shape[0]
    # x0s = sorted(x0s, key=lambda x: x[-1])
    
    sorted_results = [[] for ii in range(num_x0s)]
    for ii, x0 in enumerate(x0s):
        for result in results: 
            if np.linalg.norm(result["xbar"]-x0)<1e-6:
                sorted_results[ii].append(result)
    sorted_results = sorted(sorted_results, key=lambda x: x[0]["depth"])

    colors_x0 = get_n_colors(num_x0s)

    for results in tqdm(sorted_results, desc="plotting branchdata"): 

        # print('results[0]["x0"]', results[0]["x0"])
        # print('results[0]["depth"]', results[0]["depth"])

        # render all trajectories on a master figure 
        fig = plt.figure()
        gs = GridSpec(6, 8)   

        positions_3d_ax = fig.add_subplot(gs[0:4,0:4], projection="3d")
        attitudes_3d_ax = fig.add_subplot(gs[0:4,4:8], projection="3d")
        
        control_dt = results[0]["config_dict"]["ground_mdp_dt"] * results[0]["config_dict"]["ground_mdp_control_hold"]
        
        state_lims = np.reshape(np.array(results[0]["config_dict"]["ground_mdp_X"]), (13, 2), order="F")
        control_lims = np.reshape(np.array(results[0]["config_dict"]["ground_mdp_U"]), (8, 2), order="F")
        
        state_labels = results[0]["config_dict"]["ground_mdp_state_labels"]
        control_labels = results[0]["config_dict"]["ground_mdp_control_labels"]

        # change coordinates 
        state_lims[2,:] = -1 * state_lims[2,:]

        if results[0]["config_dict"]["flight_mode"] != "quadrotor":
            print("plot results not implemented for flight mode")
            return 
        control_lims = control_lims[3:7,:]
        control_labels = control_labels[3:7]

        px_ax = fig.add_subplot(gs[4,0])
        py_ax = fig.add_subplot(gs[4,1])
        pz_ax = fig.add_subplot(gs[4,2])
        vx_ax = fig.add_subplot(gs[5,0])
        vy_ax = fig.add_subplot(gs[5,1])
        vz_ax = fig.add_subplot(gs[5,2])
        ro_ax = fig.add_subplot(gs[4,3])
        pi_ax = fig.add_subplot(gs[4,4])
        ya_ax = fig.add_subplot(gs[4,5])
        rr_ax = fig.add_subplot(gs[5,3])
        pr_ax = fig.add_subplot(gs[5,4])
        yr_ax = fig.add_subplot(gs[5,5])

        px_ax.set_title(r"$p_x$")
        py_ax.set_title(r"$p_y$")
        pz_ax.set_title(r"$p_z$")
        vx_ax.set_title(r"$v_x$")
        vy_ax.set_title(r"$v_y$")
        vz_ax.set_title(r"$v_z$")
        ro_ax.set_title(r"$\phi$")
        pi_ax.set_title(r"$\theta$")
        ya_ax.set_title(r"$\psi$")
        rr_ax.set_title(r"$\dot{\phi}$")
        pr_ax.set_title(r"$\dot{\theta}$")
        yr_ax.set_title(r"$\dot{\psi}$")

        th_ax = fig.add_subplot(gs[4,6])
        tx_ax = fig.add_subplot(gs[4,7])
        ty_ax = fig.add_subplot(gs[5,6])
        tz_ax = fig.add_subplot(gs[5,7])

        th_ax.set_title(r"$f_{th}$")
        tx_ax.set_title(r"$\tau_x$")
        ty_ax.set_title(r"$\tau_y$")
        tz_ax.set_title(r"$\tau_z$")

        # # render 
        # body_radius = 0.05
        # robot_alpha = 0.5
        # x0_color = "gray"
        # xbar = results[0]["xbar"]
        # xbar[2] = -1 * xbar[2]
        # render_sixdofaircraft(xbar, results[0]["config_dict"], body_radius, fig, positions_3d_ax, x0_color, robot_alpha)
        positions_3d_ax.set_xlim(state_lims[0,:])
        positions_3d_ax.set_ylim(state_lims[1,:])
        positions_3d_ax.set_zlim(state_lims[2,:])

        colors = get_n_colors(len(results))
        for ii, result in enumerate(results): 

            # if result["branch_idx"] != 2:
            # if not result["branch_idx"] in [0,2]:
            #     continue 

            xs_np = np.array(result["xs"])
            us_np = np.array(result["us"])
            zs_np = np.array(result["zs_ref"])
            z_unscaled = result["delta_z_H_unscaled"] + result["zbar_H"]
            z_scaled = result["delta_z_H"] + result["zbar_H"]

            if ii == 0:
                render_xs(xs_np, result["config_dict"]["ground_mdp_name"], result["config_dict"], fig=fig, ax=positions_3d_ax)

            # change coordinates
            xs_np[:,2] = -1 * xs_np[:,2]
            zs_np[:,2] = -1 * zs_np[:,2]
            z_unscaled[2] = -1 * z_unscaled[2]
            z_scaled[2] = -1 * z_scaled[2]

            if result["is_valid"]:
                color = colors[ii]
                alpha = 0.5
            else:
                color = "gray"
                alpha = 0.2

            if result["branch_idx"] == 0:
                color="blue"
                alpha = 0.5


            positions_3d_ax.plot(xs_np[:,0], xs_np[:,1], xs_np[:,2], color=color, alpha=alpha)
            positions_3d_ax.plot(zs_np[:,0], zs_np[:,1], zs_np[:,2], color=color, alpha=alpha, linestyle="dashed")
            positions_3d_ax.plot(np.nan*np.ones([1]), np.nan*np.ones([1]), np.nan*np.ones([1]), 
                                 color=color, alpha=alpha, label=result["branch_idx"])

            # render_sixdofaircraft(xs_np[-1,:], result["config_dict"], body_radius, fig, positions_3d_ax, color, robot_alpha)

            attitudes_3d_ax.plot(xs_np[:,6], xs_np[:,7], xs_np[:,8], color=color, alpha=alpha)
            attitudes_3d_ax.plot(zs_np[:,6], zs_np[:,7], zs_np[:,8], color=color, alpha=alpha, linestyle="dashed")

            x_time = control_dt * np.arange(0,xs_np.shape[0])
            z_time = control_dt * np.arange(0,zs_np.shape[0])

            px_ax.plot(x_time, xs_np[:,0], alpha=alpha, color=color)
            py_ax.plot(x_time, xs_np[:,1], alpha=alpha, color=color)
            pz_ax.plot(x_time, xs_np[:,2], alpha=alpha, color=color)
            vx_ax.plot(x_time, xs_np[:,3], alpha=alpha, color=color)
            vy_ax.plot(x_time, xs_np[:,4], alpha=alpha, color=color)
            vz_ax.plot(x_time, xs_np[:,5], alpha=alpha, color=color)
            ro_ax.plot(x_time, xs_np[:,6], alpha=alpha, color=color)
            pi_ax.plot(x_time, xs_np[:,7], alpha=alpha, color=color)
            ya_ax.plot(x_time, xs_np[:,8], alpha=alpha, color=color)
            rr_ax.plot(x_time, xs_np[:,9], alpha=alpha, color=color)
            pr_ax.plot(x_time, xs_np[:,10], alpha=alpha, color=color)
            yr_ax.plot(x_time, xs_np[:,11], alpha=alpha, color=color)

            px_ax.plot(z_time, zs_np[:,0], alpha=alpha, color=color, linestyle="dashed")
            py_ax.plot(z_time, zs_np[:,1], alpha=alpha, color=color, linestyle="dashed")
            pz_ax.plot(z_time, zs_np[:,2], alpha=alpha, color=color, linestyle="dashed")
            vx_ax.plot(z_time, zs_np[:,3], alpha=alpha, color=color, linestyle="dashed")
            vy_ax.plot(z_time, zs_np[:,4], alpha=alpha, color=color, linestyle="dashed")
            vz_ax.plot(z_time, zs_np[:,5], alpha=alpha, color=color, linestyle="dashed")
            ro_ax.plot(z_time, zs_np[:,6], alpha=alpha, color=color, linestyle="dashed")
            pi_ax.plot(z_time, zs_np[:,7], alpha=alpha, color=color, linestyle="dashed")
            ya_ax.plot(z_time, zs_np[:,8], alpha=alpha, color=color, linestyle="dashed")
            rr_ax.plot(z_time, zs_np[:,9], alpha=alpha, color=color, linestyle="dashed")
            pr_ax.plot(z_time, zs_np[:,10], alpha=alpha, color=color, linestyle="dashed")
            yr_ax.plot(z_time, zs_np[:,11], alpha=alpha, color=color, linestyle="dashed")

            # px_ax.plot(z_time[-1], z_unscaled[0], marker="o", alpha=alpha, color=color)
            # py_ax.plot(z_time[-1], z_unscaled[1], marker="o", alpha=alpha, color=color)
            # pz_ax.plot(z_time[-1], z_unscaled[2], marker="o", alpha=alpha, color=color)
            # vx_ax.plot(z_time[-1], z_unscaled[3], marker="o", alpha=alpha, color=color)
            # vy_ax.plot(z_time[-1], z_unscaled[4], marker="o", alpha=alpha, color=color)
            # vz_ax.plot(z_time[-1], z_unscaled[5], marker="o", alpha=alpha, color=color)
            # ro_ax.plot(z_time[-1], z_unscaled[6], marker="o", alpha=alpha, color=color)
            # pi_ax.plot(z_time[-1], z_unscaled[7], marker="o", alpha=alpha, color=color)
            # ya_ax.plot(z_time[-1], z_unscaled[8], marker="o", alpha=alpha, color=color)
            # rr_ax.plot(z_time[-1], z_unscaled[9], marker="o", alpha=alpha, color=color)
            # pr_ax.plot(z_time[-1], z_unscaled[10], marker="o", alpha=alpha, color=color)
            # yr_ax.plot(z_time[-1], z_unscaled[11], marker="o", alpha=alpha, color=color)

            px_ax.scatter(z_time[-1], z_scaled[0], marker="s", alpha=alpha, color=color, s=2)
            py_ax.scatter(z_time[-1], z_scaled[1], marker="s", alpha=alpha, color=color, s=2)
            pz_ax.scatter(z_time[-1], z_scaled[2], marker="s", alpha=alpha, color=color, s=2)
            vx_ax.scatter(z_time[-1], z_scaled[3], marker="s", alpha=alpha, color=color, s=2)
            vy_ax.scatter(z_time[-1], z_scaled[4], marker="s", alpha=alpha, color=color, s=2)
            vz_ax.scatter(z_time[-1], z_scaled[5], marker="s", alpha=alpha, color=color, s=2)
            ro_ax.scatter(z_time[-1], z_scaled[6], marker="s", alpha=alpha, color=color, s=2)
            pi_ax.scatter(z_time[-1], z_scaled[7], marker="s", alpha=alpha, color=color, s=2)
            ya_ax.scatter(z_time[-1], z_scaled[8], marker="s", alpha=alpha, color=color, s=2)
            rr_ax.scatter(z_time[-1], z_scaled[9], marker="s", alpha=alpha, color=color, s=2)
            pr_ax.scatter(z_time[-1], z_scaled[10], marker="s", alpha=alpha, color=color, s=2)
            yr_ax.scatter(z_time[-1], z_scaled[11], marker="s", alpha=alpha, color=color, s=2)

            th_ax.plot(x_time, us_np[:,0], color=color, alpha=alpha)
            tx_ax.plot(x_time, us_np[:,1], color=color, alpha=alpha)
            ty_ax.plot(x_time, us_np[:,2], color=color, alpha=alpha)
            tz_ax.plot(x_time, us_np[:,3], color=color, alpha=alpha)

        # # plot to root 
        path_to_root = np.array(results[0]["path_to_root"], ndmin=2) # (t, n)
        if path_to_root.shape[1]>0:
            path_to_root[:,2] = -1 * path_to_root[:,2]
            positions_3d_ax.plot(path_to_root[:,0], path_to_root[:,1], path_to_root[:,2], color="black", alpha=0.5)

        positions_3d_ax.legend()
        positions_3d_ax.grid()
        attitudes_3d_ax.grid()

        px_ax.axhline(state_lims[0,0], color="gray")
        px_ax.axhline(state_lims[0,1], color="gray")
        py_ax.axhline(state_lims[1,0], color="gray")
        py_ax.axhline(state_lims[1,1], color="gray")
        pz_ax.axhline(state_lims[2,0], color="gray")
        pz_ax.axhline(state_lims[2,1], color="gray")
        vx_ax.axhline(state_lims[3,0], color="gray")
        vx_ax.axhline(state_lims[3,1], color="gray")
        vy_ax.axhline(state_lims[4,0], color="gray")
        vy_ax.axhline(state_lims[4,1], color="gray")
        vz_ax.axhline(state_lims[5,0], color="gray")
        vz_ax.axhline(state_lims[5,1], color="gray")
        ro_ax.axhline(state_lims[6,0], color="gray")
        ro_ax.axhline(state_lims[6,1], color="gray")
        pi_ax.axhline(state_lims[7,0], color="gray")
        pi_ax.axhline(state_lims[7,1], color="gray")
        ya_ax.axhline(state_lims[8,0], color="gray")
        ya_ax.axhline(state_lims[8,1], color="gray")
        rr_ax.axhline(state_lims[9,0], color="gray")
        rr_ax.axhline(state_lims[9,1], color="gray")
        yr_ax.axhline(state_lims[10,0], color="gray")
        yr_ax.axhline(state_lims[10,1], color="gray")
        pr_ax.axhline(state_lims[11,0], color="gray")
        pr_ax.axhline(state_lims[11,1], color="gray")

        th_ax.axhline(control_lims[0,0], color="gray")
        th_ax.axhline(control_lims[0,1], color="gray")
        tx_ax.axhline(control_lims[1,0], color="gray")
        tx_ax.axhline(control_lims[1,1], color="gray")
        ty_ax.axhline(control_lims[2,0], color="gray")
        ty_ax.axhline(control_lims[2,1], color="gray")
        tz_ax.axhline(control_lims[3,0], color="gray")
        tz_ax.axhline(control_lims[3,1], color="gray")

        px_ax.grid()
        py_ax.grid()
        pz_ax.grid()
        vx_ax.grid()
        vy_ax.grid()
        vz_ax.grid()
        ro_ax.grid()
        pi_ax.grid()
        ya_ax.grid()
        rr_ax.grid()
        pr_ax.grid()
        yr_ax.grid()
        th_ax.grid()
        tx_ax.grid()
        ty_ax.grid()
        tz_ax.grid()

        scale_fig(fig, 1)
        fig.suptitle("branch_idxs_to_root: {}".format(results[0]["branch_idxs_to_root"][:-1]))

        # fig, ax = make_fig()
        # eigenVectors = results[0]["eigenVectorss"][0]
        # ax.imshow(eigenVectors)
    return 


def plot_branchdata_results2(results):

    # # render 
    fig2_on = True
    fix_ax_on = True
    plot_each_branch_sequentially_on = False
    plot_all_results_on_one_figure = False

    if plot_all_results_on_one_figure:
        fig, ax = make_3d_fig()
        scale_fig(fig, 1)

    config_dict = results[0]["config_dict"]
    mdp = get_mdp(config_dict["ground_mdp_name"], results[0]["config_path"])

    from learning.feedforward import Feedforward
    if config_dict["ground_mdp_name"] == "SixDOFAircraft" and config_dict["aero_mode"] == "neural":
        model = Feedforward(config_dict["model_device"],
            config_dict["model_overfit_mode"],
            config_dict["model_training_mode"],
            config_dict["model_lipshitz_const"],
            config_dict["model_train_test_ratio"],
            config_dict["model_batch_size"],
            config_dict["model_initial_learning_rate"],
            config_dict["model_num_hidden_layers"],
            config_dict["model_num_epochs"],
            config_dict["model_path"],
            16,
            config_dict["model_hidden_dim"],
            6)
        if not os.path.exists(config_dict["model_path"]):
            exit("error: model path does not exist")
        model.load(config_dict["model_path"])
        weightss, biass = model.extract_ff_layers()
        mdp.set_weights(weightss, biass)

    # sort by initial condition 
    x0s = np.unique([result["xbar"] for result in results], axis=0)
    num_x0s = x0s.shape[0]
    # x0s = sorted(x0s, key=lambda x: x[-1])
    
    sorted_results = [[] for ii in range(num_x0s)]
    for ii, x0 in enumerate(x0s):
        for result in results: 
            if np.linalg.norm(result["xbar"]-x0)<1e-6:
                sorted_results[ii].append(result)
    sorted_results = sorted(sorted_results, key=lambda x: x[0]["depth"])

    for results in tqdm(sorted_results, desc="plotting branchdata"): 

        body_radius = 0.05
        robot_alpha = 0.5
        x0_color = "gray"
        xbar = results[0]["xbar"]

        colors = get_n_colors(len(results))
        # for ii, _ in enumerate(results): 
        # for ii in :
        if plot_each_branch_sequentially_on:
            iis = range(len(results))
        else:
            iis = [len(results)-1]
        
        for ii in iis:
            
            if fig2_on:
                fig2, ax2s = make_fig(nrows=3,ncols=2)

            # render all trajectories on a master figure 
            if not plot_all_results_on_one_figure:
                fig, ax = make_3d_fig()
            # render_sixdofaircraft(results[ii]["xbar"], result["config_dict"], body_radius, fig, ax, x0_color, robot_alpha)
            
            path_from_root = np.array(results[ii]["path_to_root"]) # time x state_dim
            render_xs(path_from_root, results[ii]["config_dict"]["ground_mdp_name"], results[ii]["config_dict"], 
                fig=fig, ax=ax, color="black", alpha=0.2)

            for jj in range(ii+1):
            # for jj in [ii]:

                result = results[jj]

                # if True:
                if result["is_valid"]:
                    color = colors[jj]
                else:
                    color = "gray"

                if plot_each_branch_sequentially_on and jj == ii:
                    alpha = 0.9
                else:
                    alpha = 0.2

                # render_sixdofaircraft(result["xs"][-1], result["config_dict"], body_radius, fig, ax, color, robot_alpha)

                xs_np = np.array(result["xs"])
                us_np = np.array(result["us"])
                us_ref_np = np.array(result["us_ref"])
                zs_np = np.array(result["zs_ref"])

                # change coordinates
                xs_np[:,2] = -1 * xs_np[:,2]
                zs_np[:,2] = -1 * zs_np[:,2]

                ax.plot(xs_np[:,0], xs_np[:,1], xs_np[:,2], color=color, alpha=alpha)
                ax.plot(zs_np[:,0], zs_np[:,1], zs_np[:,2], color=color, alpha=alpha, linestyle="dashed")
                ax.plot(np.nan, np.nan, np.nan, color=color, alpha=alpha, label=result["branch_idx"])

                x0 = np.zeros((13,))
                x0[0:12] = xbar
                xs_open_loop = [x0]
                for kk in range(xs_np.shape[0]):
                    xs_open_loop.append(mdp.F(xs_open_loop[-1], us_ref_np[kk,:]))
                xs_open_loop = np.array(xs_open_loop[1:])
                xs_open_loop[:,2] = -1 * xs_open_loop[:,2]
                ax.plot(xs_open_loop[:,0], xs_open_loop[:,1], xs_open_loop[:,2], color=color, alpha=alpha, linestyle="dotted")

                if fig2_on:
                    time_x = np.arange(xs_np.shape[0])
                    ax2s[0,0].plot(time_x, np.linalg.norm(xs_np[:,0:-1]-zs_np[0:xs_np.shape[0],:], axis=1), color=color, alpha=alpha)
                    ax2s[0,0].plot(time_x, np.linalg.norm(xs_open_loop[:,0:-1]-zs_np[0:xs_np.shape[0],:], axis=1), color=color, alpha=alpha, linestyle="dotted")
                    
                    xs_np_report = result["xs"]
                    xs_open_loop_np_report = np.copy(xs_open_loop)
                    xs_open_loop_np_report[:,2] = -1 * xs_open_loop_np_report[:,2]

                    X = np.reshape(np.array(result["config_dict"]["ground_mdp_X"]), (13, 2), order="F")
                    time_x = np.arange(xs_np.shape[0])
                    state_colors = get_n_colors(xs_np.shape[1])
                    state_labels = result["config_dict"]["ground_mdp_state_labels"]
                    for kk in range(xs_np_report.shape[1]):
                        label = state_labels[kk] if (jj==0) else None
                        xs_np_kk_norm = (xs_np_report[:,kk] - X[kk,0]) / (X[kk,1] - X[kk,0])
                        xs_open_loop_np_kk_norm = (xs_open_loop_np_report[0:xs_np_report.shape[0],kk] - X[kk,0]) / (X[kk,1] - X[kk,0])
                        ax2s[1,0].plot(time_x, xs_np_kk_norm, color=state_colors[kk], alpha=alpha, label=label)
                        ax2s[1,0].plot(time_x, xs_open_loop_np_kk_norm, color=state_colors[kk], alpha=alpha, linestyle="dotted")

                    U = np.reshape(np.array(result["config_dict"]["ground_mdp_U"]), (8, 2), order="F")
                    time_u = np.arange(us_np.shape[0])
                    control_colors = get_n_colors(us_np.shape[1])
                    control_labels = result["config_dict"]["ground_mdp_control_labels"]
                    for kk in range(us_np.shape[1]):
                        label = control_labels[kk+3] if (jj==0) else None
                        us_np_kk_norm = (us_np[:,kk] - U[kk+3,0]) / (U[kk+3,1] - U[kk+3,0])
                        us_ref_np_kk_norm = (us_ref_np[0:us_np.shape[0],kk] - U[kk+3,0]) / (U[kk+3,1] - U[kk+3,0])
                        ax2s[2,0].plot(time_u, us_np_kk_norm, color=control_colors[kk], alpha=alpha, label=label)
                        ax2s[2,0].plot(time_u, us_ref_np_kk_norm, color=control_colors[kk], alpha=alpha, linestyle="dotted")

                    if result["config_dict"]["ground_mdp_name"] == "SixDOFAircraft":
                        aero_labels = ["f_x", "f_y", "f_z", "m_x", "m_y", "m_z"]
                        aeros = []
                        for kk in range(xs_np.shape[0]):
                            aeros.append(wrapper_aero_model(xs_np[kk,0:12], us_np[kk], mdp))
                        aeros_np = np.array(aeros) # (len_traj, 6)
                        aeros_np[:,2] = -1 * aeros_np[:,2] # flip "z" coordinate
                        for ii in range(6):
                            label = aero_labels[ii] if (jj==0) else None
                            ax2s[0,1].plot(aeros_np[:,ii], label=label)
        
            if fig2_on:
                ax2s[1,0].legend(loc="center left")
                ax2s[2,0].legend(loc="center left")
                ax2s[0,1].legend()

            # vector fields 
            # todo 

            if fix_ax_on:
                scale = 0.5
                # ax.set_xlim([xbar[0]-0.5*scale, xbar[0]+1.5*scale])
                ax.set_xlim([xbar[0]-scale, xbar[0]+scale])
                ax.set_ylim([xbar[1]-scale, xbar[1]+scale])
                ax.set_zlim([-1 * xbar[2]-scale, -1 * xbar[2]+scale])

            if not plot_all_results_on_one_figure:
                scale_fig(fig, 1)
                fig.suptitle("branch_idxs: {}".format(results[0]["branch_idxs_to_root"]))

        if fig2_on:
            scale_fig(fig2, 1)

    return 


