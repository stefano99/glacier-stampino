"""Module providing a simulation of GLACIER."""

# Import lybraries
from typing import Iterable, Optional
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Sets All desired variables for the symulation
n = 1000000 # number of particles
a = 30 # cm lenght of the scintillator bars
b = 4  # cm width of the scintillator barss
s = 1  # cm height of the scintillator bars
c = 5  # cm distance between two top layers

dx = s/(2*c)*a
dy = s/(2*c)*b

# Generates all particles positions on the top layer of the detector
x = np.array(np.random.rand(n))*(a+2*dx)
y = np.array(np.random.rand(n))*(b+2*dy)
z = np.array([2*c, c, 0.])

def plot_glacier(
    x_pl: np.ndarray, # x position of the points
    y_pl: np.ndarray, # y position of the points
    sample_idx: Optional[Iterable[int]] = None, # index of plotted points
    a_pl: float = 30., # cm lenght of the scintillator bars
    b_pl: float = 4., # cm width of the scintillator barss
    c_pl: float = 5., # cm distance between two top layers
    s_pl: float = 1., # cm height of the scintillator bars
    z_levels: tuple[float, float, float] = (0.0, 5.0, 10.0), # default spacing between planes
    figsize: tuple[float, float] = (8, 8),
    title: str = "GLACIER"):

    """
    Function that can generate GLACIER plot: returns the figure object.
    The ax and the index of all plotted points
    """

    x_pl = np.asarray(x_pl)
    y_pl = np.asarray(y_pl)
    assert x.shape == y.shape, "x and y must have the same shape"

    # Plots only a subsample of the given points
    if sample_idx is not None:
        idx = np.array(list(sample_idx), dtype=int)
    else:
        # if the sample index is not set it gets the min between 250 and half the square root of n
        sample_size = min(int(np.sqrt(len(x_pl))/2.), 250)
        rng = np.random.default_rng()
        idx = rng.choice(len(x_pl), size=sample_size, replace=False)
    xs, ys = x_pl[idx], y_pl[idx]
    z_top = float(np.max(z_levels))
    dx_pl = s_pl/(2*c_pl)*a_pl
    dy_pl = s_pl/(2*c_pl)*b_pl
    xlim = (dx_pl, a_pl+dx_pl)
    ylim = (dy_pl, b_pl+dy_pl)

    # Draws GLACIER
    charm_colors = ["#0000FF", "#00FF00", "#FF0000"]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for z_pl, color in zip(z_levels, charm_colors):
        if z_pl == np.max(z_levels):
            verts = [[
            (xlim[0]-dx_pl, ylim[0]-dy_pl, z_pl),
            (xlim[1]+dx_pl, ylim[0]-dy_pl, z_pl),
            (xlim[1]+dx_pl, ylim[1]+dy_pl, z_pl),
            (xlim[0]-dx_pl, ylim[1]+dy_pl, z_pl),
            ]]
        else:
            verts = [[
            (xlim[0], ylim[0], z_pl),
            (xlim[1], ylim[0], z_pl),
            (xlim[1], ylim[1], z_pl),
            (xlim[0], ylim[1], z_pl),
            ]]
        poly = Poly3DCollection(verts, facecolors=color, alpha=0.5,  linewidths=0.5)
        ax.add_collection3d(poly)

    # Adds the sampled points
    ax.scatter(xs, ys, np.full_like(xs, z_top), s=20)
    ax.set_title(title, fontweight='bold', fontsize=22)
    ax.set_xlabel("x [cm]", fontweight='bold', fontsize=18)
    ax.set_ylabel("y [cm]", fontweight='bold', fontsize=18)
    ax.set_zlabel("z [cm]", fontweight='bold', fontsize=18)
    ax.set_xlim(-2., max(np.max(x_pl), np.max(y_pl)))
    ax.set_ylim(-2., max(np.max(x_pl), np.max(y_pl)))
    dz = (max(z_levels) - min(z_levels)) * 0.1
    ax.set_zlim(min(z_levels) - dz, max(z_levels) + dz)
    plt.show()

    return fig, ax, idx

fig_plot, ax_plot, idx_plot = plot_glacier(x, y, z_levels=z, sample_idx=np.arange(0,15,1), title="GLACIER PLOT")

# Generates the direction of the incoming particles
theta = np.array(np.random.rand(n)*np.pi/2.)
phi = np.array(np.random.rand(n)*2*np.pi)

# Defines the vector of the 3D-line (particle direction)
l = np.sin(theta)*np.cos(phi)
m = np.sin(theta)*np.sin(phi)
n = np.cos(theta)

# intercept the bottom plane
t = (z[-1]-z[0])/n
x0 = l*t + x
y0 = m*t + y

# visualize some incoming particles
segments = [ [(x0[j], y0[j], z[-1]), (x[j], y[j], z[0])] for j in idx_plot ]
lc = Line3DCollection(segments, linewidths=1, alpha=0.9, colors="#5A2ACA")
ax_plot.add_collection3d(lc)
# display(fig_plot)

signal = np.where((dx<x0) & (x0<a+dx) & (dy<y0) & (y0<b+dy))[0]
print(f"Particle generated: {n}")
print(f"Particle counted:   {len(signal)}")
G = len(signal)/n
print(f"Angular Coverage: {round(G*100,2)} %, meaning {round(4*np.pi*G,2)} str.")
print(f"Acceptance: {round(4*np.pi*G*a*b,2)} cm^2")
