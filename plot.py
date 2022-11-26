import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import mediapipe as mp

pose_connections = mp.solutions.pose.POSE_CONNECTIONS
mp_pose = mp.solutions.pose


# Empty array for the 33 landmarks
poselandmarks_list = []

for id, lm in enumerate(mp_pose.PoseLandmark):
    lm_str = repr(lm).split('.')[1].split(':')[0]
    poselandmarks_list.append(lm_str)

def scale_axes(ax):
    # Scale axes

    # get axis view limits 3d version
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xrange = abs(xlim[1]-xlim[0])
    xmid = np.mean(xlim)
    yrange = abs(ylim[1]-ylim[0])
    ymid = np.mean(ylim)
    zrange = abs(zlim[1]-zlim[0])
    zmid = np.mean(zlim)

    plot_radius = 0.5 * max(xrange, yrange, zrange)

    ax.set_xlim3d([xmid - plot_radius, xmid + plot_radius])
    ax.set_ylim3d([ymid - plot_radius, ymid + plot_radius])
    ax.set_zlim3d([zmid - plot_radius, zmid + plot_radius])

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])

def plot_data(data, ax, rotate = True):
    if rotate:
        ax.scatter(data[0, :], data[2, :], -data[1, :])

        for i in pose_connections:
            ax.plot3D([data[0, i[0]], data[0, i[1]]],
                    [data[2, i[0]], data[2, i[1]]],
                    [-data[1, i[0]], -data[1, i[1]]],
                    color='k', lw=1)

        ax.view_init(elev = 10, azim = 60)
    
    else:
        ax.scatter(data[0, :], data[2, :], -data[1, :])

        for i in pose_connections:
            ax.plot3D([data[0, i[0]], data[0, i[1]]],
                    [data[2, i[0]], data[2, i[1]]],
                    [-data[1, i[0]], -data[1, i[1]]],
                    color='k', lw=1)

        ax.view_init(elev = -90, azim = -90)

def rotate_and_save(figure, axis, filename, save = False):
    def init():
        return figure,
    
    def animate(i):
        axis.view_init(elev=10, azim = i)
        return figure,

    # animate
    anim = animation.FuncAnimation(figure, animate, init_func=init, frames=360, interval=20, blit = True)

    plt.close()

    # save
    if save:
        anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'], dpi=300)


def time_animate(data, figure, ax, rotate_data=True, rotate_animation=False):
    frame_data = data[:, :, 0]
    if rotate_data:
        plot = [ax.scatter(frame_data[0, :], -frame_data[2, :], -frame_data[1, :], color='tab:blue')]

        for i in pose_connections:
            plot.append(ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                                  [-frame_data[2, i[0]], -frame_data[2, i[1]]],
                                  [-frame_data[1, i[0]], -frame_data[1, i[1]]],
                                  color='k', lw=1)[0])

        ax.view_init(elev=10, azim=120)

    else:
        ax.scatter(frame_data[0, :], frame_data[1, :], frame_data[2, :], color='tab:blue')

        for i in pose_connections:
            ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                      [frame_data[1, i[0]], frame_data[1, i[1]]],
                      [frame_data[2, i[0]], frame_data[2, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=-90, azim=-90)

    scale_axes(ax)

    def init():
        return figure,

    def animate(i):
        frame_data = data[:, :, i]

        for idxx in range(len(plot)):
            plot[idxx].remove()

        plot[0] = ax.scatter(frame_data[0, :], -frame_data[2, :], -frame_data[1, :], color='tab:blue')

        idx = 1
        for pse in pose_connections:
            plot[idx] = ax.plot3D([frame_data[0, pse[0]], frame_data[0, pse[1]]],
                                  [-frame_data[2, pse[0]], -frame_data[2, pse[1]]],
                                  [-frame_data[1, pse[0]], -frame_data[1, pse[1]]],
                                  color='k', lw=1)[0]
            idx += 1

        if rotate_animation:
            ax.view_init(elev=10., azim=120 + (360 / data.shape[-1]) * i)

        return figure,

    # Animate
    anim = animation.FuncAnimation(figure, animate, init_func=init,
                                   frames=144, interval=20, blit=True)

    plt.close()

    return anim