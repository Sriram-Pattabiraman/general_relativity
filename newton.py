# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:45:22 2023

@author: Sri
"""


import math
import numpy as np
from scipy.constants import gravitational_constant as G

import matplotlib.pyplot as plt
 
from matplotlib.animation import FuncAnimation

from functools import partial

from tqdm import tqdm


def newton_force(body1_mass, body1_pos, body2_mass, body2_pos):
    displacement_vec = body2_pos - body1_pos
    distance = np.linalg.norm(displacement_vec)
    force_on_1 = displacement_vec *( (G * body1_mass * body2_mass)/(distance**3) ) # displacement_vec is unnormalized.
    return force_on_1

def calculate_forces(body_masses, body_poses):
    running_total_forces = np.zeros_like(body_poses)
    for i in range(len(body_poses)):
        for j in filter(lambda j: j<i, range(len(body_poses))): #the filter makes sure that self interactions aren't included and that we don't calculate forces twice per (body1,body2) pair.
            force_on_i = newton_force(body_masses[i], body_poses[i], body_masses[j], body_poses[j])
            running_total_forces[i] += force_on_i
            running_total_forces[j] += -force_on_i #newton's first law
    return running_total_forces


def solve(masses, initial_poses, initial_velocities, start_time=0, end_time=5, dt=.001, store_history=False, yield_only_time_and_pos=True, velocity_verlet=True):
    current_poses, current_velocities = initial_poses, initial_velocities
    #we use verlet integration, which is a symplectic integrator, which means the phase space area will be conserved and energies of the system won't spiral out of accuracy.
    #the reason why this conserves that is that a generalized coordinate update happens, and then a generalized momentum update happens *using the new generalized coordinate*.
    #that is, the p update is a function of q only, and the q update is a function of p only, leading to a distortion/sliding of a phase space rectangle but not any area change.
    #this specific scheme finds the velocity at t+.5dt using v(t) and a(t), then uses that to find x(t+dt), then uses x(t+dt) to find a(t+dt), and then propogates v(t+.5dt) under a(t+dt) to find v(t+dt).
    #this is sometimes called "kick-drift-kick".
    #the most understandable source for this can be found here: https://www.av8n.com/physics/symplectic-integrator.htm
    current_forces = calculate_forces(masses, current_poses)
    masses = np.reshape(masses, (len(masses),1))
    current_accelerations = current_forces/masses
    if store_history:
        history_times, history_poses, history_velocities, history_forces, history_accelerations = [], [], [], [], []
    for time in tqdm(np.arange(start_time, end_time, dt)):
        if yield_only_time_and_pos:
            yield time, current_poses
        else:
            yield time, current_poses, current_velocities, current_forces, current_accelerations
        if store_history:
            history_times.append(time)
            history_poses.append(current_poses)
            history_velocities.append(current_velocities) 
            history_forces.append(current_forces)
            history_accelerations.append(current_accelerations)
            
        if velocity_verlet:
            half_step_velocities = current_velocities + .5*dt*current_accelerations
            current_poses += half_step_velocities*dt
            current_forces = calculate_forces(masses, current_poses)
            current_accelerations = current_forces/masses
            current_velocities = half_step_velocities + .5*dt*current_accelerations
        else:
            current_poses += dt*current_velocities
            current_velocities += dt*current_accelerations
            current_forces = calculate_forces(masses, current_poses)
            current_accelerations = current_forces/masses
            
    if store_history:
        return history_times, history_poses, history_velocities, history_forces, history_accelerations
    

def nbody_frame_update(time_and_poses, scatter_plot_artist, time_text_artist, plot_3D=False):
    time, poses = time_and_poses[0], time_and_poses[1]
    scatter_plot_artist.set_data(poses[:,0], poses[:,1])
    if plot_3D:
        scatter_plot_artist.set_3d_properties(poses[:,2])
    else:
        pass
    time_text_artist.set_text(f"Time: {round(time,5)}")
    return scatter_plot_artist, time_text_artist
     
def animate(time_and_poses_gen, frame_count, frame_interval, window_one_axis=[-20, 20], plot_3D=False):
    
    fig = plt.figure()
    
    if plot_3D:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim3d=window_one_axis, xlabel='X')
        ax.set(ylim3d=window_one_axis, ylabel='Y')
        ax.set(zlim3d=window_one_axis, zlabel='Z')
        scatter_plot_artist = ax.plot([],[],[],'.')[0]
    else: 
        ax = fig.add_subplot()
        ax.set(xlim=window_one_axis, xlabel='X')
        ax.set(ylim=window_one_axis, ylabel='Y')
        scatter_plot_artist = ax.plot([],[],'.')[0]
    time_text_artist = ax.set_title("")
    ani = FuncAnimation(fig, partial(nbody_frame_update, scatter_plot_artist=scatter_plot_artist, time_text_artist=time_text_artist), frames=time_and_poses_gen, save_count=frame_count, interval=frame_interval)
    return ani

def animate_solve(masses, initial_poses, initial_velocities, start_time=0, end_time=10, dt=.01, window_one_axis=[-20, 20], velocity_verlet=True):
    time_and_poses_gen = solve(masses, initial_poses, initial_velocities, start_time, end_time, dt=dt, store_history=False, yield_only_time_and_pos=True, velocity_verlet=velocity_verlet)
    frame_interval = 1000*dt
    frame_count = math.ceil((end_time - start_time)/dt)
    ani = animate(time_and_poses_gen, frame_count, frame_interval, window_one_axis=window_one_axis)
    return ani

if __name__ == '__main__':
    
    '''
    G = 1
    masses = np.array([50,1], dtype='float64')
    initial_poses = np.array([ [0,0,0], [2,0,0] ], dtype='float64')
    initial_velocities = np.array([ [0,0,0], [0, 5, 0] ], dtype='float64')
    start_time=0
    end_time=30
    dt=.01
    window_one_axis = [-5, 5]
    ani = animate_solve(masses, initial_poses, initial_velocities, start_time=start_time, end_time=end_time, dt=dt, window_one_axis=window_one_axis, velocity_verlet=True)
    ani.save('test_verlet.mp4')
    '''
    
    G = 1
    masses = np.array([50,1], dtype='float64')
    initial_poses = np.array([ [0,0], [2,0] ], dtype='float64')
    initial_velocities = np.array([ [0,0], [0, 5] ], dtype='float64')
    start_time=0
    end_time=30
    dt=.01
    window_one_axis = [-5, 5]
    ani = animate_solve(masses, initial_poses, initial_velocities, start_time=start_time, end_time=end_time, dt=dt, window_one_axis=window_one_axis, velocity_verlet=True)
    ani.save('results/test_verlet.mp4')
    
    
    
    '''
    G = 1
    masses = np.array([50,1], dtype='float64')
    initial_poses = np.array([ [0,0,0], [2,0,0] ], dtype='float64')
    initial_velocities = np.array([ [0,0,0], [0, 5, 0] ], dtype='float64')
    start_time=0
    end_time=30
    dt=.01
    window_one_axis=[-5, 5]
    ani = animate_solve(masses, initial_poses, initial_velocities, start_time=start_time, end_time=end_time, dt=dt, window_one_axis=window_one_axis, velocity_verlet=False)
    ani.save('results/test_not_verlet.mp4')
    '''
    
    
    
    '''
    G = 10
    rng = np.random.default_rng()
    masses = rng.uniform(low=25, high=100, size=15)
    initial_poses = rng.uniform(low=-5, high=5, size=[15, 3])
    initial_velocities = rng.uniform(low=-5, high=5, size=[15, 3])
    start_time=0
    end_time=10
    dt=.01
    window=np.repeat([[-50, 50]], 3, axis=0)
    ani = animate_solve(masses, initial_poses, initial_velocities, start_time=start_time, end_time=end_time, dt=dt, window=window)
    ani.save('results/random.mp4')
    '''