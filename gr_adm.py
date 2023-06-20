# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:30:07 2023

@author: Sri
"""


import math
import numpy as np
from scipy.constants import speed_of_light as c
from scipy.constants import gravitational_constant as G

import random

import matplotlib.pyplot as plt
 
from matplotlib.animation import FuncAnimation

from functools import partial

from tqdm import tqdm




def find_fixed_point(func, *guess, mode='picard', fixed_point_iterations=2, error_thresh=1e-10):
    previous = guess
    for i in range(fixed_point_iterations):
        current = func(*previous)
        error = np.linalg.norm(np.array(current) - np.array(previous))
        previous = current
        if abs(error) < error_thresh:
            return previous
    return previous
    


def solve_hamiltonian_implicit_equations(hamiltonian, coords, momenta, init_guess_coords, init_guess_momenta, dt=1e-2, too_small_thresh=1e-10, too_small_fallback_randomness=.1, fixed_point_iterations=3, fixed_point_error_thresh=1e-10, dim=2,): 
    #IMPORTANT: the hamiltonian is assumed time independent
    #order of args for the hamiltonian is q,p
    
    #takes in space coords and space momenta at *this* time slice. 
    #it's a pretty obvious what the mathematical formula generalizing this to general dimensions would be, but it is entirely unnecessary for our purposes to code up the logic implementing that generalization.
    
    #the equations are from https://iopscience.iop.org/article/10.3847/1538-4365/aac9ca#apjsaac9caapp1 and pg 210 of Kang Feng, Mengzhao Qin - Symplectic Geometric Algorithms for Hamiltonian Systems -Springer (2010) 
    def next_coords_and_momenta_implicit(guess_next_coords, guess_next_momenta, hamiltonian=hamiltonian, coords=coords, momenta=momenta, dt=dt, too_small_thresh=too_small_thresh, too_small_fallback_randomness=too_small_fallback_randomness, dim=dim):
        if dim==2:
            #the point of renaming the variables is to try to make the proceeding formulas more readable. my general philosophy as of writing this comment is that for long formulae, this should be done, but for usage of the variable in short formulae or elsewhere the full name should be used.
            H = hamiltonian
            
            qx_0, qy_0 = coords
            qx_1, qy_1 = guess_next_coords
            
            px_0, py_0 = momenta
            px_1, py_1 = guess_next_momenta
            
            dqx, dqy = qx_1-qx_0, qy_1-qy_0
            dpx, dpy = px_1-px_0, py_1-py_0
            
            out_qx, out_qy = 0, 0
            out_px, out_py = 0, 0
            
            if abs(dpx) < too_small_thresh:
                step = too_small_thresh * (-1)**(random.randint(0,1))
                out_qx = qx_0 + dt * (H(qx_0, qy_0, px_0+step, py_0) - H(qx_0, qy_0, px_0-step, py_0)) / (2*step)
            else:
                out_qx = qx_0 + dt * (1/dpx) * (1/4) * ( H(qx_1, qy_0, px_1, py_0)-H(qx_1, qy_0, px_0, py_0) + H(qx_1, qy_1, px_1, py_1)-H(qx_1, qy_1, px_0, py_1) + H(qx_0, qy_0, px_1, py_0)-H(qx_0, qy_0, px_0, py_0) + H(qx_0, qy_1, px_1, py_1)-H(qx_0, qy_1, px_0, py_1) )
            
            if abs(dpy) < too_small_thresh:
                step = too_small_thresh * (-1)**(random.randint(0,1))
                out_qy = qy_0 + dt * (H(qx_0, qy_0, px_0, py_0+step) - H(qx_0, qy_0, px_0, py_0-step)) / (2*step)
            else:
                out_qy = qy_0 + dt * (1/dpy) * (1/4) * ( H(qx_1, qy_1, px_1, py_1)-H(qx_1, qy_1, px_1, py_0) + H(qx_0, qy_1, px_0, py_1)-H(qx_0, qy_1, px_0, py_0) + H(qx_1, qy_0, px_1, py_1)-H(qx_1, qy_0, px_1, py_0) + H(qx_0, qy_0, px_0, py_1)-H(qx_0, qy_0, px_0, py_0) )
            
            
            if abs(dqx) < too_small_thresh:
                step = too_small_thresh * (-1)**(random.randint(0,1))
                out_px = px_0 - dt * (H(qx_0+step, qy_0, px_0, py_0) - H(qx_0-step, qy_0, px_0, py_0)) / (2*step)
            else:
                out_px = px_0 - dt * (1/dqx) * (1/4) * ( H(qx_1, qy_0, px_0, py_0)-H(qx_0, qy_0, px_0, py_0) + H(qx_1, qy_1, px_0, py_1)-H(qx_0, qy_1, px_0, py_1) + H(qx_1, qy_0, px_1, py_0)-H(qx_0, qy_0, px_1, py_0) + H(qx_1, qy_1, px_1, py_1)-H(qx_0, qy_1, px_1, py_1))
            
            if abs(dqy) < too_small_thresh:
                step = too_small_thresh*(-1)**(random.randint(0,1))
                out_py = py_0 - dt * (H(qx_0, qy_0+step, px_0, py_0) - H(qx_0, qy_0-step, px_0, py_0)) / (2*step)
            else:
                out_py = py_0 - dt * (1/dqy) * (1/4) * ( H(qx_1, qy_1, px_1, py_0)-H(qx_1, qy_0, px_1, py_0) + H(qx_0, qy_1, px_0, py_0)-H(qx_0, qy_0, px_0, py_0) + H(qx_1, qy_1, px_1, py_1)-H(qx_1, qy_0, px_1, py_1) + H(qx_0, qy_1, px_0, py_1)-H(qx_0, qy_0, px_0, py_1) )

        
            return [out_qx, out_qy], [out_px, out_py]
     
        
    return find_fixed_point(next_coords_and_momenta_implicit, init_guess_coords, init_guess_momenta, fixed_point_iterations=fixed_point_iterations, error_thresh=fixed_point_error_thresh)

def hamiltonian_solve(hamiltonian, init_coords, init_momenta, runtime, dt=1e-2, guessing_delta=1e-5, too_small_thresh=1e-10, fixed_point_iterations=3, fixed_point_error_thresh=1e-10, dim=2):
    if dim == 2:
        previous_qx, previous_qy = init_coords
        previous_px, previous_py = init_momenta
        
        for time in tqdm(np.arange(0, runtime, dt)):
            guess_qx = previous_qx + (guessing_delta) * (-1)**(random.randint(0,1))
            guess_qy = previous_qy + (guessing_delta) * (-1)**(random.randint(0,1))
            guess_px = previous_px + (guessing_delta) * (-1)**(random.randint(0,1))
            guess_py = previous_py + (guessing_delta) * (-1)**(random.randint(0,1))

            current_coords, current_momenta = solve_hamiltonian_implicit_equations(hamiltonian, [previous_qx, previous_qy], [previous_px, previous_py], [guess_qx, guess_qy], [guess_px, guess_py], dt=dt, too_small_thresh=too_small_thresh, fixed_point_iterations=fixed_point_iterations, fixed_point_error_thresh=fixed_point_error_thresh, dim=dim)
            yield time, current_coords
            previous_qx, previous_qy = current_coords
            previous_px, previous_py = current_momenta

def H_sho(qx, qy, px, py):
    return np.linalg.norm([px, py])**2/2 + (2*math.pi)**2 * np.linalg.norm([qx, qy])**2/2

def make_hamiltonian(schwarzschild_spatial_metric, lapse, shift, lightlike_or_timelike_constant=1): #1 is for timelike, 0 is for lightlike
    def H_ADM(x1, x2, u1, u2, schwarzschild_spatial_metric=schwarzschild_spatial_metric, lapse=lapse, shift=shift, lightlike_or_timelike_constant=lightlike_or_timelike_constant):
        u = [u1, u2]
        this_spatial_metric = schwarzschild_spatial_metric(x1, x2)
        u_lowered = [0,0]
        for i in range(2):
            for j in range(2):
                u_lowered[i] += this_spatial_metric[i,j] * u[j]
        
        this_lapse = lapse(x1,x2)
        this_shift = shift(x1,x2)
        this_inverse_spatial_metric = np.linalg.inv(this_spatial_metric)
        
        acc_out = 0
        for i in range(2):
            for j in range(2):
                acc_out += this_lapse * (this_inverse_spatial_metric[i,j] * u_lowered[i] * u_lowered[j] + lightlike_or_timelike_constant)*.5 - this_shift[j] * u_lowered[j]
        
        return acc_out
    
    return H_ADM

def nbody_frame_update(time_and_poses, artists, single_particle_mode=True, plot_3D=False, trail=50):
    time, poses = time_and_poses[0], time_and_poses[1]
    
    time_text_artist = artists[-1]
    line_artists = artists[:-1]
    
    time_text_artist.set_text(f"Time: {round(time,5)}")
    for particle_index in range(len(line_artists)):
        line_artist = line_artists[particle_index]
        this_pos = poses[particle_index] if not single_particle_mode else poses
        new_x, new_y = this_pos[0], this_pos[1]
        
        if plot_3D:
            new_z = this_pos[2]
            trailing_xs, trailing_ys, trailing_zs = line_artist.get_data_3d()
        else:
            trailing_xs, trailing_ys = line_artist.get_data()
            
        if len(trailing_xs) < trail:
            trailing_xs = np.append(trailing_xs, new_x)
            trailing_ys = np.append(trailing_ys, new_y)
            if plot_3D:
                trailing_zs = np.append(trailing_zs, new_z)
        else:
            trailing_xs = np.roll(trailing_xs,-1)
            trailing_ys = np.roll(trailing_ys,-1)
            if plot_3D:
                trailing_zs = np.roll(trailing_zs,-1)
                
            trailing_xs[-1] = new_x
            trailing_ys[-1] = new_y
            if plot_3D:
                trailing_zs[-1] = new_z
        
        line_artist.set_data(trailing_xs, trailing_ys)
        if plot_3D:
            line_artist.set_3d_properties(trailing_zs)
        
    return [*line_artists, time_text_artist]
     
def animate(time_and_poses_gen, num_particles, frame_count, frame_interval, window_one_axis=[-20, 20], plot_3D=False, trail=500):
    
    fig = plt.figure()
    
    if plot_3D:
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim3d=window_one_axis, xlabel='X')
        ax.set(ylim3d=window_one_axis, ylabel='Y')
        ax.set(zlim3d=window_one_axis, zlabel='Z')
        line_artists = []
        for line_index in range(num_particles):
            line_artists.append(ax.plot([],[],[], 'o-', markersize=5, markevery=[-1], animated=True)[0])
    else: 
        ax = fig.add_subplot()
        ax.set(xlim=window_one_axis, xlabel='X')
        ax.set(ylim=window_one_axis, ylabel='Y')
        line_artists = []
        for line_index in range(num_particles):
            line_artists.append(ax.plot([],[], 'o-', markersize=5, markevery=[-1], animated=True)[0])
    time_text_artist = ax.set_title("")
    time_text_artist.set_animated(True)
    ani = FuncAnimation(fig, partial(nbody_frame_update, artists=[*line_artists, time_text_artist], trail=trail), frames=time_and_poses_gen, save_count=frame_count, interval=frame_interval, blit=True)
    return ani

def solve_and_animate_geodesic(hamiltonian, init_coords, init_momenta, runtime, dt=1e-2, guessing_delta=1e-5, too_small_thresh=1e-10, fixed_point_iterations=3, fixed_point_error_thresh=1e-10, dim=2, window_one_axis=[-20, 20], slowdown=1, plot_every_nth=1):
    time_and_poses_gen = hamiltonian_solve(hamiltonian, init_coords, init_momenta, runtime, dt=dt, guessing_delta=guessing_delta, too_small_thresh=too_small_thresh, fixed_point_iterations=fixed_point_iterations, fixed_point_error_thresh=fixed_point_error_thresh, dim=dim)
    
    frame_interval = 1000*dt*slowdown*plot_every_nth
    frame_count = math.ceil((runtime)/(plot_every_nth*dt))
    num_particles = 1#len(init_poses)
    
    ani = animate(time_and_poses_gen, num_particles, frame_count, frame_interval, window_one_axis=window_one_axis)
    return ani



if __name__ == '__main__':
    runtime=1e3
    dt=1
    
    guessing_delta=1e-5
    too_small_thresh=1e-10 
    fixed_point_iterations=3
    fixed_point_error_thresh=1e-10
    dim=2
    window_one_axis=[-20, 20]
    slowdown=.1
    plot_every_nth=1
    
    
    def schwarzschild_spatial_metric(x,y):
        r_squared = x**2 + y**2
        r = r_squared**.5
        
        denominator = (r_squared * (-2 + r))
        comp00 = 1 + 2*(x**2)/denominator
        comp01 = 2*(x*y)/denominator
        comp10 = comp01
        comp11 = 1 + 2*(x**2)/denominator
        
        out_spatial_metric = np.array([[comp00, comp01], [comp10, comp11]])
        return out_spatial_metric
        
    schwarzschild_lapse = lambda x1, x2: (1 - 2/((x1**2 + x2**2)**.5))**.5
    schwarzschild_shift = lambda x1, x2: [0,0]
    
    
    hamiltonian=make_hamiltonian(schwarzschild_spatial_metric, schwarzschild_lapse, schwarzschild_shift)
    init_coords=[10, 0]
    init_proper_vel=[0, .4]
    
    ani = solve_and_animate_geodesic(hamiltonian, init_coords, init_proper_vel, runtime, dt=dt, guessing_delta=guessing_delta, too_small_thresh=too_small_thresh, fixed_point_iterations=fixed_point_iterations, fixed_point_error_thresh=fixed_point_error_thresh, dim=dim, window_one_axis=window_one_axis, slowdown=slowdown, plot_every_nth=plot_every_nth)
    ani.save('results/adm_schwarzschild_precess_test.mp4')