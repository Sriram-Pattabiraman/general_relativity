# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:36:30 2023

@author: Sri
"""


import math
import numpy as np
from scipy.constants import speed_of_light as c
from scipy.constants import gravitational_constant as G

import matplotlib.pyplot as plt
 
from matplotlib.animation import FuncAnimation

from functools import partial

from tqdm import tqdm



#note: names like "time", or "[blank]" where blank is a name for a time derivative of a quantity (e.g. "velocity") refers to the 3-space quantity expressed in local coordinates with time coordinate t=x^0. this means that in x's reference frame, x^0 = t.
#to avoid unit problems, everything will be done in G,c=1 units, and input data will be converted from SI to geometrized units for the simulation functions to simulate with, and then potentially converted back to SI units to display to the user.
def SI_to_geometrized(si_value, si_unit):
    if si_unit=='m': #length
        return si_value
    elif si_unit=='s': #time
        return si_value*c
    elif si_unit=='kg': #mass
        return si_value*G/(c**2)
    elif si_unit=='m/s': #velocity
        return si_value/(c)
    elif si_unit=='m/s^2': #acceleration
        return si_value/(c**2)
    elif si_unit=='J': #energy
        return si_value*G/(c**4)
    elif si_unit=='N': #force
        return si_value*G/(c**4)

def fixed_point_iteration(func, start=0, n=5):
    previous = start
    current = func(previous)
    i = 0
    while i < n:
        previous = current
        current = func(current)
        i+=1
        
    return current

def fixed_point_newton(func, start=0, n=5):
    first_x = start
    previous_x = func(first_x)
    dx = previous_x - first_x
    
    func_to_find_zero_of = lambda arg: arg-func(arg)
    
    f_first = start - previous_x
    f_previous = func_to_find_zero_of(previous_x)
    df = f_previous - f_first
    if np.all(dx==0) or np.all(df==0):
        return previous_x
    jacobian_inverse = np.eye(len(start)) #this is a terrible guess. for nice enough cases, though, this will converge anyways.
    
    i = 1
    while i < n:
        new_x = previous_x - np.matmul(jacobian_inverse, f_previous)
        new_f = func_to_find_zero_of(new_x)
        
        dx = new_x - previous_x
        df = new_f - f_previous
        if np.all(dx==0) or np.all(df==0):
            return previous_x
        jacobian_inverse += np.matmul((dx - np.matmul(jacobian_inverse, df))/(np.dot(df, df)), df.T) #called the "bad boyden method", which isn't so bad after all.
        
        previous_x = new_x
        i += 1
    
    return new_x

def fixed_point(func, start=0, n=5, mode='newton'):
    if mode=='newton':
        return fixed_point_newton(func, start=start, n=n)
    elif mode=='iters':
        return fixed_point_iteration(func, start=start, n=n)


def solve_diffyq(dep_accel_func, init_deps, init_derivs, start_indep=0, end_indep=5, indep_step=.001, fixed_point_n=5, yield_every_nth=1, show_pbar=True, store_history=False, yield_only_indep_and_dependent=False, yield_velocity=True, yield_acceleration=False):
    #solves 2nd order ODEs.
    current_deps, current_derivs = init_deps, init_derivs
    current_indep = start_indep
    
    to_find_fixed_point = lambda half_step_deriv, particle_index=0: current_derivs[particle_index] + .5 * indep_step * dep_accel_func(current_indep, current_deps[particle_index], half_step_deriv)
    half_step_derivs = np.empty_like(current_derivs)
    for i in range(len(current_derivs)):
        half_step_derivs[i] = fixed_point(partial(to_find_fixed_point, particle_index=i), start=current_derivs[i], n=fixed_point_n)
    
    if store_history:
        current_dep_accels = (half_step_derivs - current_derivs) / (.5 * indep_step)
        history_indeps, history_deps, history_derivs, history_dep_accels = [], [], [], [], []
    iter_num = 0
    for current_indep in tqdm(np.arange(start_indep, end_indep, indep_step), desc='Solving DiffyQ...', disable=not(show_pbar)):
        if (iter_num % yield_every_nth) == 0:
            if yield_only_indep_and_dependent:
                yield current_indep, current_deps
            elif yield_velocity:
                yield current_indep, current_deps, current_derivs
            elif yield_acceleration:
                yield current_indep, current_deps, current_derivs, current_dep_accels
        if store_history:
            history_indeps.append(current_indep)
            history_deps.append(current_deps)
            history_derivs.append(current_derivs) 
            history_dep_accels.append(current_dep_accels)


        current_deps = current_deps + half_step_derivs * indep_step
        to_find_fixed_point = lambda current_deriv, particle_index=0: half_step_derivs[particle_index] + .5 * indep_step * dep_accel_func(current_indep, current_deps[particle_index], current_deriv)
        for i in range(len(current_derivs)):
            current_derivs[i] = fixed_point(partial(to_find_fixed_point, particle_index=i), start=half_step_derivs[i], n=fixed_point_n)
        
        to_find_fixed_point = lambda half_step_deriv, particle_index=0: current_derivs[particle_index] + .5 * indep_step * dep_accel_func(current_indep, current_deps[particle_index], half_step_deriv)
        for i in range(len(current_derivs)):
            half_step_derivs[i] = fixed_point(partial(to_find_fixed_point, particle_index=i), start=current_derivs[i], n=fixed_point_n)
        
        iter_num += 1
    if store_history:
        return history_indeps, history_deps, history_derivs, history_dep_accels
        
    


def deriv(in_comp_func, derivative_indice, dx=.001, input_pad_to=4):
    full_offset_vector = np.zeros(input_pad_to, dtype='float32')
    full_offset_vector[derivative_indice] += dx/2
    def comp_func(event, in_comp_func=in_comp_func, recip_dx=1/dx, full_offset_vector=full_offset_vector, input_pad_to=input_pad_to):
        if len(event) <= derivative_indice:
            event = np.pad(event, (0,input_pad_to-len(event)))
            offset_vector = full_offset_vector
        else:
            offset_vector = full_offset_vector[:len(event)]
        return (recip_dx) * (in_comp_func(event + offset_vector) - in_comp_func(event - offset_vector))
    return comp_func

def grid_deriv(data, grid, deriv_indice, point_indices, data_indices=None, data_is_func=True):
    #time_mesh = raw_grid[0]
    #time_mesh = time_mesh[~time_mesh.mask]
    #grid = [time_mesh, *raw_grid[1:]]
    
    #data_is_func distinguishes between using e.g. .christoffel() or .christoffel_grid[]
    offset_vector = np.zeros_like(point_indices)
    offset_vector[deriv_indice] = 1
    len_of_deriv_axis = len(grid[deriv_indice])
    
    
    if point_indices[deriv_indice] >= len_of_deriv_axis-1:
        right_point_indices = point_indices
    else:
        right_point_indices = point_indices + offset_vector

    if point_indices[deriv_indice] == 0:
        left_point_indices = point_indices
    else:
        left_point_indices = point_indices - offset_vector
    
    
    right_point = np.array([grid[i][right_point_indices[i]] for i in range(len(right_point_indices))])
    left_point = np.array([grid[i][left_point_indices[i]] for i in range(len(left_point_indices))])
    dx = np.linalg.norm(right_point - left_point)

    if data_indices == None:
        if data_is_func:
            right_data = data(right_point_indices)
            left_data = data(left_point_indices)
        else:
            right_data = data[right_point_indices]
            left_data = data[left_point_indices]
    else:
        if data_is_func:
            right_data = data(data_indices, right_point_indices)
            left_data = data(data_indices, left_point_indices)
        else:
            right_data = data[data_indices][right_point_indices]
            left_data = data[data_indices][left_point_indices]
            
    return (right_data - left_data)/dx
    
def flat_metric(indices):
    i,j = indices
    if i!=j:
        return 0
    elif i==0:
        return -1
    else:
        return 1

def make_grid(time_window=[0,10], space_1d_window=[-5,5], space_dim=3, dt=.01, dx=.01, meshgrid=False, return_find=True, return_steps=True): #dt, dx are not exactly honored. they are used to calculate num_t, num_x to be fed into np.linspace, which may result in a slightly different step size.
    grid = []
    num_t = math.ceil((time_window[1]-time_window[0])/dt)
    num_x = math.ceil((space_1d_window[1]-space_1d_window[0])/dx)
    time_1d_arr, dt = np.linspace(*time_window, num_t, retstep=True)
    time_1d_arr = np.ma.array(time_1d_arr, mask=np.repeat([True], len(time_1d_arr)))
    time_len = len(time_1d_arr)
    space_1d_arr, dx = np.linspace(*space_1d_window, num_x, retstep=True)
    space_len = len(space_1d_arr)
    if meshgrid:
        grid = np.meshgrid(time_1d_arr, *np.tile(space_1d_arr, (space_dim, 1)))
    else:
        grid = [time_1d_arr, *np.tile(space_1d_arr, (space_dim, 1))]
    
    outs = [grid]
    if return_find:
        
        def find(time, space_poses, time_window=time_window, space_1d_window=space_1d_window, time_len=time_len, space_len=space_len, num_x=num_x, num_t=num_t, dt=dt, dx=dx):
            time_start, time_end = time_window
            space_1d_start, space_1d_end = space_1d_window
            space_len = num_x
           
            if len(np.shape(space_poses)) == 1:
                one_particle_mode = True
            else:
                one_particle_mode = False
                
            if time >= time_end:
                time = time_end
            
            space_indexes = np.empty_like(space_poses, dtype='uint')
            if not one_particle_mode:
                for i,space_pos in enumerate(space_poses):
                    for j,space_comp in enumerate(space_pos):
                        this_comp = space_poses[i][j]
                        if this_comp >= space_1d_end:
                            space_indexes[i,j] = space_len-1
                        elif this_comp < space_1d_start:
                            space_indexes[i,j] = 0
                        else:
                            if np.isnan(space_comp):
                                space_indexes[i,j] = 0
                            else:
                                space_indexes[i,j] = int((this_comp - space_1d_start)//dx)
                        
            else:
                for j,space_comp in enumerate(space_poses):
                    if space_comp >= space_1d_end:
                        space_indexes[j] = space_len-1
                    elif space_comp < space_1d_start:
                        space_indexes[j] = 0
                    else:
                        if np.isnan(space_comp):
                            space_indexes[j] = 0
                        else:
                            space_indexes[j] = int((space_comp - space_1d_start)//dx)
                 
            time_index = int((time - time_start)//dt) 
            if time_index >= time_len:
                time_index = time_len -1
            elif time_index < 0:
                time_index = 0
                
            return time_index, space_indexes
        '''
        def find(time, space_poses, time_window=time_window, space_1d_window=space_1d_window, time_len=time_len, space_len=space_len, dt=dt, dx=dx):
            time_start, time_end = time_window
            space_1d_start, space_1d_end = space_1d_window
           
            if len(np.shape(space_poses)) == 1:
                one_particle_mode = True
            else:
                one_particle_mode = False
                
            if time >= time_end:
                time = time_end
            
            if not one_particle_mode:
                for i,space_pos in enumerate(space_poses):
                    for j,space_comp in enumerate(space_pos):
                        if space_poses[i,j] >= space_1d_end:
                            space_poses[i,j] = space_1d_end
            else:
                for j,space_comp in enumerate(space_poses):
                    if space_poses[j] >= space_1d_end:
                        space_poses[j] = space_1d_end
                        
            time_index = int((time - time_start)//dt) 
            if time_index >= time_len:
                time_index = time_len -1
            elif time_index < 0:
                time_index = 0
            
            space_indexes = ((space_poses-space_1d_start)//dx).astype('int')
            space_indexes[space_indexes >= space_len] = space_len - 1
            space_indexes[space_indexes < 0] = 0
            return time_index, space_indexes
        '''
        outs.append(find)
    
    if return_steps:
        outs.append([dt,dx])
    
    return outs
        


class SpacetimeMesh:
    def __init__(self, deviation_field, stored_space_dim, coord_grids, find, keep_time_size=3):
        '''
        #!!!todo: spaceblocks
        here's the idea: suppose we want discretized spacetime to be (n,m,l) sized. we already have a way of advancing time and forgetting the past to take care of the time dimension (saving memory and time).
        while with time, we can be very confident that we will only need the local time window, we know that this may be false for space.
        the idea is to store data on a block of space. if a value is requested outside of the currently loaded block, we (write to disk or throw away) the current block and load in the requested block.
        perhaps have an adjustable number_of_blocks (along with size), or have the size of blocks be computed from the number of blocks. this way the we can tell the mesh how many blocks will be needed by guessing that we will need as many blocks as we have particles with different trajectories.
        '''
        stored_spacetime_dim = stored_space_dim+1
        self.data_dim = stored_spacetime_dim
        
        self.lowest_kept_time_index = 0
        self.keep_time_size = keep_time_size
        coord_grids[0].mask[:keep_time_size] = False
        self.full_time_arr = np.ma.copy(coord_grids[0])
        self.grid = coord_grids
        self.grid[0] = self.grid[0][~self.grid[0].mask]
        self.find_raw = find
        
        self.grid_dim = len(coord_grids)
        self.sizes = np.zeros(len(coord_grids), dtype='int')
        self.sizes[0] = coord_grids[0].count()
        self.sizes[1:] = np.array([len(coord_grids[i]) for i in range(1, len(coord_grids))])
        
        self.deviation_field = deviation_field
        self.deviation_grid = np.ma.masked_all((self.data_dim, self.data_dim, *self.sizes), dtype='float32')
        
        self.christoffel_grid = np.ma.masked_all((self.data_dim, self.data_dim, self.data_dim, *self.sizes), dtype='float32')

    def find(self, time, space_poses):
        time_index, space_indexes = self.find_raw(time, space_poses)
        if time_index >= self.keep_time_size + self.lowest_kept_time_index:
            self.advance()
        time_index -= self.lowest_kept_time_index
        return time_index, space_indexes
    
    def calc_data_on_grid(self, data_func, data_grid_name, data_indices, *args):
        if len(args) == 1:
            spacetime_index = args[0]
        elif len(args) == 2:
            time_index, space_index = args[0], args[1]
            spacetime_index = (time_index, *space_index)
        else:
            spacetime_index = args
        
        data_grid = getattr(self, data_grid_name)
        if not data_grid.mask[(*data_indices, *spacetime_index)]:
            return data_grid[(*data_indices, *spacetime_index)]
        else:
            data_out = data_func(data_indices, spacetime_index)
            data_grid[(*data_indices, *spacetime_index)] = data_out
            data_grid.mask[(*data_indices, *spacetime_index)] = False
            
        return data_out
    
    def deviation_data_func(self, deviation_indices, *args):
        if len(args) == 1:
            spacetime_index = args[0]
        elif len(args) == 2:
            time_index, space_index = args[0], args[1]
            spacetime_index = (time_index, *space_index)
        else:
            spacetime_index = args
        
        event = np.array([self.grid[i][spacetime_index[i]] for i in range(len(spacetime_index))])
        return self.deviation_field(deviation_indices)(event)
    
    def deviation(self, deviation_indices, *args):
        i,j = deviation_indices
        if j > i: #symmetry! means we don't store the same thing twice, nor do we compute it twice.
            placeholder = j
            j = i
            i = placeholder
            
        return self.calc_data_on_grid(self.deviation_data_func, 'deviation_grid', (i,j), *args)
        
    def christoffel_data_func(self, christoffel_indices, *args):
        if len(args) == 1:
            spacetime_index = args[0]
        elif len(args) == 2:
            time_index, space_index = args[0], args[1]
            spacetime_index = (time_index, *space_index)
        else:
            spacetime_index = args
        
        k,i,j = christoffel_indices
        l=k #as otherwise \eta^{k,l}=0 so the contribution to \Gamma^{k}_{a,b} [in the linear approximation] is 0.
        diff_1 = grid_deriv(self.deviation, self.grid, i, spacetime_index, data_indices=(j,l))
        
        diff_2 = grid_deriv(self.deviation, self.grid, j, spacetime_index, data_indices=(i,l))
        
        diff_3 = grid_deriv(self.deviation, self.grid, l, spacetime_index, data_indices=(i,j))
        
        christoffel_out = .5 * flat_metric((k,l)) * ( diff_1 + diff_2 - diff_3 )
        return christoffel_out
    
    def christoffel(self, christoffel_indices, *args):
        k,i,j = christoffel_indices # \Gamma^{k}_{i,j}
        if j > i: #symmetry! means we don't store the same thing twice, nor do we compute it twice.
            placeholder = j
            j = i
            i = placeholder
            
        return self.calc_data_on_grid(self.christoffel_data_func, 'christoffel_grid', (k,i,j), *args)
    
    def advance_a_grid_or_data(self, grid_or_data_name):
        if grid_or_data_name=='grid':
            advance_by = self.keep_time_size-1
            self.full_time_arr.mask[self.lowest_kept_time_index : self.lowest_kept_time_index + advance_by] = True
            self.full_time_arr.mask[self.lowest_kept_time_index + advance_by : self.lowest_kept_time_index + advance_by + self.keep_time_size] = False
            self.grid[0] = self.full_time_arr[~self.full_time_arr.mask]
            self.lowest_kept_time_index += advance_by
        else:
            data_grid = getattr(self, grid_or_data_name)

            spacetime_dim = self.grid_dim
            space_dim = spacetime_dim - 1

            last_time_slice_index_tuple = (..., -1, *[0]*space_dim)
            first_time_slice_index_tuple = (..., 0, *[0]*space_dim)
            middle_time_slice_index_tuple = (..., np.s_[1:], *[0]*space_dim)
            
            data_grid[first_time_slice_index_tuple] = data_grid[last_time_slice_index_tuple]
            data_grid.mask[middle_time_slice_index_tuple] = False
            
            setattr(self, grid_or_data_name, data_grid)
            
            
    def advance(self, names_to_advance=['grid', 'deviation_grid', 'christoffel_grid']):
        for name in names_to_advance:
            self.advance_a_grid_or_data(name)


def make_scalar(tensor_field_two_indices):
    def scalar(event):
        running = 0
        for i in range(np.shape(tensor_field_two_indices)[0]):
            running += flat_metric((i,i)) * tensor_field_two_indices((i,i))
        return running
    return scalar


def make_christoffel_field(deviation_field, dx=.001): #as far as i know, christoffel symbols *are* actually tensors, it's just that the *it's a different tensor* in a different frame, so one must use a non-tensorial transform when changing frame. 
    def christoffel_field(indices, deviation_field=deviation_field, dx=dx):
        def comp_func(event, indices=indices, deviation_field=deviation_field, dx=dx):
            k,i,j = indices # \Gamma^{k}_{i,j}
            l=k #as otherwise \eta^{k,l}=0 so the contribution to \Gamma^{k}_{a,b} [in the linear approximation] is 0.
            to_diff_1 = deviation_field((j,l))
            to_diff_2 = deviation_field((i,l))
            to_diff_3  = deviation_field((i,j))
            running = .5 * flat_metric((k,l)) * ( deriv(to_diff_1,i, dx=dx)(event) + deriv(to_diff_2,j, dx=dx)(event) - deriv(to_diff_3,l, dx=dx)(event) )

            return running
        return comp_func
    return christoffel_field

def make_ricci_field(deviation_or_christoffel_field, dx=.001, mode='deviation'):
    def ricci_field(indices, deviation_or_christoffel_field=deviation_or_christoffel_field, dx=dx, mode=mode):
        def comp_func(event, indices=indices, deviation_or_christoffel_field=deviation_or_christoffel_field, dx=dx, mode=mode):
            i,j = indices
            running = 0
            if mode=='deviation':
                deviation_field = deviation_or_christoffel_field=deviation_or_christoffel_field
                deviation_scalar = make_scalar(deviation_field)
                    
                part_4 = partial(partial(deviation_scalar, j, dx=dx), i, dx=dx)(event)
                running = 0
                for k in range(0,4):
                    to_diff_1 = partial(deviation_field(i,k), j, dx=dx)
                    to_diff_2 = partial(deviation_field(j,k), i, dx=dx)
                    to_diff_3 = partial(deviation_field(i,j), k, dx=dx)
                    running += .5*partial(to_diff_1, k, dx=dx)(event) + .5*partial(to_diff_2, k, dx=dx)(event) - .5*partial(to_diff_3, k, dx=dx)(event) - .5*part_4
                return running
                    
            elif mode=='christoffel':
                christoffel_field = deviation_or_christoffel_field=deviation_or_christoffel_field
                for k in range(0,4):
                    to_diff_1 = christoffel_field((k,i,j))
                    to_diff_2 = christoffel_field((k,k,j))
                    running += partial(to_diff_1, k, dx=dx)(event) - partial(to_diff_2, i)(event)
                    
            return running
        
        return comp_func
    return ricci_field


def make_einstein_field(deviation_or_christoffel_or_ricci_field, dx=.001, mode='deviation'):
    if mode=='deviation':
        ricci_field = make_ricci_field(deviation_or_christoffel_or_ricci_field, dx=dx, mode='deviation')
    elif mode=='christoffel':
        ricci_field = make_ricci_field(deviation_or_christoffel_or_ricci_field, dx=dx, mode='christoffel')
    elif mode=='ricci':
        ricci_field = deviation_or_christoffel_or_ricci_field
    
    ricci_scalar = make_scalar(ricci_field)

    def einstein_field(indices, ricci_field=ricci_field, ricci_scalar=ricci_scalar):
        def comp_func(event, indices=indices, ricci_field=ricci_field, ricci_scalar=ricci_scalar):
            i,j = indices
            return ricci_field((i,j))(event) - .5*flat_metric((i,j))*ricci_scalar(event)
        return comp_func
    return einstein_field
        


def make_geodesic_diffy(christoffel_field_or_mesh, mode='mesh'): #important: coordinate velocity is not the four velocity! it's (e.g. in a coordinate system not using proper time)
    def geodesic_diffy(time, space_poses, space_vels, christoffel_field_or_mesh=christoffel_field_or_mesh, mode=mode):
        if len(np.shape(space_poses)) == 1:
            one_particle_mode = True
        else:
            one_particle_mode = False
        
        if not one_particle_mode:
            space_dim = np.shape(space_poses)[1] #this way doing 2+1 D dynamics for a system constrained to a plane is as simple as giving 2+1 D vectors.
        else:
            space_dim = len(space_poses)
        spacetime_dim = space_dim+1
        
        if not one_particle_mode:
            spacetime_poses = np.insert(space_poses, 0, np.full(np.shape(space_poses)[0], time, dtype='float32'), axis=1)
            spacetime_vels = np.insert(space_vels, 0, np.ones(np.shape(space_poses)[0], dtype='float32'), axis=1)
        else:
            spacetime_poses = np.concatenate([[time], space_poses], dtype='float32')
            spacetime_vels = np.concatenate([[1], space_vels], dtype='float32')
            
        space_second_derivs = np.zeros_like(space_poses, dtype='float32')
        
        for a in range(1,spacetime_dim): #a is the index of components_of_second_derivative. a=0 always results in zero second derivative.
            for i in range(spacetime_dim):
                for j in filter(lambda j: j<=i, range(spacetime_dim)):
                        #since \Gamma^{k}_{i,j} is symmetric in the bottom, and since we have products of v^i v^j and no other i,j terms, we can just multiply the result for (i,j) by 2 if i!=j.
                        if i!=j:
                            double_count_factor = 2
                        else:
                            double_count_factor = 1
                        if mode=='field':
                            christoffel_comp_part_1 = christoffel_field_or_mesh((a, i, j))
                            christoffel_comp_part_2 = christoffel_field_or_mesh((0, i, j))
                            
                            if not one_particle_mode:
                                space_second_derivs[:, a-1] += double_count_factor*(- np.apply_along_axis(christoffel_comp_part_1, 1, spacetime_poses) * spacetime_vels[:, i] * spacetime_vels[:, j] + np.apply_along_axis(christoffel_comp_part_2, 1, spacetime_poses) * spacetime_vels[:, i] * spacetime_vels[:, j] * spacetime_vels[:, a])
                            else:
                                space_second_derivs[a-1] += double_count_factor*(- christoffel_comp_part_1(spacetime_poses) * spacetime_vels[i] * spacetime_vels[j] + christoffel_comp_part_2(spacetime_poses) * spacetime_vels[i] * spacetime_vels[j] * spacetime_vels[a])
                        elif mode=='mesh':
                            mesh = christoffel_field_or_mesh
                            time_indice, space_indexes = mesh.find(time, space_poses)
                            christoffel_comp_part_1 = lambda space_index: mesh.christoffel((a, i, j), time_indice, space_index)
                            christoffel_comp_part_2 = lambda space_index: mesh.christoffel((0, i, j), time_indice, space_index)
                            
                            if not one_particle_mode:
                                space_second_derivs[:, a-1] += double_count_factor*(- np.apply_along_axis(christoffel_comp_part_1, 1, space_indexes) * spacetime_vels[:, i] * spacetime_vels[:, j] + np.apply_along_axis(christoffel_comp_part_2, 1, space_indexes) * spacetime_vels[:, i] * spacetime_vels[:, j] * spacetime_vels[:, a])
                            else:
                                space_second_derivs[a-1] += double_count_factor*(- christoffel_comp_part_1(space_indexes) * spacetime_vels[i] * spacetime_vels[j] + christoffel_comp_part_2(space_indexes) * spacetime_vels[i] * spacetime_vels[j] * spacetime_vels[a])
                        
        return space_second_derivs
    return geodesic_diffy


def solve_geodesic(deviation_or_christoffel_field_or_mesh, init_space_poses, init_space_vels, init_time, end_time, dx=.001, dt=.0001, yield_every_nth=1, mode='mesh'): #important: dt is the time step in time=x^0 units, not the proper time step.

    if mode=='deviation':
        christoffel_field_or_mesh = make_christoffel_field(deviation_or_christoffel_field_or_mesh, dx=dx)
        mode='field'
    elif mode=='christoffel':
        christoffel_field_or_mesh = deviation_or_christoffel_field_or_mesh
        mode='field'
    elif mode=='mesh':
        christoffel_field_or_mesh = deviation_or_christoffel_field_or_mesh
        
    this_diffy = make_geodesic_diffy(christoffel_field_or_mesh, mode=mode)
    solution_generator = solve_diffyq(this_diffy, init_space_poses, init_space_vels, start_indep=init_time, end_indep=end_time, indep_step=dt, yield_every_nth=yield_every_nth)
    return solution_generator


def null_field(indices):
    return lambda event: 0

def make_newton_potential(source_point=[0,0,0], source_mass=1):
    def newton_potential(point):
        return -source_mass/(np.dot(point,point)**.5)
    return newton_potential
    
origin_potential = make_newton_potential()
def newton_deviation(indices, newton_potential=origin_potential): #if given less dims, pads on the right with 0. e.g. if given 2-space coords, we assume we are working on the z=0 plane.
    def comp_func(event, indices=indices):
        i,j = indices
        space_pos = event[1:]
        if i!=j:
            return 0
        else:
            return -2 * newton_potential(space_pos)
    return comp_func




def nbody_frame_update(time_and_poses, artists, plot_3D=False, trail=50):
    time, poses, vels = time_and_poses[0], time_and_poses[1], time_and_poses[2]
    
    time_text_artist = artists[-2]
    total_vel_text_artist = artists[-1]
    line_artists = artists[:-2]
    
    total_velocity = np.sum(vels, axis=0)
    total_velocity_magnitude = np.linalg.norm(total_velocity)
    time_text_artist.set_text(f"Time: {round(time,5)}")
    total_vel_text_artist.set_text(f"Total Velocity: {round(total_velocity_magnitude,3)}")
    
    for particle_index in range(len(line_artists)):
        line_artist = line_artists[particle_index]
        this_pos = poses[particle_index]
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
        
    return [*line_artists, time_text_artist, total_vel_text_artist]
     
def animate(time_and_poses_gen, num_particles, frame_count, frame_interval, window_one_axis=[-20, 20], plot_3D=False, trail=50):
    
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
    total_vel_text_artist = ax.text(.75,1, "", transform=ax.transAxes)
    ani = FuncAnimation(fig, partial(nbody_frame_update, artists=[*line_artists, time_text_artist, total_vel_text_artist], trail=trail), frames=time_and_poses_gen, save_count=frame_count, interval=frame_interval, blit=True)
    return ani

def solve_and_animate_geodesic(deviation_or_christoffel_field_or_mesh, init_poses, init_vels, init_time=0, end_time=5, dx=.01, dt=.01, window_one_axis=[-20, 20], slowdown=1, plot_every_nth=1, mode='mesh'):
    time_and_poses_gen = solve_geodesic(deviation_or_christoffel_field_or_mesh, init_poses, init_vels, init_time=init_time, end_time=end_time, dx=dx, dt=dt, yield_every_nth=plot_every_nth, mode=mode) #important: dt is the coordinate time step, not the proper time step.
    
    frame_interval = 1000*dt*slowdown*plot_every_nth
    frame_count = math.ceil((end_time - init_time)/(plot_every_nth*dt))
    num_particles = len(init_poses)
    
    ani = animate(time_and_poses_gen, num_particles, frame_count, frame_interval, window_one_axis=window_one_axis)
    return ani


if __name__ == '__main__':
    init_time = 0
    end_time = 100
    dx = .001
    dt = .0001
    #dt = .00025
    slowdown = .1 #1
    plot_every_nth = 100 #150
    window_one_axis = [-7,7]#[-150,150]
    

    
    '''
    deviation_field = null_field#newton_deviation
    grid, find = make_grid(time_window=[init_time, end_time], space_1d_window=window_one_axis, space_dim=2, num_t=1000, num_x=1000)
    null_spacetime = SpacetimeMesh(deviation_field, 2, grid, find)
    init_poses = np.array([ [-3,0], [3,0] ], dtype='float32')
    init_vels = np.array([ [0,1], [0,-1] ], dtype='float32')
    
    end_time = 10
    dt = .01
    slowdown = 5
    plot_every_nth = 100
    window_one_axis = [-10,10]
    
    ani = solve_and_animate_geodesic(null_spacetime, init_poses, init_vels, init_time=init_time, end_time=end_time, dx=dx, dt=dt, window_one_axis=window_one_axis, slowdown=slowdown, plot_every_nth=plot_every_nth)
    ani.save('results/linearized_null_dev.mp4')
    '''
    
    
    deviation_field = newton_deviation
    grid, find, (dt,dx) = make_grid(time_window=[init_time, end_time], space_1d_window=window_one_axis, space_dim=2, dt=dt, dx=dx)
    newton_spacetime = SpacetimeMesh(deviation_field, 2, grid, find)
    init_poses = np.array([ [5,0], ], dtype='float32')
    #init_vels = np.array([ [0, .2], ], dtype='float32') 
    init_vels = np.array([ [-0.01,  0.5], ], dtype='float32')  
    ani = solve_and_animate_geodesic(newton_spacetime, init_poses, init_vels, init_time=init_time, end_time=end_time, dx=dx, dt=dt, window_one_axis=window_one_axis, slowdown=slowdown, plot_every_nth=plot_every_nth)
    ani.save('results/lin_newton_ellipse_long.mp4')
    
    
    '''
    deviation_field = newton_deviation
    grid, find, (dt,dx) = make_grid(time_window=[init_time, end_time], space_1d_window=window_one_axis, space_dim=2, dt=dt, dx=dx)
    newton_spacetime = SpacetimeMesh(deviation_field, 2, grid, find)
    init_poses = np.array([ [5,0], ], dtype='float32')
    init_vels = np.array([ [0,  0.5], ], dtype='float32')  
    ani = solve_and_animate_geodesic(newton_spacetime, init_poses, init_vels, init_time=init_time, end_time=end_time, dx=dx, dt=dt, window_one_axis=window_one_axis, slowdown=slowdown, plot_every_nth=plot_every_nth)
    ani.save('results/lin_newton_profiling.mp4')
    '''
    
    '''
    deviation_field = newton_deviation
    grid, find, (dt,dx) = make_grid(time_window=[init_time, end_time], space_1d_window=window_one_axis, space_dim=2, dt=dt, dx=dx)
    newton_spacetime = SpacetimeMesh(deviation_field, 2, grid, find)
    #init_poses = np.array([ [100,0], ], dtype='float32')
    init_poses = np.array([ [1000,0], ], dtype='float32')
    #init_vels = np.array([ [0,.1], ], dtype='float32')  
    init_vels = np.array([ [0,.01], ], dtype='float32')  
    ani = solve_and_animate_geodesic(newton_spacetime, init_poses, init_vels, init_time=init_time, end_time=end_time, dx=dx, dt=dt, window_one_axis=window_one_axis, slowdown=slowdown, plot_every_nth=plot_every_nth)
    ani.save('results/lin_newton_ellipse_slow_velocity.mp4')
    '''
    
    '''
    init_poses = np.array([ [-3,0], [3,0] ], dtype='float32')
    init_vels = np.array([ [0,1], [0,-1] ], dtype='float32')
    ani = solve_and_animate_geodesic(null_field, init_poses, init_vels, init_time=init_time, end_time=end_time, dx=dx, dt=dt, window_one_axis=window_one_axis, slowdown=slowdown, plot_every_nth=plot_every_nth)
    #ani.save('results/linearized_null_dev.mp4')
    '''
    
    '''
    slowest_v = .69
    fastest_v = .75
    num_v = 20
    init_poses = np.repeat([[-3.,0]], num_v, axis=0)
    init_vels = np.zeros_like(init_poses, dtype='float32')
    vel_mags = np.concatenate([np.linspace(slowest_v,fastest_v,num_v)], dtype='float32')
    init_vels[:,1] = vel_mags
    #init_poses = np.array([[-3,0],[3,0]], dtype='float32')
    #init_vels = np.array([[0,.75],[0,-.75]], dtype='float32')
    ani = solve_and_animate_geodesic(deviation_field, init_poses, init_vels, init_time=init_time, end_time=end_time, dx=dx, dt=dt, window_one_axis=window_one_axis, slowdown=slowdown, plot_every_nth=plot_every_nth)
    ani.save('results/lin_newton_circle_many.mp4')
    '''
    
    '''
    init_poses = np.array([ [100,0], ], dtype='float32')
    init_vels = np.array([ [0,.1], ], dtype='float32')
    ani = solve_and_animate_geodesic(deviation_field, init_poses, init_vels, init_time=init_time, end_time=end_time, dx=dx, dt=dt, window_one_axis=window_one_axis, slowdown=slowdown, plot_every_nth=plot_every_nth)
    ani.save('results/lin_newton_circle.mp4')
    '''
    
    