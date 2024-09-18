#!/usr/bin/env python3

'''
This file performs fire/disease spread simulations.

To get solution for lab 1: Run these commands:

>>> blah
>>> blah blah

'''
#import essential libraries:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use('fivethirtyeight')




def fire_spread(nNorth=3, nEast=3, maxiter=4, pbare=0, pspread=1.0, pfire=0, center_square =True):
    '''
    This function performs a fire/disease spread scenterimultion.

    Parameters
    ==========
    nNorth, nEast : integer, defaults to 3
        Set the north-south (i) and east-west (j) size of grid.
        Default is 3 squares in each direction.
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition
    pspread : float, defaults to 1
        Chance fire spreads from 0 to 1 (0 to 100%).
    pbare: float, defaults to 0
        Percentage of the non forested cells in the grid (0 to 100%). 
    center_square: boolean, true if center square is on fire and
        false otherwise.        
    '''

    # Create forest and set initial condition
    forest = np.zeros([maxiter, nNorth, nEast]) + 2
    
    # Set fire! To the center of the forest.
    #istart, jstart = nNorth//2, nEast//2
    #forest[0, istart, jstart] = 3
    ignite = np.random.rand(nNorth, nEast)
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    # Plot initial condition
    

    # Ignite center square only.
    if center_square is True:
        forest[0, nNorth//2, nEast//2] = 3
    else:
        # Randomly set bare spots using pbare.
        bare_or_immune = np.random.rand(nNorth,nEast) < pbare
        forest[0, bare_or_immune] = 1

        # Randomly ignite portions of the forest
        burning_or_infected = np.random.rand(nNorth, nEast) < pfire 

        forest[0, burning_or_infected] = 3

    fig, ax = plt.subplots(1, 1)
    contour = ax.pcolor(forest[0, :, :], cmap=forest_cmap, vmin=1, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
    cbar=plt.colorbar(contour, ax=ax, ticks=[1.33, 2, 2.67])
    cbar.ax.set_yticklabels(['Bare', 'Forested', 'Fire'])
    ax.set_ylabel('y (km)')
    ax.set_xlabel('x (km)')
    fig.savefig(f'fig{0:04d}.png')
    #plt.show()
    

    # Propagate the solution.
    for k in range(maxiter-1):
        # Set chance to burn:
        ignite = np.random.rand(nNorth, nEast)

        # Use current step to set next step:
        forest[k+1, :, :] = forest[k, :, :]

        # Burn from north to south:
        doburn = (forest[k, :-1, :] == 3) & (forest[k, 1:, :] == 2) & \
            (ignite[:-1, :] <= pspread)
        forest[k+1, 1:, :][doburn] = 3

         # Burn from south to north:
        doburn = (forest[k, 1:, :] == 3) & (forest[k, :-1, :] == 2) & \
            (ignite[1:, :] <= pspread)
        forest[k+1, :-1, :][doburn] = 3
        
        # From east to west
        for i in range(nNorth):
            for j in range(nEast-1):
                # Is current patch burning AND adjacent forested?
                if (forest[k, i, j] == 3) & (forest[k, i, j+1] == 2):
                    # Spread fire to new square:
                    forest[k+1, i, j+1] = 3

        # From west to east
        for i in range(nNorth):
            for j in range(1, nEast):
                # Is current patch burning AND adjacent forested?
                if (forest[k, i, j] == 3) & (forest[k, i, j-1] == 2):
                    # Spread fire to new square:
                    forest[k+1, i, j-1] = 3

        # Set currently burning to bare:
        wasburn = forest[k, :, :] == 3  # Find cells that WERE burning
        forest[k+1, wasburn] = 1       # ...they are NOW bare.
       
        
        fig, ax = plt.subplots(1, 1)
        contour = ax.pcolor(forest[k+1, :, :], cmap=forest_cmap, vmin=1, vmax=3)
        ax.set_title(f'Iteration = {k+1:03d}')
        cbar=plt.colorbar(contour, ax=ax, ticks=[1.33, 2, 2.67])
        cbar.ax.set_yticklabels(['Bare', 'Forested', 'Fire'])
        ax.set_ylabel('y (km)')
        ax.set_xlabel('x (km)')
        fig.savefig(f'fig{k+1:03d}.png')

        # Quit if no spots are on fire.
        nBurn = (forest[k+1, :, :] == 3).sum()
        if nBurn == 0:
            print(f"Burn completed in {k+1} steps")
            break

    return k+1


def explore_burnrate():
    ''' Vary burn rate and see how fast fire ends.'''

    prob = np.arange(0, 1, .05)
    nsteps = np.zeros(prob.size)

    for i, p in enumerate(prob):
        print(f"Buring for pspread = {p}")
        nsteps[i] = fire_spread(nEast=3, pspread=p, maxiter=100)

    plt.plot(prob, nsteps)

def explore_bare():
    '''Vary amount of non-forested cells from 0 to 100%'''
    
    prob = np.arange(0, 1, .05)
    nsteps = np.zeros(prob.size)







