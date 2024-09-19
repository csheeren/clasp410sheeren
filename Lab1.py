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
        Set the maximum number of iterations including initial condition.
    pbare: float, defaults to 0
        Probability that a cell is bare to begin with from 0 to 1 (0 to 100%).
    pspread : float, defaults to 1
        Chance fire spreads from 0 to 1 (0 to 100%).
    pfire: float, defaults to 0.  
        Probability that a square is on fire from 0 to 1 (0 to 100%).
    center_square: boolean, true if center square is on fire and
        false otherwise.        
    '''

    # Create forest and set initial condition
    forest = np.zeros([maxiter, nNorth, nEast]) + 2
    
    #ignite = np.random.rand(nNorth, nEast)
    #Create color map for plotting:
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    
    

    # Ignite center square only.
    if center_square is True:
        forest[0, nNorth//2, nEast//2] = 3
    #If center square is not ignited:
    else:
        # Randomly set bare spots using pbare:
        bare_or_immune = np.random.rand(nNorth,nEast) < pbare
        forest[0, bare_or_immune] = 1

        # Randomly ignite portions of the forest using pfire:
        burning_or_infected = np.random.rand(nNorth, nEast) < pfire 

        forest[0, burning_or_infected] = 3
    #Plot initial condition:
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    contour = ax.pcolor(forest[0, :, :], cmap=forest_cmap, vmin=1, vmax=3)
    ax.set_title(f'Forest Status, Iteration = {0:03d}')
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
       
        #Creating plots for each timestep:
        fig, ax = plt.subplots(1, 1, figsize=(10,7))
        contour = ax.pcolor(forest[k+1, :, :], cmap=forest_cmap, vmin=1, vmax=3)
        ax.set_title(f'Forest Status, Iteration = {k+1:03d}')
        cbar=plt.colorbar(contour, ax=ax, ticks=[1.33, 2, 2.67])
        cbar.ax.set_yticklabels(['Bare', 'Forested', 'Fire'])
        ax.set_ylabel('y (km)')
        ax.set_xlabel('x (km)')
        fig.savefig(f'fig{k+1:03d}.png')
        plt.close('all')

        # Quit if no spots are on fire.
        nBurn = (forest[k+1, :, :] == 3).sum()
        if nBurn == 0:
            print(f"Burn completed in {k+1} steps")
            break

    return k+1


def explore_burnrate():
    ''' Vary burn rate and see how fast fire ends.'''

    prob = np.arange(0, 1.05, .05)
    nsteps = np.zeros(prob.size)

    for i, p in enumerate(prob):
        print(f"Buring for pspread = {p}")
        nsteps[i] = fire_spread(nNorth= 72, nEast=72, pspread=p, maxiter=1000)
        plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    ax.scatter(prob, nsteps)
    ax.set_xlabel('pspread')
    ax.set_ylabel('Number of Iterations')
    ax.set_title('Number of Iterations vs. pspread')
    ax.set_xlim(0,1)
    ax.set_ylim(0,300)
    fig.savefig('Pspreadplot.png')



def p_bare_plot():
    '''Create a plot showing the relationship between the number of iterations
    until a grid is completely bare and the probability of a grid being bare 
    to begin with. '''

    pbare_range = np.arange(0,1.05,0.05)
    iterations = np.zeros(pbare_range.size)

    for i,p in enumerate(pbare_range):
        print(f"Burning for pbare = {p}")
        iterations[i] = fire_spread(nNorth=72, nEast=72, pspread=0.2, pbare=p, maxiter=1000)
        plt.close('all')
        #print(iterations)
    fig, ax = plt.subplots(1, 1, figsize=(10,7))
    ax.scatter(pbare_range, iterations)
    ax.set_xlim(0,1)
    ax.set_ylim(0,300)
    ax.set_xlabel('pbare')
    ax.set_ylabel('Number of Iterations')
    ax.set_title('Number of Iterations vs. pbare')
    fig.savefig('Pbareplot.png')
    













