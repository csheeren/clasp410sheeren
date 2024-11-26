#!/usr/bin/env python 3
'''
This file contains the code for the snowball earth simulation.

To reproduce figures in the lab report, use the following commands:
Question 1: run Lab5.py
            test_snowball()
Question 2: vary_lambda()
            vary_emiss()
            warm_earth_equilibrium()
Question 3: initial_condition_vary()
Question 4: solar_forcing_impact()

'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#Some constants:
radearth = 6357000 #Earth's radius in meters
mxdlyr = 50 #depth of mixed layer (m)
sigma = 5.67e-8 #Steffan Boltzman constant
C = 4.2e6 #Heat capacity of water
rho = 1020 #Density of sea water (kg/m^3)
albedo_ice = 0.6 #Albedo of ice when dynmaic albedo is used
albedo_gnd = 0.3 #Albedo of the ground when dynamic albedo is used, default albedo value otherwise


def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 latitude (where 0 is South Pole, 180 is north)
    where each returned point represents the cell center

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes
    '''

    dlat = 180/nbins #latitude spacing
    lats = np.arange(0, 180, dlat) + dlat/2

    #Alternative way to obtain grid:
    #lats = np.linspace(dlat/2, 180-dlat/2, nbins)
    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celsius.
    '''

    #Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats_in, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation

def snowball_earth(nbins=18, dt=1, tstop=10000, lam=100, spherecorr=True,
                   debug=False, albedo=0.3, emiss=1, S0=1370, dynamic_albedo=False, 
                   init_temp=None, gamma=1):
    '''
    Perform snowball earth simulation.

    Paramters
    ---------
    nbins : int, defaults to 18
        Number of latitude bins
    dt: float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn  on or off debug print statements.
    albedo : float, defaults to 0.3
        Set the earth's albedo.
    emiss : float, defaults to 1.0
        Set ground emissivity. Set to zero to turn off radiative cooling.
    S0 : float, defaults to 1370
        Set incoming solar forcing constant. Change to zero to turn off insolation. 
    dynamic_albedo : boolean, defaults to False.
        Set to True if albedo is varying based on ground conditions.
    init_temp : object, defaults to None.
        Initial temperature of the Earth in degrees Celsius. 
    gamma: float, defaults to 1.
        Solar multiplier factor applied to insolation term,
        utilized to explore the impact of solar forcing on the Earth.

    Returns
    -------
    lats: Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude
    
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    if dynamic_albedo: 
        albedo = np.zeros(len(lats))
    # Generate insolation:
    insol = gamma * insolation(S0, lats)

    # Create initial condition:
    if init_temp is None:
        Temp = temp_warm(lats)
    #If temp_init is an array being passed in of varying values:
    elif isinstance(init_temp, (list, tuple, np.ndarray)):
        Temp = init_temp
    #If temp_init is the same number:
    else:
        Temp = np.full(len(lats), init_temp, dtype=float)
    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    for i in range(nstep):
        # Update albedo based on conditions:
        if dynamic_albedo:
            loc_ice = Temp <= -10
            albedo[loc_ice] = albedo_ice
            albedo[~loc_ice] = albedo_gnd
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)

        #Apply insolation and radiative losses:
        #print('T before insolation:', Temp)
        radiative = (1-albedo) * insol - emiss*sigma*(Temp+273.15)**4
        #print('\t Rad term = ', dt_sec * radiative / (rho*C*mxdlyr))
        Temp += dt_sec * radiative  / (rho * C * mxdlyr)
        #print('\t T after rad:', Temp)

        Temp = np.matmul(L_inv, Temp)
        

    return lats, Temp


def test_snowball(tstop=10000):
    '''
    Reproduce example plot in lecture/handout.

    Using our DEFAULT values (grid size, diffusion, etc.) and a warm-Earth
    initial condition, plot:
        - Initial condition
        - Plot simple diffusion only
        - Plot simple diffusion + spherical correction
        - Plot simple diff + sphere corr + insolation
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)

    # Get simple diffusion solution:
    lats, t_diff = snowball_earth(tstop=tstop, spherecorr=False, S0=0, emiss=0)

    # Get diffusion + spherical correction:
    lats, t_sphe = snowball_earth(tstop=tstop, S0=0, emiss=0)
    
    #Get diffusion + spherical correction + radiative term:
    lats, t_rad = snowball_earth(tstop=tstop)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(lats, initial, label='Warm Earth Init. Cond.')
    ax.plot(lats, t_diff, label='Simple Diffusion')
    ax.plot(lats, t_sphe, label='Diffusion + Sphere. Corr.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative')
    ax.set_title("Snowball Earth Simulation With Varying Included Terms", fontsize=16)
    ax.set_xlabel('Latitude (0°=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')

    ax.legend(loc='best')

    fig.tight_layout()
    fig.show()

def vary_lambda():
    '''
    Varies the value of lambda from 0 to 150, keeping all other parameters constant.

    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #Creating array of lambda values from 0 to 150, each 15 apart.
    lam_values = np.linspace(0, 150, 11)
    for lam in lam_values:
        lats_lam, t_lam = snowball_earth(lam=lam)
        ax.plot(lats_lam, t_lam, label=lam)
    
    initial_temp = temp_warm(lats_lam)
    #Plotting the temperature versus latitude curve for all of the lambda array values:
    ax.plot(lats_lam, initial_temp, label='Warm Earth Equilibrium', color='black')
    ax.set_xlabel('Latitude (0°=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)') 
    ax.set_title("Snowball Earth Simulation with Varying Lambda Values", fontsize=16)   
    ax.legend(loc='best')
    fig.tight_layout()
    fig.show()
    #When emissivity is 1: the shape of the curve is closest to warm earth at diffusivity constant of 30.

def vary_emiss():
    '''
    Varies the value of emissivity from 0 to 1, keeping all other parameteres constant. 

    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #Creating an array of emissivity values from 0 to 1, each 0.1 apart. 
    emiss_values = np.linspace(0, 1, 11)
    for emiss in emiss_values:
        lats_emiss, t_emiss = snowball_earth(emiss=emiss)
        ax.plot(lats_emiss, t_emiss, label=emiss)
    
    initial_temp = temp_warm(lats_emiss)
    #Plotting the temperature versus latitude for all of the emissivity array values:
    ax.plot(lats_emiss, initial_temp, label='Warm Earth Equilibrium', color='black')
    ax.set_title("Snowball Earth Simulation with Varying Emissivity Values", fontsize=16)
    ax.set_xlabel('Latitude (0°=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')    
    ax.legend(loc='best')
    fig.tight_layout()
    fig.show()
    #When diffusivity is set to 100: emissivity is closest to warm earth curve between 0.7 and 0.8.

def warm_earth_equilibrium():
    '''
    Create plot to reproduce the warm earth equilibrium curve,
    with lambda and emissivity values found through utilizing the 
    vary_lam() and vary_emiss() functions.
    '''
    #Running the snowball earth simulation with our determined equilibrium lambda and emissivity values:
    lats_opt, t_opt = snowball_earth(lam=45, emiss=0.725) 
    initial_temp = temp_warm(lats_opt)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #Plotting the warm earth snowball earth simulation curve with our reproduced warm earth simulation curve:
    ax.plot(lats_opt, initial_temp, label='Warm Earth Equilibrium Curve')
    ax.plot(lats_opt, t_opt, label="Warm Earth Reproduction Curve")
    ax.set_xlabel('Latitude (0°=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')
    ax.set_title("Warm Earth Equilibrium for the Snowball Earth Simulation", fontsize=16)
    ax.legend(loc='best')

    fig.tight_layout()
    fig.show()

def initial_condition_vary():
    '''
    Creating plot for the snowball earth simulation with varying initial 
    conditions.
    '''
    #Running snowball earth simultion for a "hot" earth:
    lats_warm, temp_warm = snowball_earth(lam=45, emiss=0.725, dynamic_albedo=True, init_temp=60)
    #Running the snowball earth simulation for a "cold" earth:
    lats_cold, temp_cold = snowball_earth(lam=45, emiss=0.725, dynamic_albedo=True, init_temp=-60)
    #Running the snowball earth simulation for the "flash freeze" earth scenario:
    lats_06, temp_06 = snowball_earth(lam=45, emiss=0.725, albedo=0.6)

    #Plotting the snowball earth simulation for the three above scenarios:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(lats_warm, temp_warm, label='Warm Earth')
    ax.plot(lats_cold, temp_cold, label='Cold Earth')
    ax.plot(lats_06, temp_06, label="Albedo = 0.6")
    ax.set_xlabel('Latitude (0°=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)') 
    ax.set_title("Snowball Earth Simulation with Varying Initial Conditions", fontsize=16)   
    ax.legend(loc='best')
    fig.tight_layout()
    fig.show()
    
    


def solar_forcing_impact():
    '''
    Determining the impact of solar forcing through varying gamma and 
    calculating the associated average global temperatures. 
    '''
    #Creating three empty lists for calculations:
    outputs = []
    average_temperatures_up = []
    average_temperatures_down = []
    #Creating two gamma arrays:
    gamma_array = np.arange(0.4, 1.45, 0.05) #Array for increasing gamma values
    gamma_array_reverse = np.arange(1.35, 0.35, -0.05) #Array for decreasing gamma values, excluding 1.4 to avoid duplication
    
    #taking into account cold earth simulation for initial gamma value of 0.4:
    lats, temperature = snowball_earth(init_temp = -60, lam=45, emiss=0.725, dynamic_albedo=True, gamma=gamma_array[0])
    ave_temp = sum(temperature * np.sin(lats*(np.pi/180)) * 5) / 180
    average_temperatures_up.append(ave_temp)
    outputs.append(temperature)
    #Looping through array of increasing gamma values, excluding 0.4 (from 0.4 to 1.4):
    for i in gamma_array[1:]:
        lats_gamma, temperature_gamma = snowball_earth(init_temp = outputs[-1], lam=45, emiss=0.725, dynamic_albedo=True, gamma=i)
        outputs.append(temperature_gamma)
        #Averaging the global temperature for each gamma value:
        ave_temp = sum(temperature_gamma * np.sin(lats_gamma*(np.pi/180)) * 5) / 180
        average_temperatures_up.append(ave_temp)
    
    #Looping through array of decreasing gamma values, from 1.35 to 0.4:
    for i in gamma_array_reverse[0:]:
        lats_gamma, temperature_gamma = snowball_earth(init_temp = outputs[-1], lam=45, emiss=0.725, dynamic_albedo=True, gamma=i)
        outputs.append(temperature_gamma)
        #Averaging the global temperature for each gamma value:
        ave_temp = sum(temperature_gamma * np.sin(lats_gamma*(np.pi/180)) * 5) / 180
        average_temperatures_down.append(ave_temp)

    #Plotting the average temperature versus gamma for both increasing and decreasing gamma values:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(gamma_array, average_temperatures_up, label='Increasing Gamma Values')
    ax.plot(gamma_array_reverse, average_temperatures_down, label='Decreasing Gamma Values')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Average Global Temperature ($^{\circ} C$)')
    ax.set_title("Average Global Temperature for Varying Gamma", fontsize=16)
    ax.legend(loc='best')
    fig.tight_layout()
