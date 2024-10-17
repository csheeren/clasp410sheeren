#!/usr/bin/env python3
'''
This file contains a set of tools for solving the N-layer atmosphere energy
balance problem and perform useful analysis.

To reproduce figures in lab write-up, run these commands:

run Lab3.py
Figure 1: emissivity_var(temp_goal=288)
Figure 2: find_temp(50, 288, 0.255, debug=False)
          n_layer_atmos(5, 0.255, debug=False)
Figure 3: n_layer_atmos(5, 0.5, debug=False, nuclear_winter=True)
'''

import numpy as np
import matplotlib.pyplot as plt

#Define some useful constants here:
sigma = 5.67E-8 #Steffan-Boltzman constant.

def n_layer_atmos(N, epsilon, S0= 1350, albedo=0.33, debug=False, nuclear_winter=False):
    '''
    Solve the n-layer atmosphere problem and return temperature at each layer.

    Parameters
    ==========
    N: int
        Set the number of layers
    epsilon: float, default=1.0
        Set the emisivity of the atmospheric layers.
    albedo: float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0: float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2
    debug: boolean, default=False
        Turn on debug output. 
    nuclear_winter: boolean, default=False
        Set to true if it is a nuclear winter, false otherwise. 

    Returns
    --------
    temps: Numpy array of size N+1 
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest.
    '''

#Create matrices:
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)
    b[0] = -S0/4 * (1-albedo)
#Modifying initial matrices based on if it is a nuclear winter:
    if nuclear_winter:
        b[-1] = -S0/4 * (1-albedo)
        b[0] = -S0/4
#Debug statement:
    if debug:
        print(f"Populating N+1 X N+1 matrix (N = {N})")
#Populating our A matrix:
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            #Diagonal elements are always -2 (except at the Earth's surface)
            if i == j:
                #print(f"i>0 -> {i>0}")
                A[i,j] = -1*(i>0) - 1
                #print(f"Result: A[i,j] = {A[i,j]}")
            else:
                #This is the pattern we solved for in class:
                m = np.abs(j-i) - 1
                A[i,j] = epsilon * (1-epsilon)**m
    #At Earth's surface, epsilon=1, breaking our pattern.
    #Divide by epsilon along the surface to get correct results.
    A[0, 1:] /= epsilon

    #Verify our A matrix:
    if debug:
        print(A)
    #Get the inverse of our A matrix.
    Ainv = np.linalg.inv(A)
    #Multiply Ainv by b to get Fluxes.
    fluxes = np.matmul(Ainv, b)
    #Convert fluxes to temperatures.
    #Fluxes for all atmospheric layers:
    #flux = epsilon * sigma * T^4
    temps = (fluxes/epsilon/sigma)**0.25
    temps[0] = (fluxes[0]/sigma)**0.25 #Flux at ground, where epsilon = 1.
    layer_number = np.arange(len(temps))
    #Plotting the Altitude profile for a modeled system:
    plt.clf()
    plt.plot(temps, layer_number) 
    plt.xlabel('Temperature (K)')  
    plt.ylabel('Altitude')  
    plt.title('Altitude Profile of Modeled System')
    plt.savefig("Altitude Profile.png") #Saving phase diagram plot
    plt.show()
    return temps

#Creating a function to vary the emissivity: 
def emissivity_var(temp_goal,**kwargs): #if wanting to use debug, need to specify here
    '''
    Solves for the temperature of the earth given a 1 layer atmosphere for a 
    particular emisivity value in the range from 0 to 1.
    Parameters
    ==========
    temp_goal: int
    Set the current surface temperature.
    epsilon: float, default=1.0
        Set the emisivity of the atmospheric layers.
    albedo: float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0: float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2
    debug: boolean, default=False
        Turn on debug output.
    Returns
    --------
    temperatures_earth_np: Numpy array of size N+1 
        Array of temperatures at the Earth's surface (element 0) corresponding to a particular 
        emissivity value ranging from 0 to 1.
    '''
    temperatures_earth = [] #Creating an empty list that will build a list of temperatures
    emissivity = np.arange(0.01,1.045,0.045) #Creating an list of emisivity values ranging from 0 to 1, 0.05 apart
    for i in emissivity:
        earth = n_layer_atmos(1, i, **kwargs)
        #Adding on every value from the function for surface temperature based on emissivity value into the temperature array.
        temperatures_earth.append(earth[0])
    temperatures_earth_np = np.array(temperatures_earth)
    #Calculating the difference between the values in the temperature array and the current surface temperature:
    temperatures_diff_earth = np.abs(temperatures_earth_np-temp_goal)
    earth_288 = np.abs(temperatures_diff_earth)
    #Determining emissivity of the atmosphere:
    #First determine index where the absolute value difference is smallest:
    earth_emissivity_index = np.argmin(earth_288)
    #Find the value in the emissivity array at this index:
    earth_emissivity = emissivity[earth_emissivity_index]
    print("Emissivity is")
    print(earth_emissivity)
  
    #Plotting the surface temperature versus emissivity:
    plt.clf()
    plt.plot(emissivity, temperatures_earth_np) 
    plt.xlabel('Emissivity')  
    plt.ylabel('Surface Temperature (K)')  
    plt.title('Surface Temperature Versus Emissivity')
    plt.savefig("Emissivity_Variation.png")
    plt.show()
    

    return temperatures_earth_np, earth_emissivity

#Creating a function that solves for the number of layers based on a given surface temperature:
def find_temp(max_iter, temp_goal, epsilon, **kwargs): #if wanting to use debug, need to specify here
    '''
    Solves for the number of layers in the atmosphere based on a particular surface temperature. 
    Parameters
    ==========
    max_iter: int
        Sets the maximum possible amount of layers of the atmosphere.
    temp_goal: int
        Set the current surface temperature.
    epsilon: float, default=1.0
        Set the emisivity of the atmospheric layers.
    albedo: float, default=0.33
        Set the planetary albedo from 0 to 1.
    S0: float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2
    debug: boolean, default=False
        Turn on debug output. 
    Returns
    --------
    num_layers: int
        Number of layers of the atmosphere expected on a planet given a surface temperature.
    '''
    temperatures_venus = [] #Creating an empty list that will build a list of temperatures
    for i in range(max_iter):
        venus = n_layer_atmos(i, epsilon, **kwargs)
         #Adding on every value from the function for surface temperature based on number of layers into the temperature array.
        temperatures_venus.append(venus[0]) 
    temperatures_venus_np = np.array(temperatures_venus)
    #Calculating the difference between the values in the temperature array and the current surface temperature:
    temperatures_diff = np.abs(temperatures_venus_np-temp_goal)
    venus_700 = np.abs(temperatures_diff)
    #Determining number of layers in the atmosphere based on index corresponding to the smallest absolute value difference:
    num_layers = np.argmin(venus_700)
    print(num_layers)

    return num_layers
