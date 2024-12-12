#!/usr/bin/env python 3
'''
This file contains the code for the pressure profile analysis.

To reproduce figures in the lab report, use the following commands:
Question 1: run final_project.py
            question_1()
Question 2: question_2()
Question 3: question_3()
Question 4: question_4()
            plot_temperature_profile()

'''
#Import essential libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use('fivethirtyeight')

#Define constants:
g = 9.81 #gravitational constant in m/s^2
R_d = 287 #ideal gas constant in J/(kg*K)
t_0 = 288.15 #Average global surface temperature in K
p_0 = 101325 #mean sea level pressure in Pa
z_span = (0, 40000) #z values
L = 0.0098 #Dry adiabatic lapse rate in K/m

def get_lapse(z):
    '''
    Defines the lapse rate based on height in the atmosphere for 
    a convective environment.
    Parameters
    ----------
    z : int
        Height above the surface in meters.     

    Returns
    --------
    An int correponding to the lapse rate at a particular height.
    '''
    #A parcel of air will follow the moist adiabatic lapse rate in the 10000m closest to the surface:
    if (z >= 0 and z <= 10000):
        return 0.0058
    #Defining a transition zone between moist adiabatic lapse rate and dry adiabatic lapse rate:
    elif (z > 10000 and z <= 11000):
        return (0.0058 + ((z - 10000) * (0.004 / 1000)))
    #Parcel follows dry adiabatic lapse rate in the rest of the tropopause:
    elif (z > 11000 and z <= 40000):
        return 0.0098
    else:
        return 0.0098


#To set up our model, we create three functions:
def differential_output(z, p):
    '''
    Defines the output for the pressure profile in an isothermal atmosphere.
    Parameters
    ----------
    z : int
        Height above the surface in meters. 
    p : int
        Atmospheric pressure in hectopascals.

    Returns
    -------
    dp_dz : numpy array of float
        Differential equation values showing the change in pressure 
        with respect to height.
    '''
    num = -p * g
    denom = R_d * t_0
    dp_dz = num/denom
    return dp_dz

def differential_output_vary(z, p):
    '''
    Defines the output for the pressure profile in an atmosphere with
    a constant lapse rate.

    Parameters
    ----------
    z : int
        Height above the surface in meters. 
    p : int
        Atmospheric pressure in hectopascals.

    Returns
    -------
    dp_dz_vary : numpy array of floats
        Values set equal to the differential equation showing the change in pressure
        with respect to height.
    '''
    num = -p * g
    #Creating a formula for temperature with a constant lapse rate:
    temp_lapse = t_0 + (L * z)
    denom = R_d * temp_lapse
    dp_dz_vary = num/denom
    return dp_dz_vary

def differential_output_vary_lapse(z, p):
    '''
    Defines the output for the pressure profile in an atmosphere with
    a varying lapse rate.

    Parameters
    ----------
    z : int
        Height above the surface in meters. 
    p : int
        Atmospheric pressure in hectopascals.

    Returns
    -------
    dp_dz_vary : numpy array of floats
        Values set equal to the differential equation showing the change in pressure
        with respect to height.
    '''
    num = -p * g
    #Creating a formula for temperature with a constant lapse rate:
    lapse = get_lapse(z)
    temp_lapse = t_0 + (lapse * z)
    denom = R_d * temp_lapse
    dp_dz_vary = num/denom
    return dp_dz_vary

def solve_diff(vary, varyLapse, stepSize: int):
    '''
    Solves the differential equation for the atmospheric pressure profile.

    Parameters
    ----------
    vary: boolean
        Set to true when the temperature is varying with height,
        set false when the atmosphere is isothermal.  
    varyLapse: boolean
        Set to true when the lapse rate is not constant, 
        set to false otherwise.
    stepSize: cast as int
        Distance dz utilied in integration of the ordinary differential equation.

    Returns
    -------
    heights_lapse: numpy array of floats.
        Array of height values in meters for an atmosphere with a varying lapse rate.
    pressure_values_lapse: numpy array of floats. 
        Array of pressure values in Pascals at the corresponding height values for 
        an atmosphere with a varying lapse rate.
    heights: numpy array of floats 
        Array of height values above the surface in meters.
    pressure_values: numpy array of floats.
        Array of pressure values in Pascals for the atmospheric pressure profile.
    heights_vary: numpy array of floats.
        Array of height values above the surface for an atmosphere with varying
        temperatures. 
    pressure_values_vary: numpy array of floats.
        Array of pressure values in Pascals at the corresponding height values
        for an atmosphere with varying temperatures.
    '''
    numStep = int(40000 / stepSize) + 1
    #Creating the atmospheric pressure profile model for an atmosphere with a varying lapse rate:
    if(varyLapse == True):
        hydrostatic_solution_vary_lapse = solve_ivp(differential_output_vary_lapse, z_span, [p_0], t_eval=np.linspace(0,40000,numStep))
        heights_lapse = hydrostatic_solution_vary_lapse.t
        pressure_values_lapse = hydrostatic_solution_vary_lapse.y[0]
        return heights_lapse, pressure_values_lapse
    
    #Creating the atmopsheric profile for an isothermal atmosphere:
    else:
        if(vary == False):
            hydrostatic_solution = solve_ivp(differential_output, z_span, [p_0], t_eval=np.linspace(0,40000,numStep))
            heights = hydrostatic_solution.t
            pressure_values = hydrostatic_solution.y[0]
            return heights, pressure_values
        #Creating the atmospheric pressure profile model for an atmosphere with a varying temperature:
        else:
            hydrostatic_solution_vary = solve_ivp(differential_output_vary, z_span, [p_0], t_eval=np.linspace(0,40000,numStep))
            heights_vary = hydrostatic_solution_vary.t
            pressure_values_vary = hydrostatic_solution_vary.y[0]
            return heights_vary, pressure_values_vary

#Function calls that will be utilized throughout analysis:
height_vary, pressure_vary = solve_diff(True, False, 10)
heights_iso, pressure_iso = solve_diff(False, False, 10)
height_lapse, pressure_lapse = solve_diff(False, True, 10)


#Varying the step size for integration:
step_sizes = [10000, 4000, 2000, 100, 10]
profiles_step_vary = []


for step in step_sizes:
    height_step, pressure_step = solve_diff(False, False, step)
    profiles_step_vary.append((height_step, pressure_step))

#Plotting the atmospheric profile with varying step sizes:
plt.clf()
num_profiles = len(profiles_step_vary)
cols = 3
rows = (num_profiles + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True, sharey=True)

axes = axes.flatten()
for i, (height_step, pressure_step) in enumerate(profiles_step_vary):
    axes[i].plot(pressure_step/100, height_step)
    axes[i].set_title(f"Step Size: {step_sizes[i]}", fontsize=16)
    axes[i].set_xlabel("Pressure (hPa)")
    axes[i].set_ylabel("Height (m)")
    axes[i].grid(True)

for ax in axes[num_profiles:]:
    ax.axis("off")

plt.suptitle('Atmospheric Pressure Profiles in an Isothermal Atmosphere with Varying Step Sizes', fontsize=20)
plt.tight_layout()
plt.show()

#y --> p (dependent variable)
#t --> z (independent variable)


def question_1():
    '''
    Plotting the atmospheric pressure profile for an isothermal atmosphere
    with varying step sizes.
    '''
    #Plotting the atmospheric pressure profile with an isothermal atmosphere:
    plt.plot(pressure_iso/100, heights_iso) #Dividing pressure values by 100 to comvert back to hPa.
    plt.xlabel("Pressure (hPa)")
    plt.ylabel("Height (m)")
    plt.title("Pressure Profile for an Isothermal Atmosphere", fontsize=16)
    plt.tight_layout()
    plt.show()

    return

def question_2():
    '''
    Determining the scale height in meters for the atmospheric pressure profile
    in an isothermal atmosphere.
    '''
    #Determining location of scale height:
    pressure_scale_height = p_0 * (1/np.e) #Scale height is the height where the pressure is reduced by a factor of 1/e.
    #Determine the index where the scale height occurs:
    idx_scale = (np.abs(pressure_iso - pressure_scale_height)).argmin()
    scale_height = heights_iso[idx_scale]
    scale_height_pressure = pressure_iso[idx_scale]
    print(f'The scale height of this pressure profile is {scale_height} m.')
    print(f'The pressure at the scale height of this pressure profile is {scale_height_pressure} Pa.')

    return

def question_3():
    '''
    Creating plot and calculating scale height for atmospheric pressure profile
    with a constant lapse rate. 
    '''
    #Plotting the atmospheric profile for an atmospheric with a constant lapse rate:
    plt.clf()
    plt.plot(pressure_vary/100, height_vary) #Dividing pressure values by 100 to comvert back to hPa.
    plt.xlabel("Pressure (hPa)")
    plt.ylabel("Height (m)")
    plt.title("Pressure Profile for an Atmosphere with a Constant Lapse Rate", fontsize=14)
    plt.tight_layout()
    plt.show()

    #Determining location of scale height for this atmospheric profile:
    pressure_scale_height = p_0 * (1/np.e) #Scale height is the height where the pressure is reduced by a factor of 1/e.
    #Determine the index where the scale height occurs:
    idx_scale = (np.abs(pressure_vary - pressure_scale_height)).argmin()
    scale_height = height_vary[idx_scale]
    scale_height_pressure = pressure_vary[idx_scale]
    print(f'The scale height of this pressure profile is {scale_height} m.')
    print(f'The pressure at the scale height of this pressure profile is {scale_height_pressure} Pa.')

    return

def question_4():
    '''
    Plotting the pressure profile for an atmosphere with 
    a varying lapse rate. 
    '''
    #Plotting the atmospheric pressure profile with a varying lapse rate:
    plt.clf()
    plt.plot(pressure_lapse/100, height_lapse) #Dividing pressure values by 100 to comvert back to hPa.
    plt.xlabel("Pressure (hPa)")
    plt.ylabel("Height (m)")
    plt.title("Pressure Profile for an Atmosphere with a Varying Lapse Rate", fontsize=14)
    plt.tight_layout()
    plt.show()

    #Determining location of scale height for this atmospheric profile:
    pressure_scale_height = p_0 * (1/np.e) #Scale height is the height where the pressure is reduced by a factor of 1/e.
    #Determine the index where the scale height occurs:
    idx_scale = (np.abs(pressure_lapse - pressure_scale_height)).argmin()
    scale_height = height_lapse[idx_scale]
    scale_height_pressure = pressure_lapse[idx_scale]
    print(f'The scale height of this pressure profile is {scale_height} m.')
    print(f'The pressure at the scale height of this pressure profile is {scale_height_pressure} Pa.')

    return

def plot_temperature_profile():
    '''
    Plotting the temperature profile for an atmopshere with a varying lapse rate. 
    '''
    #Plotting the temperature profile for an atmosphere with  varying lapse rate:
    z_list = np.linspace(0,40000,81)
    temp_list = []
    for z in z_list:
        new_temp = t_0 - (z * get_lapse(z))
        temp_list.append(new_temp)
    
    plt.plot(temp_list, z_list)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Height (m)")
    plt.title("Temperature Profile for an Atmosphere with a Varying Lapse Rate", fontsize=14)
    plt.tight_layout()
    plt.show()

    return