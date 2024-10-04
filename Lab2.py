#!/usr/bin/env python3
'''
This file implements two ODE solvers for the two types of 
Lotka Volterra equations.

To get solution for lab 2, run these commands:

%run Lab2.py

'''
#import essential libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys

#Turn on interactive mode for plotting:
plt.ion()
#Creating function for the Lotka-Volterra Predator-Prey equations:
def dNdt_predatorprey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator prey equations for two 
    species. Given normalized populations, `N1` and `N2`, as well as the four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.

    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]
    return dN1dt, dN2dt 


#Creating a function for the Lotka-Volterra Competition Equations:
def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    
    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]
    return dN1dt, dN2dt 

#Defining the function for the Euler solver:
def euler(func, N1_init=.3, N2_init=.6, dT=0.05, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    Given a function representing the first derivative of f, an
    initial condiition f0 and a timestep, dt, solve for f using
    Euler's method.

    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init : float, initial condition for species 1.
    N2_init : float, initial condition for species 2.
    dT : float, change in timestep for the euler solver in years. 
    t_final : float, default=100
    Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
    Lotka-Volterra coefficient values

    Returns
    -----------
    time : Numpy array
    Time elapsed in years.
    N1, N2 : Numpy arrays
    Normalized population density solutions.   
    '''

    # Initialize time and population density solutions for each species:
    time = np.arange(0.0, t_final+dT, dT)
    
    N_1 = np.zeros(time.size)
    N_2 = np.zeros(time.size)
    
    #Set up initial conditions:
    N_1[0] = N1_init
    N_2[0] = N2_init

    # Integrate!
    for i in range(0, time.size-1):
        dN1, dN2 = func(i+1, [N_1[i], N_2[i]], a=a,b=b,c=c,d=d)
        N_1[i+1] = N_1[i]+dN1*dT
        N_2[i+1] = N_2[i]+dN2*dT

    # Return values to caller:
    return time, N_1, N_2

#Defining the function for the Runge-Kutta Solver:
def solve_rk8(func, N1_init=0.3, N2_init=0.6, dT=1, t_final=100.0,
a=1, b=2, c=1, d=3):
    '''
    Solve a single ODE using the Dormand Prince 8th order adaptive RK
    method with dense output (using Scipy's `solve_ivp` function).

    Arguments are the same as `euler`:  Given a function representing the first derivative of f, an
    initial condiition f0 and a timestep, dt, solve for f using the RK method.
 

    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
    Largest timestep allowed in years.
    t_final : float, default=100
    Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
    Lotka-Volterra coefficient values

    Returns
    -------
    time : Numpy array
    Time elapsed in years.
    N1, N2 : Numpy arrays
    Normalized population density solutions.    
    '''

    from scipy.integrate import solve_ivp

    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
        args=[a, b, c, d], method='DOP853', max_step=dT)

    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    # Return values to caller.
    return time, N1, N2

#Defininting varaibles to represent the function inputs for the two ODE solvers:
#These commands also prompt the user to enter values for each argument.
#arg_1 corresponds to initial value for N1:
arg_1 = input("Enter arg_1: ") 
#arg_2 corresponds to the initial value for N2:
arg_2 = input("Enter arg_2: ") 
#arg_3 corresponds to dT for the euler solver for the predator-prey model
arg_3 = input("Enter arg_3: ") 
#arg_4 corresponds to dT for the Runge-Kutta solver and for the euler solver for the competition model:
arg_4 = input("Enter arg_4: ") 
#Prompting the user to enter values for coefficients a, b, c, and d:
a = float(input("Enter coefficient a: "))
b = float(input("Enter coefficient b: "))
c = float(input("Enter coefficient c: "))
d = float(input("Enter coefficient d: "))

#Calculating the equilibrium values algebraically based on the inputs entered by the user: 
denom = c * a - b * d
if denom != 0: #if the denominator is equal to zero, there is no equilibrium value.
    N1_eq = c * (a - b) / denom
    N2_eq = a * (c - d) / denom
else:
    N1_eq = None
    N2_eq = None

#Set sys.argv for inputs (excluding the coefficients):
sys.argv = ['Lab2.py', arg_1, arg_2, arg_3, arg_4]
#Converts values each argument to type float:
arg_1, arg_2, arg_3, arg_4 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

#Running the two ODE solvers for each Lotka-Volterra model:
time_euler_pp, N_1_euler_pp, N_2_euler_pp = euler(dNdt_predatorprey, N1_init=arg_1, N2_init=arg_2, dT=arg_3, a=a, b=b, c=c, d=d)
time_rk8_pp, N_1_rk8_pp, N_2_rk8_pp = solve_rk8(dNdt_predatorprey, N1_init=arg_1, N2_init=arg_2, dT=arg_4, a=a, b=b, c=c, d=d)
time_euler_comp, N_1_euler_comp, N_2_euler_comp = euler(dNdt_comp, N1_init=arg_1, N2_init=arg_2, dT=arg_4, a=a, b=b, c=c, d=d)
time_comp_rk8, N_1_rk8_comp, N_2_rk8_comp = solve_rk8(dNdt_comp, N1_init=arg_1, N2_init=arg_2, dT=arg_4, a=a, b=b, c=c, d=d)

#Plotting the Lotka-Volterra Predator-Prey Model:
plt.clf() #Clearing figure to not have overlap from previous runs of the code file.
plt.plot(time_euler_pp, N_1_euler_pp, color='deepskyblue', label='N1 (Prey) Euler')
plt.plot(time_euler_pp, N_2_euler_pp, color='red', label="N2 (Predator) Euler")
plt.plot(time_rk8_pp, N_1_rk8_pp, color='deepskyblue', linestyle=':', label="N1 (Prey) RK8")
plt.plot(time_rk8_pp, N_2_rk8_pp, color='red', linestyle=':', label="N2 (Predator) RK8")

plt.xlabel('Time (years)')  
plt.ylabel('Population Carrying Capacity')  
plt.title('Lotka-Volterra Predator-Prey Model') 
plt.xlim(0, 100)
plt.ylim(0, 2.5)
plt.legend() 
plt.savefig('LVplot.png') #Saving Predator-Prey model plot
plt.show()

#Creating plot for Lotka-Volterra Competition Model:
plt.clf()  # Clear the  figure from the last code run for the competition model
plt.plot(time_euler_comp, N_1_euler_comp, color='deepskyblue', label='N1 Competition Euler')
plt.plot(time_euler_comp, N_2_euler_comp, color='red',label="N2 Competition Euler")
plt.plot(time_comp_rk8, N_1_rk8_comp, color='deepskyblue', linestyle=':', label="N1 Competition RK8")
plt.plot(time_comp_rk8, N_2_rk8_comp, color='red', linestyle=':',label="N2 Competition RK8")
# Add equilibrium lines for reference on this plot (this was added while trying to determine equilibrium)
if N1_eq is not None and N2_eq is not None:
    plt.axhline(y=N1_eq, color='orange', linestyle='--', label='N1 Equilibrium')
    plt.axhline(y=N2_eq, color='yellow', linestyle='--', label='N2 Equilibrium')
plt.xlabel('Time (years)')
plt.ylabel('Population Carrying Capacity')
plt.title('Lotka-Volterra Competition Model')
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.legend()
plt.savefig('Competition_LVplot.png')  # Save the competition model plot
plt.show()

#Plotting the phase diagrams for the predator prey: 
plt.clf() #Clearing the plot from the last code run
#We plot the results from the Runge-Kutta solver because this provides
# a more accurate depiction of the phase diagram:
plt.plot(N_1_rk8_pp, N_2_rk8_pp, label="RK8") 
plt.xlabel('Prey Species Population')  
plt.ylabel('Predator Species Population')  
plt.title('Lotka-Volterra Predator-Prey Model Phase Diagram')
plt.savefig("Predator_Prey_Phase_Diagram.png") #Saving phase diagram plot
plt.show()
