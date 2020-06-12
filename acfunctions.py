from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
from pandas import DataFrame
import matplotlib.pyplot as plt
import astropy.stats as astat
from solartwins import *
import pickle

#McQuillan function with adjustable resolution 
def ac_mcquillan(star, num_points): 
    varied_k = [] #array of autocorrelated fluxes for a given star with varied lag time 
    flux = star.flux[0:] 
    N = len(flux)
    time = star.time[0:]
    dt = time[1] - time[0] #cadence
    k = np.linspace(5, N/2, num_points, dtype = int) #lag time
    flux_bar = np.average(flux)
    
    
    #equation for McQuillan's AC function
    for i, v in enumerate(k): 
        numerator = []
        for ii in range(N - k[i]): 
            num = (flux[ii]-flux_bar)*(flux[ii + k[i]] - flux_bar)  
            numerator.append(num)
            
        denominator = []
        for jj in range(N):
            den = (flux[jj]-flux_bar)**2
            denominator.append(den)
            
        rk = (np.sum(numerator)/np.sum(denominator))
        varied_k.append(rk)
    
    tau = k * dt

    return varied_k, tau 


#McQuillan et al Autocorrelation Function with the number of points fixed at 100

def ac_mcquillan_set_k(star): 
    varied_k = [] #array of autocorrelated fluxes for a given star with varied lag times
    flux = star.flux[0:] 
    N = len(flux) 
    time = star.time[0:]
    dt = time[1] - time[0] #cadence
    k = np.linspace(5, N/2, 100, dtype = int) #lag time
    flux_bar = np.average(flux)
    
    #equation for McQuillan's AC function
    for i, v in enumerate(k): 
        numerator = []
        for ii in range(N - k[i]): 
            num = (flux[ii]-flux_bar)*(flux[ii + k[i]] - flux_bar)  
            numerator.append(num)
            
        denominator = []
        for jj in range(N):
            den = (flux[jj]-flux_bar)**2
            denominator.append(den)
            
        rk = (np.sum(numerator)/np.sum(denominator))
        varied_k.append(rk)
    
    tau = k * dt 

    return varied_k, tau 
