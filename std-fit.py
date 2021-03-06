'''
STD NMR Build Up Curve Solver
Written by Sam Walpole
sam@samwalpole.com
v1.2.0 7/12/18
'''

from math import exp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns



def parse_input(fname):
    '''
    Read a CSV file and return a tuple containing the t sats([0]) and std values([1])
    
    Keyword arguments:
    fname -- the file name to use as input (str)
    '''
    
    inputs = np.loadtxt(fname, delimiter=',', dtype=str, encoding=None)
    return (inputs[1:,0], np.delete(inputs,0,axis=1).astype(float))
    
    
    
def solve_build_up_curves(t_sats, exp_stds):
    '''
    Take a set of experimental STD measurements and return smooth predicted build up curves
    
    Keyword arguments:
    t_sats -- a set of experimental t sat values (array(float))
    exp_stds -- 2D array of experimental STD values, where each row is one build up curve (ndarray(float))
    '''
    
    #solve std max and k sat for each row of std values
    solved_params = solve_params(t_sats, exp_stds)
    save_results(titles, solved_params)
    
    #use solved parameters to calculate build up curve with many data points
    smooth_t_sats = calc_tsats_smooth((t_sats[0], t_sats[len(t_sats) - 1]))
    
    #recursively add a build up curve from each solved set of parameters
    #to the accumalator and return it
    def inner(solved_params, prev):
        build_up_curve = calc_build_up_curve(solved_params[0,0], solved_params[0,1], smooth_t_sats)
        accumulator = np.vstack((prev, build_up_curve))
        if(len(solved_params) == 1):
            return accumulator
        return inner(solved_params[1:], accumulator)
        
    #call the recursive function with smooth_t_sats as the first row in 
    #the accumulator
    return inner(solved_params, smooth_t_sats)
    
    
    
def solve_params(t_sats, exp_stds, prev=None):
    '''
    Take a set of experimental STD measurements and return optimised values for STD max and k sat as an array
    
    Keyword arguments:
    t_sats -- a set of experimental t sat values (array(float))
    exp_stds -- 2D array of experimental STD values, where each row is one build up curve (ndarray(float))
    '''
    def F(x):
        
        #unpack optimised variable into std max and ksat
        std_max, k_sat = x
        
        #calculate build up curve with these parameters
        calc_stds = calc_build_up_curve(std_max, k_sat, t_sats)
        
        #square difference between experimental and calculated
        #first row in exp_stds
        #ignore zero values in calculation
        return sum_square_diffs(exp_stds[0][exp_stds[0] != 0], calc_stds[exp_stds[0] != 0])
    
    mini = minimize(F, [1, 1], bounds=[[0,None],[0,None]]).x
    
    #check if prev already contains values
    #and append current minimisation to accumulator
    if(np.any(prev)):
        accumulator = np.vstack((prev, mini))  
    #if prev is empty then the accumaltor is just this minimisation
    else:
        accumulator = mini
        
    #recursively go through exp_stds    
    if(len(exp_stds) == 1):
        return accumulator
    return solve_params(t_sats, exp_stds[1:], accumulator)
    
    
    
def sum_square_diffs(a,b):
    '''
    Take two arrays and return the sum of the square differences for each element
    
    Keyword arguments:
    a -- first array to use (array(float))
    b -- second array to use (array(float))
    '''
    
    return np.sum((a - b)**2)
    
    
    
def save_results(titles, params):
    '''
    Take proton titles, STD max and k sat, and create a CSV file containing STD max, k sat and STD0 for each proton
    
    Keyword arguments:
    titles -- an array of each proton title (array(str))
    params -- an array containing STD max ([0]) and k sat ([1]) (array(float))
    '''
    
    header = 'Proton,STDmax,ksat,STD0'
    
    #write titles as columns
    t_titles = titles.reshape(len(titles), 1)
    
    #calculate std0 from optimised std max and k sat
    std_zeros = np.apply_along_axis(lambda p:p[0] * p[1], 1, params)
    
    #write std0s as columns
    t_std_zeros = std_zeros.reshape(len(std_zeros), 1)
    
    save_csv('results.csv', np.hstack([t_titles, params, t_std_zeros]), header)
    
    
    
def save_csv(fname, data, header):
    '''
    Take an array of data and save it as a CSV within given filename and header
    
    Keyword arguments:
    fname -- the file name to save the data to (str)
    data -- an array containing the data to be printed (ndarray(str))
    header -- a string to be printed at the top of the file (str)
    '''
    
    np.savetxt(fname, data, '%s', ',', header=header)
    
    
    
def calc_tsats_smooth(t_sat_range, num_points=300):
    '''
    Take a min and max t sat value and create array with equally spaced points between (inclusive) those values
    
    Keyword values:
    t_sat_range -- minimum ([0]) and maximum ([1]) t sat values (tuple)
    num_points -- number of data points to create in range (int, default 300)
    '''
    
    step = (t_sat_range[1]-t_sat_range[0]) / num_points
    return np.arange(t_sat_range[0], t_sat_range[1] + step, step)
    
    
    
def calc_build_up_curve(std_max, k_sat, t_sats):
    '''
    Take STD max, k sat and a set of t sat values and returns an array of corresponding STD values
    
    Keyword arguments:
    std_max -- value of STD max (float)
    k_sat -- value of k sat (float)
    t_sats -- array of t sat values (array)
    '''
    
    #calculate std value for given t sat
    std = np.vectorize(lambda t_sat:std_max * (1 - exp(-1 * k_sat * t_sat)))
    return std(t_sats)
    
    
    
def plot_curves(exp_tsats, exp_stds, pred_tsats, pred_stds, labels, fname='build_up_curves'):
    '''
    Take experimental and predicted stds and t_sats and plot them as a graph to file
    
    Keyword arguments:
    exp_tsats -- 1D array containing saturation times used in the experiment (array(float))
    exp_stds -- 2D array with rows containing experimental STD build up curves (ndarray(float))
    pred_tsats -- 1D array containing smoothed t_sats used in curve prediction (array(float))
    pred_stds -- 2D array with rows containing predicted STD build up curves (ndarray(float))
    labels -- 1D array with labels for each row of stds (array(str)
    fname -- name of file to save figure to (str)
    '''
    
    sns.set_context('notebook')
    sns.set_style('ticks')
    sns.set_palette('colorblind')
        
    def plot(exp_tsats, exp_stds, pred_tsats, pred_stds, labels):
        
        #plot the predicetd values as a smooth line
        sns.lineplot(pred_tsats, pred_stds[0])
        
        #plot the experimental values as points
        #remove zero values from plot
        sns.scatterplot(exp_tsats[exp_stds[0] != 0], exp_stds[0, exp_stds[0] != 0], label=labels[0]) #TODO remove zero points
        
        #recursively plot graphs
        if(len(exp_stds) == 1):
            return
        return plot(exp_tsats, exp_stds[1:], pred_tsats, pred_stds[1:], labels[1:])
        
    plot(exp_tsats, exp_stds, pred_tsats, pred_stds, labels)
    
    plt.legend(loc='best')
    plt.xlabel('Saturation Time (s)')
    plt.ylabel('STD Intensity (%)')
    plt.savefig(fname + '.png')    



titles, exp_stds = parse_input('inputs.csv')
predicted_stds = solve_build_up_curves(exp_stds[0], exp_stds[1:])

plot_curves(exp_stds[0], exp_stds[1:], predicted_stds[0], predicted_stds[1:], titles)

