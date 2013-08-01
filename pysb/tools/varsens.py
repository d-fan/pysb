
import pysb.bng
import ghalton
import numpy
import sys

def varsens(model, solver, objective, k, n, scaling, log_scaling=False, verbose=True):
    if verbose: print "Generating Halton sequences"
    seq = ghalton.Halton(k) # half for A, and half for B
    seq.get(2*(k*k-k)) # Burn away any face exploration off the Halton
    M_1  = scale(numpy.array(seq.get(n)), scaling, log_scaling)  # See Eq (9)
    M_2  = scale(numpy.array(seq.get(n)), scaling, log_scaling)  # See Eq (9)
    N_j  = generate_N_j(M_1, M_2)                                # See Eq (11)
    N_nj = generate_N_j(M_2, M_1)
    
    (fM_1, fM_2, fN_j, fN_nj) = objective_values(model, M_1, M_2, N_j, N_nj, objective, solver, verbose) 
    
    if verbose: print "Final sensitivity calculation"
    return getvarsens(fM_1, fM_2, fN_j, fN_nj)

def scale(points, scaling, log_scaling):
    if log_scaling:
# FIXME, I THINK THIS IS ALL BACKWARD, Ugh.
        s = numpy.exp(scaling)
        return numpy.log(points*(s[1]-s[0]) + s[0])
    else:
        return points * (scaling[1] - scaling[0]) + scaling[0]

def move_spinner(i):
    spin = ("|", "/","-", "\\")
    print "[%s] %d\r"%(spin[i%4],i),
    sys.stdout.flush()

def generate_N_j(M_1, M_2):
    """when passing the quasi-random low discrepancy-treated A and B matrixes, this function
    iterates over all the possibilities and returns the C matrix for simulations.
    See e.g. Saltelli, Ratto, Andres, Campolongo, Cariboni, Gatelli, Saisana,
    Tarantola Global Sensitivity Analysis"""

    nparams = M_1.shape[1] # shape 1 should be the number of params

    # allocate the space for the C matrix
    N_j = numpy.array([M_2]*nparams) 

    # Now we have nparams copies of M_2. replace the i_th column of N_j with the i_th column of M_1
    for i in range(nparams):
        N_j[i,:,i] = M_1[:,i]

    return N_j

def objective_values(model, M_1, M_2, N_j, N_nj, objective, solver, verbose=True): #, fileobj=None):
    ''' Function parmeval calculates the fM_1, fM_2, and fN_j_i arrays needed for variance-based
    global sensitivity analysis as prescribed by Saltelli and derived from the work by Sobol
    (low-discrepancy sequences)
    '''

    # assign the arrays that will hold fM_1, fM_2 and fN_j_n
    fM_1  = numpy.zeros(M_1.shape[0])
    fM_2  = numpy.zeros(M_2.shape[0])
    fN_j  = numpy.zeros([M_1.shape[1]] + [M_1.shape[0]]) # matrix is of shape (nparam, nsamples)
    fN_nj = numpy.zeros([M_1.shape[1]] + [M_1.shape[0]])

    # First process the A and B matrices
    if verbose: print "Processing f(M_1):"
    for i in range(M_1.shape[0]):
        fM_1[i]   = objective(solver(model, M_1[i]))
        if verbose: move_spinner(i)

    if verbose: print "Processing f(M_2):"
    for i in range(M_2.shape[0]):
        fM_2[i]   = objective(solver(model, M_2[i]))
        if verbose: move_spinner(i)

    if verbose: print "Processing f(N_j)"
    for i in range(N_j.shape[0]):
        if verbose: print " * parameter %d"%i
        for j in range(N_j.shape[1]):
            fN_j[i][j] = objective(solver(model, N_j[i][j]))
            if verbose: move_spinner(j)

    if verbose: print "Processing f(N_nj)"
    for i in range(N_j.shape[0]):
        if verbose: print " * parameter %d"%i
        for j in range(N_j.shape[1]):
            fN_nj[i][j] = objective(solver(model, N_nj[i][j]))
            if verbose: move_spinner(j)

    return fM_1, fM_2, fN_j, fN_nj

def getvarsens(fM_1, fM_2, fN_j, fN_nj):
    """Calculate the array of S_i and ST_i for each parameter given fM_1, fM_2, fN_j matrices
    from the multi-sampling runs. Calculate S_i and ST_i as follows:

    Parameter sensitivity:
    ----------------------
            U_j - E^2 
    S_j = ------------                   # Eq (10)
               V(y)

    U_j = 1/(n-1) \sum fM_1 * fN_j       # Eq (12)

    E^2 = 1/n \sum fM_1 * fM_2           # Eq (21)

    Total effect sensitivity (i.e. non additive part):
    --------------------------------------------------
                  U_-j - E^2
     ST_j = 1 - -------------            # Eq (27)
                      V(y)

    U_-j = 1/(n-1) \sum fM_2 * fN_j


    In both cases, calculate V(y) from fM_1 and fM_2


    """

    nparms   = fN_j.shape[0] # should be the number of parameters
    nsamples = fN_j.shape[1] # should be the number of samples from the original matrix

    E_2 = sum(fM_1*fM_2) / nsamples      # Eq (21)

    # Estimate 
    U_j  = numpy.sum(fM_1 * fN_j,  axis=1) / (nsamples - 1)  # Eq (12)
    U_nj = numpy.sum(fM_1 * fN_nj, axis=1) / (nsamples - 1)  # Eq (unnumbered one after 18)

    #estimate V(y) from fM_1 and fM_2

    varfM_1 = numpy.var(fM_1, axis=0, ddof=1)
    varfM_2 = numpy.var(fM_2, axis=0, ddof=1)

    #allocate the S_i and ST_i arrays
    Sens  = numpy.zeros(nparms)
    SensT = numpy.zeros(nparms)

    # now get the U_j and U_-j values and store them 
    for j in range(nparms):
        Sens[j]  =       (U_j[j]  - E_2)/varfM_1
        SensT[j] = 1.0 - (U_nj[j] - E_2)/varfM_1

    return Sens, SensT


# Working on a test function here

# This is defined on the range [0..1]
# Eq (29)
def g_function(x, a):
    return numpy.prod([gi_function(i, xi, a) for xi,i in enumerate(x)])

# Eq (30)
def gi_function(xi, i, a):
    return (abs(4*xi-2)+a[i]) / (1+a[i])


model = [0, 0.5, 3, 9, 99, 99]

# Analytical answer, Eq (34)
answer = 1.0/(3.0* ((numpy.array(model) + 1.0)**2.0))

def g_solver(model, params): return g_function(params, model)

def g_objective(x): return x

v = varsens(model, g_solver, g_objective, 6, 1024, numpy.array([[0.0]*6, [1.0]*6]))