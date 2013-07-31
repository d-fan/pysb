import pysb
from pysb import *
from pysb.bng import *
import ghalton
from pysb.examples.tyson_oscillator import model

def scale(points, scaling, log_scaling):
    if log_scaling:
        s = numpy.exp(scaling)
        return numpy.log(points*(s[1]-s[0]) + s[0])
    else:
        return points * (scaling[1] - scaling[0]) + scaling[0]

def varsens(model, solver, objective, n, scaling, log_scaling=False, verbose=True):

    pysb.bng.generate_equations(model)
    k = len(model.odes)

    if verbose: print "Generating Halton sequences"
    seq = ghalton.Halton(k) # half for A, and half for B
    seq.get(k) # Burn k points to get into better territory of sequence
    ldA = scale(numpy.array(seq.get(n)), scaling, log_scaling)
    ldB = scale(numpy.array(seq.get(n)), scaling, log_scaling)
    ldC = getCmtx(ldA, ldB)
    
    (yA, yB, yC) = parmeval(model, ldA, ldB, ldC, objective, solver, verbose)
    
    if verbose: print "Final sensitivity calculation"
    return getvarsens(yA, yB, yC)


def move_spinner(i):
    spin = ("|", "/","-", "\\")
    print "\r[%s] %d"%(spin[i%4],i),
    sys.stdout.flush()

def genCmtx(ldmtxA, ldmtxB):
    """when passing the quasi-random low discrepancy-treated A and B matrixes, this function
    iterates over all the possibilities and returns the C matrix for simulations.
    See e.g. Saltelli, Ratto, Andres, Campolongo, Cariboni, Gatelli, Saisana,
    Tarantola Global Sensitivity Analysis"""

    nparams = ldmtxA.shape[1] # shape 1 should be the number of params

    # allocate the space for the C matrix
    ldmtxC = numpy.array([ldmtxB]*nparams) 

    # Now we have nparams copies of ldmtxB. replace the i_th column of ldmtxC with the i_th column of ldmtxA
    for i in range(nparams):
        ldmtxC[i,:,i] = ldmtxA[:,i]

    return ldmtxC

def parmeval(model, ldmtxA, ldmtxB, ldmtxC, objective, solver, verbose=True): #, fileobj=None):
    ''' Function parmeval calculates the yA, yB, and yC_i arrays needed for variance-based
    global sensitivity analysis as prescribed by Saltelli and derived from the work by Sobol
    (low-discrepancy sequences)
    '''

    # assign the arrays that will hold yA, yB and yC_n
    yA = numpy.zeros([ldmtxA.shape[0]] + [len(model.observable_patterns)])
    yB = numpy.zeros([ldmtxB.shape[0]] + [len(model.observable_patterns)])
    yC = numpy.zeros(list(ldmtxC.shape[:2]) + [len(model.observable_patterns)]) # matrix is of shape (nparam, nsamples)

    # First process the A and B matrices
    if verbose: print "processing matrix A:"
    for i in range(ldmtxA.shape[0]):
        outlist = solver(model, ldmtxA[i])
        yA[i]   = objective(outlist)
        if verbose: move_spinner(i)

    if verbose: print "processing matrix B:"
    for i in range(ldmtxB.shape[0]):
        outlist = solver(model, ldmtxB[i])
        yB[i]   = objective(outlist)
        if verbose: move_spinner(i)

    if verbose: print "processing matrix C_n"
    for i in range(ldmtxC.shape[0]):
        if verbose: print "processing processing parameter %d"%i
        for j in range(ldmtxC.shape[1]):
            outlist = solver(model, ldmtxC[i][j])
            yC[i][j] = objective(outlist)
            if verbose: move_spinner(j)

    return yA, yB, yC

def getvarsens(yA, yB, yC):
    """Calculate the array of S_i and ST_i for each parameter given yA, yB, yC matrices
    from the multi-sampling runs. Calculate S_i and ST_i as follows:

    Parameter sensitivity:
    ----------------------
            U_j - E^2 
    S_j = ------------
               V(y)

    U_j = 1/n \sum yA * yC_j

    E^2 = 1/n \sum yA * 1/n \sum yB

    Total effect sensitivity (i.e. non additive part):
    --------------------------------------------------
                  U_-j - E^2
     ST_j = 1 - -------------
                      V(y)

    U_-j = 1/n \sum yB * yC_j

    E^2 = { 1/n \sum yB * yB }^2


    In both cases, calculate V(y) from yA and yB


    """
    nparms = yC.shape[0] # should be the number of parameters
    nsamples = yC.shape[1] # should be the number of samples from the original matrix
    nobs = yC.shape[-1]    # the number of observables (this is linked to BNG usage, generalize?)

    #first get V(y) from yA and yB

    varyA = numpy.var(yA, axis=0, ddof=1)
    varyB = numpy.var(yB, axis=0, ddof=1)

    # now get the E^2 values for the S and ST calculations
    E_s  = numpy.average((yA * yB), axis=0)
    E_st = numpy.average(yB, axis=0) ** 2

    #allocate the S_i and ST_i arrays
    Sens = numpy.zeros((nparms,nobs))
    SensT = numpy.zeros((nparms,nobs))

    # now get the U_j and U_-j values and store them 
    for i in range(nparms):
        Sens[i]  =        (((yA * yC[i]).sum(axis=0)/(nsamples-1.)) - E_s ) / varyA
        SensT[i] = 1.0 - ((((yB * yC[i]).sum(axis=0)/(nsamples-1.)) - E_st) / varyB)

    return Sens, SensT
