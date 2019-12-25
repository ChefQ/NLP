from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

#dataDir = '/u/cs401/A3/data/'
dataDir = "/Users/oluwaseuncardoso/CSC401/A3/data" #NOTE remove this
class theta:
    def __init__(self, name, M=8,d=13):
        # Note should i initialize here
        self.name = name # each component is a speaker
        self.omega = np.zeros((M,1)) # prior probability of the m^{th} Gaussian
        self.mu = np.zeros((M,d)) # each row is the a vector of means for each model/component
        self.Sigma = np.zeros((M,d)) # a vector representing Sigma is okay because of the independence assumption


def log_b_m_x( m, x, myTheta, preComputedForM=[]): #NOTE write a helper function that vectorizes this
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout
         As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

        # what you mean by "you pass that precomputed component in preComputedForM"

    '''
    # m is like the index for theta
    # t is exponent of e. e.g. e^{t}
    d = len(x)
    """ #NOTE remove this
    t = 0.5 * np.sum ( np.true_divide( np.power(x - myTheta.mu,2), myTheta.Sigma ) )
    numerator = np.exp(t)
    denominator = ( (2*np.pi)**d/2 ) * np.sqrt( np.prod(myTheta.Sigma) )
    b_m_x = np.true_divide(numerator,denominator)
    
    return np.log(b_m_x)
    """
    # for numerical stability

    term_1 = -0.5 * np.sum ( np.true_divide( np.power(x - myTheta.mu[m],2), myTheta.Sigma[m] ) )
    term_2 = (d*0.5) * np.log(2*np.pi) # i think this should be d^2
    term_3 = 0.5*np.sum( np.log(myTheta.Sigma[m])) # fix this

    return term_1 - term_2 - term_3

def b_m_x(m, x, myTheta):
    """
    calculates the probability of x given the m^{th} mixture componet
    """
    d = len(x)
    t = 0.5 * np.sum ( np.true_divide( np.power(x - myTheta.mu[m],2), myTheta.Sigma[m] ) )
    numerator = np.exp(t)
    denominator = ( (2*np.pi)**d/2 ) * np.sqrt( np.prod(myTheta.Sigma[m]) )
    b_m_x = np.true_divide(numerator, denominator)

    return b_m_x

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M = myTheta.omega.shape[0]
    log_b = log_b_m_x(m, x, myTheta)
    numerator = logsumexp(log_b,b=myTheta.omega)
    Ms = np.reshape(np.arange(M),(1,M))

    log_bs = np.apply_along_axis(log_b_m_x, 0, Ms, x=x, myTheta=myTheta)
    denomenator = logsumexp(log_bs, b=myTheta.omega)

    return np.subtract(numerator, denomenator)

def LOG_B_m_x(i,X,myTheta):
    """
    if this function is used with np.apply_along_axis(LOG_B_m_x,0,X.T,myTheta = myTheta ) ...
    This functions returns an MxT log_b_m_t

    :return: a vector containing log_b_m_x for each m

    """

    d = X.shape[1]
    term_1 = -0.5 * np.sum(np.true_divide(np.power(X - myTheta.mu[i], 2), myTheta.Sigma[i]), axis=1)
    term_2 = (d * 0.5) * np.log(2 * np.pi)
    term_3 = 0.5 * np.sum(np.log(myTheta.Sigma[i]))  # fix this

    return term_1 - term_2 - term_3

def LOG_B_M_X(m, X, myTheta):
    """
    :return: MxT log_b_m_t
    """
    t = X.shape[0]
    lOG_BMX = np.zeros((m, t))
    for i in range(m):

        lOG_BMX[i, :] = LOG_B_m_x(i,X,myTheta)

    return lOG_BMX

def LOG_P_M_X( m, X, myTheta,LOGBMX):
    """
    :return: MxT log_p_m_t
    """
    t = X.shape[0]
    LOG_PMX = np.zeros((m, t))

    for i in range(m):
        log_b = LOG_B_m_x(i, X, myTheta).reshape((t,1)) # vector size t
        numerator = logsumexp(log_b, b=myTheta.omega[i], axis = 1)
        denomenator = logsumexp(LOGBMX,axis = 0, b = myTheta.omega)
        LOG_PMX[i , :] = numerator - denomenator
    return LOG_PMX



def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    denomenator = logsumexp(log_Bs, axis=0, b=myTheta.omega)
    b = np.sum(denomenator)
    return b

def UpdateParameters(X,myTheta,LOG_B):
    '''
    probability of the m^{th} component given d-dimensional vector x
    :param X:
    :param myTheta:
    :return:
    '''
    T = X.shape[0]
    m = myTheta.omega.shape[0]

    LOG_PMX = LOG_P_M_X( m, X, myTheta,LOG_B)
    PMX = np.exp(LOG_PMX)
    sum_P_M_X = np.sum( PMX, axis = 1 )
    sum_P_M_X = sum_P_M_X.reshape((1,sum_P_M_X.shape[0]))
    myTheta.omega = np.true_divide( sum_P_M_X , T ).reshape((sum_P_M_X.shape[1],1)) # the shape of this changed
    X2 = np.power(X,2)
    myTheta.mu = np.true_divide(np.matmul(PMX, X), sum_P_M_X.T)
    myTheta.Sigma = np.true_divide( np.matmul(PMX, X2) , sum_P_M_X.T   ) - np.power( myTheta.mu, 2)

def train( speaker, X, M=2 , epsilon=0.0, maxIter=20 ): #NOTE test train
    #Note should i initialize here or there look up
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    # initialize theta here
    d =  X.shape[1]
    num_rows = X.shape[0]
    indices = np.random.randint(num_rows,size = 8)
    myTheta = theta( speaker, M, d )
    myTheta.omega =  np.ones((M,1))/M
    myTheta.Sigma = np.ones((M,d))
    myTheta.mu = X[indices,:]

    i = 0
    prev_L = - np.inf
    improvement = np.inf
    while i <= maxIter and improvement >= epsilon:
        LOG_B = LOG_B_M_X(M, X, myTheta)
        L = logLik(LOG_B, myTheta)

        UpdateParameters(X,myTheta,LOG_B)
        improvement = L - prev_L
        prev_L = L
        i = i + 1
    return myTheta


def test( mfcc, correctID, models, k=5 ): #NOTE ask if we should also print. I doubt I should do this
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    m = models[0].omega.shape[0] # number of components
    likelihoods = []
    names = []

    for model in models:
        # compute intermediate results
        #LOG_B = np.apply_along_axis(LOG_B_m_x, 1, mfcc, myTheta=model)
        LOG_B  = LOG_B_M_X(m, mfcc ,model)

        # ComputeLikelihood
        L = logLik(LOG_B, model)
        likelihoods.append( L )
        names.append(model.name)
    # pick the best mode
    s = len(likelihoods)
    bestModelValue = max(likelihoods)

    bestModel = likelihoods.index(bestModelValue)
    bestK = sorted(likelihoods)[s-k:s]

    correctSpeaker = models[correctID].name

    print(correctSpeaker)
    # find the k best
    for i in range(len(bestK)-1,-1,-1):
        likelihood = bestK[i]
        nameIndex = likelihoods.index(likelihood)
        name = names[nameIndex]
        print( name+" "+str(likelihood) )

    print()
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 2
    epsilon = 0.0
    maxIter = 20
    names = []
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )

            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )
            names.append(speaker)
            X = np.empty((0,d)) #
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0) # speech utterance # this can be a matrix

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )



    # evaluate
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k )
    accuracy = 1.0*numCorrect/len(testMFCCs)

