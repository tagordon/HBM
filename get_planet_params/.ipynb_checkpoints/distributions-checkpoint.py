import numpy as np

class uniform_prior:
    '''
    A uniform prior. 

    Args:
        low (float): lower bound
        high (float): upper bound
        init (float, optional): where to 
        initialize the sampler, otherwise 
        set to the mid-point of the prior.
    '''

    def __init__(self, low, high, init=None):

        self.low = low
        self.high = high
        self.width = high - low
        self.vec_prior = np.vectorize(self.prior)

        if init is None:
            self.init = 0.5 * (high + low)
        else:
            self.init = init

    def prior(self, x):
        '''
        Evaluate the distribution at a point x, and 
        return its log-probability.
        '''

        if (x >= self.low) & (x <= self.high):
            return 0.0
        else:
            return -np.inf

class normal_prior:

    '''
    A normal prior

    Args: 
        mu (float): the mean of the normal distribution 
        sig (float): the standard deviation of the 
        normal distribution 
        init (float, optional): where to initialize the 
        sampler, otherwise set to mu
    '''

    def __init__(self, mu, sig, init=None):

        self.mu = mu
        self.sig = sig
        self.width = sig
        self.vec_prior = np.vectorize(self.prior)

        if init is None:
            self.init = mu
        else:
            self.init = init

    def prior(self, x):
        '''
        Evaluate the distribution at a point x 
        and return its log-probability.
        '''
        return -0.5 * (x - self.mu)**2 / self.sig**2
    
class assymetric_normal_prior:
    
    '''
    A normal prior that's actually two halves of a 
    normal with different standard deviations stuck 
    together. 
    '''
    
    def __init__(self, mu, sigp, sigm, init=None):
        self.mu = mu
        self.sigp = sigp
        self.sigm = sigm
        self.vec_prior = np.vectorize(self.prior)
        
        if init is None:
            self.init = mu
        else:
            self.init = init
            
    def prior(self, x):
        '''
        Evaluate the distribution at a point x 
        and return its log-probability.
        '''
        
        if x < self.mu:
            return -0.5 * (x - self.mu)**2 / self.sigm**2
        if x > self.mu:
            return -0.5 * (x - self.mu)**2 / self.sigp**2
        
class trunc_assymetric_normal_prior:
    
    '''
    A normal prior that's actually two halves of a 
    normal with different standard deviations stuck 
    together. 
    '''
    
    def __init__(self, mu, sigp, sigm, low, high, init=None):
        self.mu = mu
        self.sigp = sigp
        self.sigm = sigm
        self.low = low
        self.high = high
        self.vec_prior = np.vectorize(self.prior)
        
        if init is None:
            self.init = mu
        else:
            self.init = init
            
    def prior(self, x):
        '''
        Evaluate the distribution at a point x 
        and return its log-probability.
        '''
        
        if (x >= self.low) & (x <= self.high):
            if x < self.mu:
                return -0.5 * (x - self.mu)**2 / self.sigm**2
            if x > self.mu:
                return -0.5 * (x - self.mu)**2 / self.sigp**2
        else:
            return -np.inf

class trunc_normal_prior:
    '''
    A truncated normal distribution

    Args: 
        mu (float): the mean of the normal distribution 
        sig (float): the standard deviation of the normal distribution
        low (float): lower bound
        high (float): upper bound
        init (float, optional): where to initialize the sampler, 
        otherwise set to mu
    '''

    def __init__(self, mu, sig, low, high, init=None):

        self.mu = mu
        self.sig = sig
        self.low = low
        self.high = high
        self.width = np.min([high - low, sig])
        self.vec_prior = np.vectorize(self.prior)

        if init is None:
            self.init = mu
        else:
            self.init = init
            
    def prior(self, x):
        '''
        Evaluate the distribution at a point x, and 
        return its log-probability.
        '''

        if (x >= self.low) & (x <= self.high):
            return -0.5 * (x - self.mu)**2 / self.sig**2
        else:
            return -np.inf