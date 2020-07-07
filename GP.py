import pdb

import numpy as np
from scipy import linalg, optimize
from math import inf

import cov

class GP(object):

    def __init__(self,X,y,cov_type='covSEard'):

        self.X = X # training inputs, 2d array
        self.y = y # training outputs, 1d array
        self.cov_type = cov_type # string specifying covariance function

        # initialise all log params to 0
        if self.cov_type == 'covSEiso':
            self.params = np.asarray([0.0]*3)
            self.params[-1] = np.log(0.01)
        elif self.cov_type == 'covSEard':
            self.params = np.asarray([0.0]*(self.input_dim+2))
            self.params[0:self.input_dim] = np.log(np.std(X,axis=0))
            self.params[-2] = np.log(np.var(y,axis=0))
            self.params[-1] = self.params[-2]-np.log(10)
        
        # cache for the Cholesky factor
        self.L_cache = np.empty((self.size,self.size),dtype=X.dtype)

        # cache for the K^{-1}y vector
        self.Kinvy_cache = np.empty((self.size),dtype=X.dtype)

        # cache for the Q = invK - outer(Kinvy,Kinvy) matrix
        self.Q_cache = np.empty((self.size,self.size),dtype=X.dtype)

        self.L_cached_at = None # parameters at which cached L, Kinvy values hold
        self.Q_cached_at = None # parameters at which cached Q values hold

    def NLML(self,params=None,derivs=False):

        # negative log marginal likelihood and derivatives
         
        params = self.params if params is None else params

        cov_params = params[:-1]
        sig_noise = np.exp(params[-1])

        L = self.L_cache
        Kinvy = self.Kinvy_cache

        if self.L_cached_at is None or not np.all(self.L_cached_at == params):
            # if parameters differ from cached ones, compute

            # generate covariance k(X,X)
            K  = cov.cov(self.cov_type,self.X,self.X,cov_params)
            K += np.eye(self.size)*sig_noise # add noise to the diagonal

            # Cholesky factor of k(X,X)
            #If this fails, add some diagonal jitter and re-attempt
            attempts = 0
            maxAttempts = 5
            while attempts < maxAttempts:
                try:
                    L[:] = linalg.cholesky(K,lower=True)
                    attempts = maxAttempts
                
                except np.linalg.LinAlgError as e:
                    print('Attempted Cholesky of non-PD matrix...attempt {0}'.format(attempts))
                    K += np.eye(self.size)*10**-5
                    attempts += 1
                    
                    if attempts==maxAttempts:
                        print(cov_params)
                        print(K.shape)
                        print(np.any(K<0))
                        print('Eigenvalues:')
                        print(np.linalg.eigvals(K))
                        raise e
                    
            # k(X,X)^-1*y
            Kinvy[:] = linalg.cho_solve((L,True),self.y)

            self.L_cached_at = np.array(params,copy=True)

        # log determinant of Cholesky factor of k(X,X)
        # = 0.5*log determinant of k(X,X)
        logdetL = np.sum(np.log(np.diag(L)))
        
        if not derivs:
            NLML =  0.5*np.dot(self.y,Kinvy) # 0.5 * y^T * K^-1 * y
            NLML += logdetL # 0.5 * log det(K)
            NLML += 0.5*float(self.size)*np.log(2.0*np.pi) # 0.5*N*log(2*pi)
            NLML = NLML/float(self.size)
            return NLML

        Q = self.Q_cache
        
        if self.Q_cached_at is None or not np.all(self.Q_cached_at == params):
            # compute derivatives
            invK = linalg.cho_solve((L,True),np.eye(self.size))
            Q[:] = invK - np.outer(Kinvy,Kinvy)
            self.Q_cached_at = np.array(params,copy=True)

        dparams = np.zeros_like(params)

        for i in range(len(cov_params)):
            # covariance derivatives
            dK = cov.cov(self.cov_type,self.X,self.X,cov_params,derivs=i)
            dparams[i] = 0.5*np.sum(dK*Q) # equivalent to trace(dK*Q)

        # noise derivative
        dparams[-1] = 0.5*np.sum(np.diag(Q))*sig_noise
        dparams = dparams/float(self.size)
        return dparams

    def train(self):
        
        # train the GP by minimising the negative log marginal likelihood
        f = lambda x: self.NLML(x)
        fp = lambda x: self.NLML(x,derivs=True)
        
        #Create lower upper bound on lengthscale to prevent overflow issues
        #Lower bound on 1/l is 10^-5, so we constraint the log(l) < 10*log(10)
        boundPow = 10
        hyperparamBounds = [(-inf,boundPow*np.log(10))]*self.params[:-1].shape[0]
        hyperparamBounds.append((-inf,inf))
        
        self.params = optimize.minimize(f,self.params,
                                        bounds=hyperparamBounds,
                                        method='L-BFGS-B',jac=fp).x
        #self.params = optimize.fmin_l_bfgs_b(f,self.params,fp,disp=True)[0]
        #self.params = optimize.fmin_bfgs(f,self.params,fp,disp=1)[0]

    def predict(self,Xp,variance=False,latent_variance=False,entropy=False):

        # GP prediction

        cov_params = self.params[:-1]
        sig_noise = np.exp(self.params[-1])
        
        # generate k(Xp,X)
        Kp  = cov.cov(self.cov_type,Xp,self.X,cov_params)

        Kinvy = self.Kinvy_cache
        L = self.L_cache

        if self.L_cached_at is None or not np.all(self.L_cached_at == self.params):

            # generate covariance k(X,X)
            K  = cov.cov(self.cov_type,self.X,self.X,cov_params)
            K += np.eye(self.size)*sig_noise # add noise to the diagonal

            # Cholesky factor of k(X,X)
            L[:] = linalg.cholesky(K,lower=True)

            # k(X,X)^-1*y
            Kinvy[:] = linalg.cho_solve((L,True),self.y)

            self.L_cached_at = np.array(self.params,copy=True)

        # mean of predictive distribution
        ymu = np.dot(Kp,Kinvy)

        output = (ymu,)

        if variance or latent_variance:

            # diagonal of k(Xp,Xp): prior variance
            # Kd  = cov.cov(self.cov_type,Xp,Xp,cov_params,diag=True)
            Kd = [np.exp(self.params[-2])] # prior variance (stationary kernels). 

            # diagonal of k(Xp,X) * k(X,X) * k(X,Xp)
            KinvKp = linalg.cho_solve((L,True),Kp.T)
            ent = (Kp*KinvKp.T).sum(axis=1)

            
            fs2 = Kd - ent

            if variance: output += (fs2+sig_noise,)
            if latent_variance: output += (fs2,)
            if entropy:
                ent += 1e-6
                if hasattr(self, 'beta'):
                    beta=self.beta*(ent/ent)
                else:
                    # compute the beta weighting factors for each GP expert
                    # diff in differential entropy between
                    # prior and posterior (latent) variance
                    beta = 0.5*np.log(Kd)-0.5*np.log(fs2) + 1e-6
                    
                output += (beta,)

        return output[0] if len(output) == 1 else output

    @property
    def size(self):
        # number of training samples
        return self.X.shape[0]

    @property
    def input_dim(self):
        # number of training samples
        return self.X.shape[1]

# workarounds for the DGP and multiprocessing library

def GP_NLML_wrapper(args):
    return GP.NLML(*args)

def GP_predict_wrapper(args):
    return GP.predict(*args)
