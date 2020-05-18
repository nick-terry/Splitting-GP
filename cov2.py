import pdb

import numpy as np
from scipy import spatial


def cov(cov_type,*args,**kwargs):

    cov_func = {
                'covSEiso':covSEiso,
                'covSEard':covSEard, 
               }

    return cov_func[cov_type](*args,**kwargs)
    
def covSEiso(X1,X2,params,derivs=None,diag=False):

    # squared exponential kernel (isotropic)
    
    d = spatial.distance.cdist(X1,X2,'sqeuclidean')

    l = np.exp(params[0])  # lengthscale
    sig_f = np.exp(params[1]) # signal variance

    if diag:
        K = np.empty_like(X1[:,0])
        K.fill(sig_f)
        return K

    if derivs is None:
        K = sig_f*np.exp(-0.5*d/(l**2))
    elif derivs == 0:
        # derivatives w.r.t lengthscale
        K = sig_f*np.exp(-0.5*d/(l**2))*(d/l)
    elif derivs == 1:
        # derivatives w.r.t signal variance
        K = sig_f*np.exp(-0.5*d/(l**2))

    return K

def covSEard(X1,X2,params,derivs=None,diag=False):

    # squared exponential kernel (ARD)

    l = np.exp(params[:-1])  # lengthscale
    sig_f = np.exp(params[-1]) # signal variance
    
    d = np.zeros((X1.shape[0],X2.shape[0]),dtype=X1.dtype)
    iL = np.diag(1/l)
    iL.reshape(len(l), len(l))
    Xbar = np.dot(X1,iL)
    Zbar = np.dot(X2,iL)
    for i in xrange(X1.shape[1]):
        d_i = (Xbar[:,i].reshape(-1,1) - Zbar[:,i].reshape(1,-1))**2
        if derivs is not None and derivs < X1.shape[1]:
            if i == derivs: d_derivs_i = d_i
        d -= d_i/(2.0)
#    for i in xrange(X1.shape[1]):
#        d_i = (X1[:,i].reshape(-1,1) - X2[:,i].reshape(1,-1))**2
#        if derivs is not None and derivs < X1.shape[1]:
#            if i == derivs: d_derivs_i = d_i
#        d -= d_i/(2.0*(l**2))


    if diag:
        K = np.empty_like(X1[:,0])
        K.fill(sig_f)
        return K

    if derivs is None:
        K = sig_f*np.exp(d)
    elif derivs < X1.shape[1]:
        # derivatives w.r.t lengthscale
        K = sig_f*np.exp(d)*(d_derivs_i/l[derivs])
    elif derivs ==  X1.shape[1]:
        # derivatives w.r.t signal variance
        K = sig_f*np.exp(d)

    return K
