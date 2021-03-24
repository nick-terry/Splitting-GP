import time

'''
Timer class. Emulates behaviour of matlab's tic/toc
'''

start = None

def tic():
    global start
    start = time.time()

def toc(verbose=True):
    global start
    if start is None:
        print "Timer is not running!"
        return 0
    else:
        elapsed = time.time() - start
        if verbose:
            print "Time elapsed: %lfs" % elapsed
        start = None
        return elapsed

class tictoc(object):

    def __init__(self,verbose=None):
        self.verbose = verbose
        self.start = None
        pass

    def tic(self):
        self.start = time.time()

    def toc(self,verbose=None):
        verbose = verbose if self.verbose is None else self.verbose
        verbose = True if verbose is None else False
        if self.start is None:
            raise Exception("Timer is not running!")
        elapsed = time.time() - self.start
        if self.verbose:
            print "Time elapsed: %lfs" % elapsed
        self.start = None
        return elapsed
