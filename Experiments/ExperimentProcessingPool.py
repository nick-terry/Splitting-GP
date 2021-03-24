# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 08:59:36 2020

@author: pnter
"""
from multiprocessing.pool import Pool
import multiprocessing as mp
import traceback
import itertools

def starmapstar(args):
    return list(itertools.starmap(args[0], args[1]))

def error(msg, *args):
    return mp.get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

class ExperimentProcessingPool(Pool):
    def starmap(self,func, iterable, chunksize=None):
        '''
        Like `map()` method but the elements of the `iterable` are expected to
        be iterables as well and will be unpacked as arguments. Hence
        `func` and (a, b) becomes func(a, b).
        '''
        return self._map_async(LogExceptions(func), iterable, starmapstar, chunksize).get()