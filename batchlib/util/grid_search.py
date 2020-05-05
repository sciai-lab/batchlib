import numpy as np
import multiprocessing as mp
import itertools
from tqdm.auto import tqdm


# pretty unnecessary, this is basically mp.Pool.apply
def execute_parallel(func, args_list=None, kwargs_list=None, n_jobs=0):
    assert args_list is not None or kwargs_list is not None
    if n_jobs > 0:
        apply = mp.Pool(processes=n_jobs).apply_async
    else:
        assert n_jobs == 0, f'{n_jobs}'
        def apply(func, args, kwds):
            return func(*args, **kwds)
    if args_list is not None and kwargs_list is not None:
        results = [apply(func, args=args, kwds=kwargs)
                   for args, kwargs in zip(tqdm(args_list), kwargs_list)]
    elif args_list is not None:
        results = [apply(func, args=args)
                   for args in tqdm(args_list)]
    elif kwargs_list is not None:
        results = [apply(func, kwds=kwargs)
                   for kwargs in tqdm(kwargs_list)]
    else:
        assert False, 'either args_list or kwargs_list must not be None'
    if n_jobs > 0:
        results = [p.get() for p in results]
    return results


def grid_evaluate(func, *args, return_structured_array=True, **kwargs):
    """
    Evaluates the function func on the cartesian product of the lists given for all the args and kwargs.

    :param func: callable
        Function to be evaluated.
    :param args:
        Lists of positional argument values to be iterated over.
    :param return_structured_array: bool
        If true, result is a structured array also including all the arguments
    :param kwargs:
        Lists of keyword argument values to be iterated over.


    :return:
        dict
    """
    n_jobs = kwargs.pop('n_jobs', 0)
    shape = tuple([len(arg) for arg in args + tuple(kwargs.values())])

    args_and_kwargs = list(itertools.product(*(args + tuple(kwargs.values()))))
    n_args = len(args)
    args_list = [a[:n_args] for a in args_and_kwargs]
    kwargs_list = [dict(zip(kwargs.keys(), a[n_args:])) for a in args_and_kwargs]

    result = np.empty(len(args_and_kwargs), dtype=object)
    result[...] = execute_parallel(func, args_list, kwargs_list, n_jobs)
    result = result.reshape(shape)
    if return_structured_array:  # wrap args in numpy array with same shape as result
        dtype = np.dtype([('result', object)]
                         + [(f'arg{i}', object) for i in range(len(args))]
                         + [(kw, object) for kw in kwargs])
        data = [(result,) + args_kwargs for result, args_kwargs in zip(result.flatten(), args_and_kwargs)]
        result = np.array(data, dtype).reshape(shape)
    return result


def optimal_parameters(result_grid):
    ind = np.unravel_index(np.argmin(result_grid['result']), result_grid.shape)

    return {name: result_grid[name][ind] for name in result_grid.dtype.names}
