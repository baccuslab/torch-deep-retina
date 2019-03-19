import numpy as np

#def parallel_shuffle(*args, perm=None, ret_perm=False):
#    shuffles = [np.empty(args[0].shape, dtype=args[0].dtype)]
#    for i in range(len(args[1:])):
#        assert len(args[i-1]) == len(args[i])
#        shuffles.append(np.empty(args[i].shape, dtype=args[i].dtype))
#    if perm is None:
#        perm = np.random.permutation(len(args[0]))
#    for old_idx, new_idx in enumerate(perm):
#        for arr_idx in range(len(shuffles)):
#            shuffles[arr_idx][new_idx] = args[arr_idx][old_idx]
#    if ret_perm:
#        return shuffles, perm
#    return shuffle arrays in-place, in the same order, along axis=0
def parallel_shuffle(arrays, set_seed=-1):
    """
    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed
    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)
