import numpy as np


def select_topr(vct_input, r):
    """
    Returns the R-th greatest elements indices
    in input vector and store them in idxs_n.
    Here, we're using this function instead of
    a complete sorting one, where it's more efficient
    than complete sorting function in real big data application
    parameters
    ----------
    vct_input : array, shape (T)
        indicating the input vector which is a
        vector we aimed to find the Rth greatest
        elements. After finding those elements we
        will store the indices of those specific
        elements in output vector.
    r : integer
        indicates Rth greatest elemnts which we
        are seeking for.
    Returns
    -------
    idxs_n : array, shape (R)
        a vector in which the Rth greatest elements
        indices will be stored and returned as major
        output of the function.
    """
    temp = np.argpartition(-vct_input, r)
    idxs_n = temp[:r]
    return idxs_n
