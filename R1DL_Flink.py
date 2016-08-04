#!/usr/bin/python2
# Flictionary Learning: Rank-1 Dictionary Learning in Flink!
# Probably PEP 8 compliant.

import argparse

import itertools

import numpy as np
import scipy.linalg as sla
import datetime
import os

import sys
from flink.functions.Aggregation import Sum
from flink.functions.FlatMapFunction import FlatMapFunction
from flink.functions.GroupReduceFunction import GroupReduceFunction
from flink.functions.MapFunction import MapFunction
from flink.plan.Constants import Order, WriteMode, INT, FLOAT
from flink.plan.Environment import get_environment


###################################
# Utility functions
###################################

def input_to_row_matrix(raw):
    """
    Utility function for reading the matrix data
    """
    # Parse each line of the input into a numpy array of floats. This requires
    # several steps.
    #  1: Split each string into a list of strings.
    #  2: Convert each string to a float.
    #  3: Convert each list to a numpy array.
    data = raw \
        .zip_with_index() \
        .map(lambda x: (x[0], parse_and_normalize(x[1])))
    return data


###################################
# Flink helper functions
###################################

def parse_and_normalize(line):
    """
    Utility function. Parses a line of text into a floating point array, then
    whitens the array.
    """
    x = tuple(map(float, line.strip().split()))

    # x.strip() -- strips off trailing whitespace from the string
    # .split("\t") -- splits the string into a list of strings, splitting on tabs
    # map(float, list) -- converts each element of the list from strings to floats
    # np.array(list) -- converts the list of floats into a numpy array

    # comment by Xiang: the following normalization commands work for vector u,
    # but not work here. I have double-checked it with pre-normalized matrix;
    # x -= x.mean()  # 0-mean.
    # x /= sla.norm(x)  # Unit norm.
    return x


# def select_topr(vct_input, r):
#     """
#     Returns the R-th greatest elements indices
#     in input vector and store them in idxs_n.
#     Here, we're using this function instead of
#     a complete sorting one, where it's more efficient
#     than complete sorting function in real big data application
#     parameters
#     ----------
#     vct_input : array, shape (T)
#         indicating the input vector which is a
#         vector we aimed to find the Rth greatest
#         elements. After finding those elements we
#         will store the indices of those specific
#         elements in output vector.
#     r : integer
#         indicates Rth greatest elemnts which we
#         are seeking for.
#     Returns
#     -------
#     idxs_n : array, shape (R)
#         a vector in which the Rth greatest elements
#         indices will be stored and returned as major
#         output of the function.
#     """
#     temp = np.argpartition(-vct_input, r)
#     idxs_n = temp[:r]
#     return idxs_n


def vector_matrix(u__, S_):
    """
    Generates v using u.
    """
    v = u__ \
        .join(S_).where(0).equal_to(0) \
        .using(lambda u_el, v_el: (v_el[1], v_el[2] * u_el[1])) \
        .name('VectorMatrix')

    v = v \
        .group_by(0) \
        .aggregate(Sum, 1)

    return v


class VectorMatrixGroupReducer(GroupReduceFunction):
    def reduce(self, iterator, collector):
        S_original = self.context.get_broadcast_variable("S_orig")
        iterator = np.array(sorted(iterator))
        u__ = np.take(iterator, 1, axis=1)

        for s in S_original:
            k, vector = s
            vector = np.array(vector)
            for i in range(vector.shape[0]):
                collector.collect((i, u__[k] * vector[i]))


class MatrixVectorGroupReducer(GroupReduceFunction):
    """
    Applies S * v by row-wise multiplication. No reduction needed, as all the
    summations are performed within this very function.
    """
    def reduce(self, iterator, collector):
        S_original = self.context.get_broadcast_variable("S_orig")
        iterator = np.array(sorted(iterator))
        i__ = np.take(iterator, 0, axis=1).astype(int)
        v__ = np.take(iterator, 1, axis=1)

        for s in S_original:
            k, vector = s
            vector = np.array(vector)
            innerprod = np.dot(vector[i__], v__)
            collector.collect((k, innerprod))


class RandomVectorFlatMapper(FlatMapFunction):
    """
    Takes a DataSet and replaces a single placeholder element with the elements
    of a numpy.random.random vector.
    """
    def __init__(self, t_):
        self.T = t_
        super(RandomVectorFlatMapper, self).__init__()

    def flat_map(self, value, collector):
        vec = np.random.random(self.T)
        vec -= vec.mean()
        vec /= sla.norm(vec)
        return list(enumerate(vec))


class NormalizeVectorGroupReducer(GroupReduceFunction):
    """
    Normalizes a vector in (index, value) format.
    """
    def reduce(self, iterator, collector):
        vector = np.take(sorted(iterator), 1, axis=1)
        vector -= vector.mean()
        vector /= sla.norm(vector)

        return list(enumerate(vector))


class MagnitudeGroupReducer(GroupReduceFunction):
    """
    Calculates the magnitude of a vector.
    """
    def reduce(self, iterator, collector):
        vector = np.take(sorted(iterator), 1, axis=1)
        mag = sla.norm(vector)
        with open('test-mag', mode='a') as f:
            f.write(str(mag) + '\n')
        collector.collect((0, mag))


class DeltaGroupReducer(GroupReduceFunction):
    """
    Find the delta of two unioned data sets.
    """
    def reduce(self, iterator, collector):
        elements = list(iterator)
        with open('testing2', mode='a') as f:
            f.write(str(elements) + '\n')
        a = elements[0]
        b = elements[1]
        collector.collect((a[0], a[1] - b[1]))


class SExploderGroupReducer(GroupReduceFunction):
    def reduce(self, iterator, collector):
        for x in iterator:
            for y in enumerate(x[1]):
                collector.collect((x[0], y[0], y[1]))


if __name__ == "__main__":

    print datetime.datetime.now()

    parser = argparse.ArgumentParser(description='Flictionary Learning', add_help='How to use',
                                     prog='.../pyflink2.sh R1DL_Flink.py - <args>')

    # Inputs.
    parser.add_argument("-i", "--input", required=True,
                        help="Input file containing the matrix S.")
    parser.add_argument("-t", "--rows", type=int, required=True,
                        help="Number of rows (observations) in the input matrix S.")
    parser.add_argument("-p", "--cols", type=int, required=True,
                        help="Number of columns (features) in the input matrix S.")
    parser.add_argument("-r", "--pnonzero", type=float, required=True,
                        help="Percentage of non-zero elements.")
    parser.add_argument("-m", "--mDicatom", type=int, required=True,
                        help="Number of the dictionary atoms.")
    parser.add_argument("-e", "--epsilon", type=float, required=True,
                        help="The value of epsilon.")

    # Outputs.
    parser.add_argument("-d", "--dictionary", required=True,
                        help="Output path to dictionary file.(file_D)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output path to z matrix.(file_z)")
    parser.add_argument("-prefix", "--prefix", required=True,
                        help="Prefix strings to the output files")

    args = vars(parser.parse_args())

    # Initialize the Flink environment.
    env = get_environment()

    # Read the data and convert it into a thunder RowMatrix.
    raw_data = env.read_text(args['input'])

    # Each of the following tuples: (row pos, vec pos, value)
    S_orig = input_to_row_matrix(raw_data)
    S = S_orig \
        .reduce_group(SExploderGroupReducer()) \
        .name('SExploderGroupReducer')


    ##################################################################
    # Here's where the real fun begins.
    #

    # First, we're going to initialize some variables we'll need for the
    # following operations. Next, we'll start the optimization loops. Finally,
    # we'll perform the stepping and deflation operations until convergence.
    #
    # Sound like fun?
    ##################################################################

    T = args['rows']
    P = args['cols']

    epsilon = args['epsilon']  # convergence stopping criterion
    M = args['mDicatom']  # dimensionality of the learned dictionary
    R = int(round(args['pnonzero'] * P))  # enforces sparsity

    ##################################################################
    # Apparently Flink doesn't like np.arrays. So for vectors, we use Flink
    # DataSets with indexes corresponding to their position in the vector.
    ##################################################################

    u_new = env.from_elements(*[0 for t in range(T)])  # atom updates at each iteration
    v = env.from_elements(*[0 for p in range(P)])

    v_top_R = env.from_elements(*[0 for r in range(R)])  # for top-R sorting

    max_iterations = int(P * 10)
    file_D = os.path.join(args['dictionary'], "{}_D.txt".format(args["prefix"]))
    file_z = os.path.join(args['output'], "{}_z.txt".format(args["prefix"]))

    # Start the loop!
    for m in range(M):
        # Generate a random vector, subtract off its mean, and normalize it.
        u_old = env.from_elements(0).flat_map(RandomVectorFlatMapper(T))
        u_old_it = u_old.iterate(max_iterations)

        # Start the inner loop: this learns a single atom.
        # P2: Vector-matrix multiplication step. Computes v.

        # We can't broadcast partial iteration results.
        # This replaces vector_matrix in the original Spark implementation.
        # flat map is replaced by group reduce functions because it doesn't seem to work with join etc.
        S_orig = S_orig.map(lambda x: x)  # S_orig is null (?) without this
        v = vector_matrix(u_old_it, S)  #.reduce_group(VectorMatrixGroupReducer()).with_broadcast_set('S_orig', S_orig)

        # sort v by using sort_group after grouping on a dummy field
        v = v.map(lambda x: (x[0], x[1], 0))
        v = v \
            .group_by(2).sort_group(1, Order.DESCENDING) \
            .reduce_group(lambda i, c: [(i_[0], i_[1]) for i_ in i]) \
            .name('VSorter')

        v_top_R = v.first(R)

        # P1: Matrix-vector multiplication step. Computes u.
        # u_new = v_top_R.reduce_group(MatrixVectorGroupReducer()).with_broadcast_set('S_orig', S_orig)

        # We can use a join to do a dot product.
        # (S * v by row-wise multiplication)
        # This replaces matrix_vector in the original Spark implementation.
        # Multiply each corresponding element
        u_new = v_top_R \
            .join(S).where(0).equal_to(1) \
            .using(lambda v_el, s_el: (s_el[0], s_el[1], s_el[2] * v_el[1])) \
            .name('MatrixVector')

        # Now, add up all the products to get the dot product result, and remove
        # the vector position field (second field from the left)
        u_new = u_new.group_by(0) \
            .aggregate(Sum, 2) \
            .map(lambda x: (x[0], x[2]))
        ################################################################

        # Subtract off the mean and normalize.
        u_new = u_new.reduce_group(NormalizeVectorGroupReducer()).name('NormalizeVector')

        # Update for the next iteration
        # Join function does weird things here (ClassCastException?) so a union function is used as a workaround
        # Find the difference between the two elements with the same index, since only magnitude (and not sign) matter
        # todo check if this actually affects the number of iterations
        # these operations don't exist (aren't processed???) D:
        # delta = u_old_it.group_by(0) \        # delta = delta.reduce_group(MagnitudeGroupReducer()) \
        #     .name('MagnitudeGroupReducer')
        #     .reduce_group(DeltaGroupReducer()).name('DeltaGroupReducer')
        # delta = delta.reduce_group(MagnitudeGroupReducer()) \
        #     .name('MagnitudeGroupReducer')
        # u_new = u_new.map(lambda x: x)

        # delta = u_new.join(u_old_it).where(0).equal_to(0) \
        #     .using(lambda new, old: (new[0], old[1] * new[1])).name('Delta Calculation') \
        #     .group_by(0).aggregate(Sum, 1) \
        #     .map(lambda x: (x[0], 1 - x[1]))
        # TODO causes issues
        delta = u_old_it.join(u_new).where(0).equal_to(0) \
            .using(lambda old, new: (new[0], old[1] - new[1])).name('Delta Calculation')
        delta = delta.reduce_group(MagnitudeGroupReducer()) \
            .name('MagnitudeGroupReducer')
        delta = delta.filter(lambda d: d[1] > epsilon)

        u_new_final = u_old_it.close_with(u_new, delta)

        # Save the newly-computed u and v to the output files;
        u_new_final.write_csv(file_D+"."+str(m), write_mode=WriteMode.OVERWRITE)
        u_old.write_csv(file_D+"."+str(m)+"_OLD", write_mode=WriteMode.OVERWRITE)

        # TODO
        # v = vector_matrix(u_new_final, S)
        # temp_v = v.map(lambda x: (x[0], 0))
        #
        # # Add non-zero elements of v_top_R
        # temp_v = temp_v.union(v/_top_R).group_by(0).aggregate(Sum, 1)
        # v = temp_v
        # v.write_csv(file_z+"."+str(m))

        # P4: Deflation step. Update the primary data matrix S.
        # This replaces deflate in the original Spark implementation.
        # We want k, vector - (u[k] * v) for each vector in the original data
        # Our original data is formatted in tuples of (k, pos, value)
        # First, we add u[k] to each tuple
        print m
        # TODO
        # S = S.join(u_new_final).where(0).equal_to(0) \
        #     .using(lambda s_el, u_el: (s_el[0], s_el[1], s_el[2], u_el[1]))
        #
        # # Now, we multiply u[k] by v[pos] for each tuple
        # S = S.join(v).where(1).equal_to(0) \
        #     .using(lambda s_el, v_el: (s_el[0], s_el[1], s_el[2], s_el[3] * v_el[1]))
        #
        # # We calculate val - (u[r] * v[pos])
        # S = S.map(lambda s_el: (s_el[0], s_el[1], s_el[2] - s_el[3]))
        #
        # # Finally, add up everything
        # S = S.group_by(0, 1).aggregate(Sum, 1)

        print str(m) + " done"

    env.execute(local=True)

    # All done! Write out the matrices as tab-delimited text files, with
    # floating-point values to 6 decimal-point precision.
