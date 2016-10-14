#!/usr/bin/python2
# Flictionary Learning: Rank-1 Dictionary Learning in Flink!
# Probably PEP 8 compliant.

import argparse

import itertools
import numpy as np
import scipy.linalg as sla
import datetime
import os
import math

from flink.functions.Aggregation import Sum
from flink.functions.FlatMapFunction import FlatMapFunction
from flink.functions.GroupReduceFunction import GroupReduceFunction
from flink.plan.Constants import Order, WriteMode
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


def get_top_v(r, u, s):
    """
    Calculates v, and then returns the top R elements of v
    :param r: R
    :param u: Old u
    :param s: S matrix
    :return: Top R elements of v
    """
    v = vector_matrix(u, s)

    # sort v by using sort_group after grouping on a dummy field
    v = v.map(lambda x: (x[0], x[1], 0)).name('VPreSorter')
    v = v \
        .group_by(2).sort_group(1, Order.DESCENDING) \
        .reduce_group(lambda i, c: [(i_[0], i_[1]) for i_ in i]) \
        .name('VSorter')

    v_top_R = v.first(R).name('VFirst')
    return v_top_R


def vector_matrix(u__, S_):
    """
    Generates v using u.
    """
    v_ = u__ \
        .join(S_).where(0).equal_to(0) \
        .using(lambda u_el, s_el: (s_el[1], s_el[2] * u_el[1])) \
        .name('VectorMatrix')

    v_ = v_ \
        .group_by(0) \
        .aggregate(Sum, 1) \
        .name('VectorMatrixPost')

    return v_


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
    Normalizes a vector in (index, old, new_value) format.
    """
    def reduce(self, iterator, collector):
        data = list(iterator)
        mean = 0
        mag = 0
        length = len(data)

        for val in data:
            mean += val[1]
            mag += (val[1]) ** 2

        mean /= length
        mag = math.sqrt(mag)

        for val in data:
            new_val = val[1] - mean
            new_val /= mag
            collector.collect((val[0], new_val))


class MagnitudeGroupReducer(GroupReduceFunction):
    """
    Calculates the magnitude of a vector.
    """
    def reduce(self, iterator, collector):
        mag = 0
        for val in iterator:
            mag += (val[1]) ** 2
        mag = math.sqrt(mag)
        collector.collect((0, mag))


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
    parser.add_argument("-m", "--mdicatom", type=int, required=True,
                        help="Number of the dictionary atoms.")
    parser.add_argument("-e", "--epsilon", type=float, required=True,
                        help="The value of epsilon.")
    parser.add_argument("-z", "--seed", type=long, required=False,
                        help="Random seed. (optional)")

    # Outputs.
    parser.add_argument("-d", "--dictionary", required=True,
                        help="Output path to dictionary file.(file_D)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output path to z matrix.(file_z)")
    parser.add_argument("-x", "--prefix", required=True,
                        help="Prefix strings to the output files")

    args = vars(parser.parse_args())

    # Initialize the Flink environment.
    env = get_environment()

    # Read the data and convert it into a data set of lines
    raw_data = env.read_text(args['input'])

    # Convert each line to a tuple: (row number, vec pos, value)
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
    M = args['mdicatom']  # dimensionality of the learned dictionary
    R = int(round(args['pnonzero'] * P))  # enforces sparsity

    ##################################################################
    # Apparently Flink doesn't like np.arrays. So for vectors, we use Flink
    # DataSets with indexes corresponding to their position in the vector.
    ##################################################################

    # Information we need.
    max_iterations = int(P * 10)
    file_D = os.path.join(args['dictionary'], "{}_D.txt".format(args["prefix"]))
    file_z = os.path.join(args['output'], "{}_z.txt".format(args["prefix"]))

    # Start the loop!
    for m in range(M):
        # Generate a random vector, subtract off its mean, and normalize it.
        # TODO get rid of numpy entirely?
        u_old = env.from_elements(0).flat_map(RandomVectorFlatMapper(T))
        u_old_it = u_old.iterate(max_iterations)

        # Start the inner loop: this learns a single atom.
        # P2: Vector-matrix multiplication step. Computes v.

        # We can't broadcast partial iteration results, so
        # the Flink implementation is different from Spark.
        v = get_top_v(R, u_old_it, S)

        # P1: Matrix-vector multiplication step. Computes u.

        # We can use a join to do a dot product.
        # (S * v by row-wise multiplication)
        # This replaces matrix_vector in the original Spark implementation.
        # Multiply each corresponding element
        u_new = v \
            .join(S).where(0).equal_to(1) \
            .using(lambda v_el, s_el: (s_el[0], s_el[1], s_el[2] * v_el[1])) \
            .name('MatrixVector')

        # Now, add up all the products to get the dot product result, and remove
        # the vector position field (second field from the left)
        u_new = u_new.group_by(0) \
            .aggregate(Sum, 2) \
            .map(lambda x: (x[0], x[2]))
        ################################################################

        u_new = u_new.reduce_group(NormalizeVectorGroupReducer()).name('NormalizeVector')

        # Update for the next iteration
        delta = u_new.join(u_old_it).where(0).equal_to(0) \
            .using(lambda new, old: (new[0], old[1] - new[1]))
        delta = delta.reduce_group(MagnitudeGroupReducer()) \
            .name('MagnitudeGroupReducer')
        delta = delta.filter(lambda d: d[1] > epsilon)

        u_new_final = u_old_it.close_with(u_new, delta)

        # Save the newly-computed u and v to the output files;
        u_new_final.write_csv(file_D+"."+str(m), write_mode=WriteMode.OVERWRITE)

        # Compute new v from final u
        v_final = get_top_v(R, u_new_final, S)

        # Fill in missing spots with zeroes
        v_zeroes = env.from_elements(*[(t, 0) for t in range(T)])
        v_final = v_final.union(v_zeroes)
        v_final = v_final.group_by(0).aggregate(Sum, 1)
        v_final.write_csv(file_z+"."+str(m), write_mode=WriteMode.OVERWRITE)

        # P4: Deflation step. Update the primary data matrix S.
        # This replaces deflate in the original Spark implementation.
        # We want k, vector - (u[k] * v) for each vector in the original data
        # Our original data is formatted in tuples of (k, pos, value)
        # First, we add u[k] to each tuple
        S_temp = S.join(u_new_final).where(0).equal_to(0) \
            .using(lambda s_el, u_el: (s_el[0], s_el[1], s_el[2], u_el[1]))

        # Now, we multiply u[k] by v[pos] for each tuple
        S_temp = S_temp.join(v).where(1).equal_to(0) \
            .using(lambda s_el, v_el: (s_el[0], s_el[1], s_el[2], s_el[3] * v_el[1]))

        # We calculate val - (u[r] * v[pos])
        S_temp = S_temp.map(lambda s_el: (s_el[0], s_el[1], s_el[2] - s_el[3]))

        # Finally, add up everything
        S = S_temp.group_by(0, 1).aggregate(Sum, 1)

    env.execute(local=True)

    # All done! Write out the matrices as text files.
