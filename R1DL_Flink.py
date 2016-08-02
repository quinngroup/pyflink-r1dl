#!/usr/bin/python2
# Flictionary Learning: Rank-1 Dictionary Learning in Flink!
# Probably PEP 8 compliant.

import argparse
import numpy as np
import scipy.linalg as sla
import datetime
import os
import psutil
from flink.functions.Aggregation import Sum
from flink.functions.FlatMapFunction import FlatMapFunction
from flink.functions.MapFunction import MapFunction
from flink.plan.Environment import get_environment

from functions import select_topr


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
    x = np.array(map(float, line.strip().split()))

    # x.strip() -- strips off trailing whitespace from the string
    # .split("\t") -- splits the string into a list of strings, splitting on tabs
    # map(float, list) -- converts each element of the list from strings to floats
    # np.array(list) -- converts the list of floats into a numpy array

    # comment by Xiang: the following normalization commands work for vector u,
    # but not work here. I have double-checked it with pre-normalized matrix;
    # x -= x.mean()  # 0-mean.
    # x /= sla.norm(x)  # Unit norm.
    return x


class VectorMatrixFlatMapper(FlatMapFunction):
    def flat_map(self, row, collector):
        """
        Applies u * S by row-wise multiplication, followed by a reduction on
        each column into a single vector.
        """

        u = np.array(self.context.get_broadcast_variable("_U_")[0][1])  # rm index

        # comment by Xiang: in this case there is T*log(T) complexity?
        # comment by Xiang: Also, whenever a "row_index, vector = row" is called,
        # there will be a reading on the portion of S on each node, right?

        row_index, vector = row  # Split up the [key, value] pair.

        # Generate a list of [key, value] output pairs, one for each nonzero
        # element of vector.
        # comment by Xiang: the code below seems calculating all elements for
        # vector v, rather than only the nonzero elements;
        # comment by Xiang: also I'm puzzled why we are using the "append" function,
        # as the output of this should be of the same size?
        for i in range(vector.shape[0]):
            collector.collect((i, u[row_index] * vector[i]))


class NumpyCollapseFlatMapper(FlatMapFunction):
    """
    Convert an indexed DataSet of numbers (broadcasted) into a DataSet with a single Numpy vector element.
    """

    def flat_map(self, x, collector):
        data = np.array(self.context.get_broadcast_variable("data"))
        collector.collect(np.take(sorted(data), indices=1, axis=1))


# class SelectTopRFlatMapper(MapFunction):
#     def __init__(self, r):
#         self.r = r
#         super(SelectTopRFlatMapper, self).__init__()
#
#     def flat_map(self, element):
#         v_ = self.context.get_broadcast_variable("_V_")
#         v_ = np.array(v)
#



class MatrixVectorMapper(MapFunction):
    def __init__(self, R_):
        self.R = R_
        super(MatrixVectorMapper, self).__init__()

    def map(self, row):
        """
        Applies S * v by row-wise multiplication. No reduction needed, as all the
        summations are performed within this very function.
        """
        k, vector = row

        # Extract the broadcast variables.
        v_collapsed_ = np.array(self.context.get_broadcast_variable("v_collapsed")[0])
        v_top = np.array(self.context.get_broadcast_variable("v_top")[0])

        _V_ = v_collapsed_[v_top]
        _I_ = v_top

        # Perform the multiplication using the specified indices in both arrays.
        inner_prod = np.dot(vector[_I_], _V_)

        # That's it! Return the [row, inner product] tuple.
        return k, inner_prod


class VFinalizerMapper(MapFunction):
    def map(self, value):
        v_ = np.array(self.context.get_broadcast_variable("v"))
        value[indices_V] = v_[indices_V]
        return value


class DeflateMapper(MapFunction):
    def map(self, row):
        """
        Deflates the data matrix by subtracting off the outer product of the
        broadcasted vectors and returning the modified row.
        """
        k, vector = row
        # It's important to keep order of operations in mind: we are computing
        # (and subtracting from S) the outer product of u * v. As we are operating
        # on a row-distributed matrix, we therefore will only iterate over the
        # elements of v, and use the single element of u that corresponds to the
        # index of the current row of S.
        # Got all that? Good! Explain it to me.
        u = np.array(self.context.get_broadcast_variable("_U_")[0])
        v = np.array(self.context.get_broadcast_variable("_V_")[0])
        return k, vector - (u[k] * v)


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
    S = input_to_row_matrix(raw_data)

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
    R = args['pnonzero'] * P  # enforces sparsity
    u_new = np.zeros(T)  # atom updates at each iteration
    v = np.zeros(P)

    indices_V = np.zeros(R)  # for top-R sorting

    max_iterations = P * 10
    file_D = os.path.join(args['dictionary'], "{}_D.txt".format(args["prefix"]))
    file_z = os.path.join(args['output'], "{}_z.txt".format(args["prefix"]))

    # Start the loop!
    for m in range(M):
        # Generate a random vector, subtract off its mean, and normalize it.
        u_old = np.random.random(T)
        u_old -= u_old.mean()
        u_old /= sla.norm(u_old)
        # wrap in a DataSet so we can iterate and use a numpy vector
        u_old_ds = env.from_elements((0, u_old))

        u_old_ds_it = u_old_ds.iterate(max_iterations)
        delta = 2 * epsilon

        # Start the inner loop: this learns a single atom.
        # P2: Vector-matrix multiplication step. Computes v.
        v = S \
            .flat_map(VectorMatrixFlatMapper()) \
            .with_broadcast_set("_U_", u_old_ds_it) \
            .group_by(0) \
            .aggregate(Sum, 0)

        v_collapsed = env.from_elements(0).flat_map(NumpyCollapseFlatMapper()) \
            .with_broadcast_set("data", v)
        v_top_R = v_collapsed.map(lambda x: select_topr(x, R))

        # P1: Matrix-vector multiplication step. Computes u.
        u_new = S \
            .map(MatrixVectorMapper(R)) \
            .with_broadcast_set("v_collapsed", v_collapsed) \
            .with_broadcast_set("v_top", v_top_R)

        u_new_collapsed = env.from_elements(0).flat_map(NumpyCollapseFlatMapper()) \
            .with_broadcast_set("data", u_new)
        u_new_collapsed = u_new_collapsed \
            .map(lambda x: np.take(sorted(x), indices=1, axis=1))

        # Subtract off the mean and normalize.
        u_new_collapsed = u_new_collapsed \
            .map(lambda x: x - x.mean())
        u_new_collapsed = u_new_collapsed \
            .map(lambda x: x / sla.norm(x))

        u_new_collapsed = u_new_collapsed.zip_with_index()

        # Update for the next iteration.
        delta = u_old_ds_it.join(u_new_collapsed).where(0).equal_to(0) \
            .using(lambda old, new: new - old)
        delta = delta.filter(lambda d: d > epsilon)
        u_new_final = u_old_ds_it.close_with(u_new_collapsed, delta)

        # Save the newly-computed u and v to the output files;
        u_new_final_expanded = u_new_final.flat_map(lambda x, c: list(x))
        u_new_final_expanded.write_csv(file_D)

        temp_v = v_collapsed.map(lambda x: np.zeros(x.shape))
        temp_v = temp_v.map(VFinalizerMapper()) \
            .with_broadcast_set("v", v_collapsed)

        v_collapsed = temp_v
        v_expanded = v_collapsed.flat_map(lambda x, c: list(x))
        v_expanded.write_csv(file_z)

        # P4: Deflation step. Update the primary data matrix S.
        print m
        S = S.map(DeflateMapper()).group_by(0).aggregate(Sum, 0) \
            .with_broadcast_set("_U_", u_new_final) \
            .with_broadcast_set("_V_", v_collapsed)

    env.execute(local=True)

    # All done! Write out the matrices as tab-delimited text files, with
    # floating-point values to 6 decimal-point precision.
    print datetime.datetime.now()
    process = psutil.Process(os.getpid())
    print process.memory_info().rss
