#!/usr/bin/python3
# Flictionary Learning: Rank-1 Dictionary Learning in Flink!
# Probably PEP 8 compliant.

import argparse
import math
import os
import random
import sys

import numpy as np

from flink.functions.Aggregation import Sum
from flink.functions.FlatMapFunction import FlatMapFunction
from flink.functions.GroupReduceFunction import GroupReduceFunction
from flink.plan.Constants import Order, WriteMode, INT, FLOAT
from flink.plan.Environment import get_environment


###################################
# Flink helper functions
###################################

def parse_and_normalize(line):
    """
    Utility function. Parses a line of text into a floating point array, then
    whitens the array.
    """
    return tuple(map(float, line.strip().split()))


class SExploderFlatMapper(FlatMapFunction):
    def flat_map(self, x, collector):
        data = enumerate(parse_and_normalize(x[1]))
        for y in data:
            collector.collect((x[0], y[0], y[1]))


def get_top_v(r, u, s):
    """
    Calculates v, and then returns the top R elements of v
    :param r: R
    :param u: Old u
    :param s: S matrix
    :return: Top R elements of v
    """
    v_ = vector_matrix(u, s)

    # sort v by using sort_group after grouping on a dummy field
    v_ = v_.map(lambda x: (x[0], x[1], 0)).name('VPreSorter') \
        .set_parallelism(parallelism)
    v_ = v_ \
        .group_by(2).sort_group(1, Order.DESCENDING) \
        .reduce_group(lambda i, c: [(i_[0], i_[1]) for i_ in i]) \
        .set_parallelism(parallelism) \
        .name('VSorter')

    v_top_R_ = v_.first(r).set_parallelism(parallelism).name('VFirst')
    return v_top_R_


def vector_matrix(u__, S_):
    """
    Generates v using u.
    """
    v_ = u__ \
        .join_with_huge(S_).where(0).equal_to(0) \
        .using(lambda u_el, s_el: (s_el[1], s_el[2] * u_el[1])) \
        .set_parallelism(parallelism) \
        .name('VectorMatrix')

    v_ = v_ \
        .group_by(0) \
        .aggregate(Sum, 1) \
        .set_parallelism(parallelism) \
        .name('VectorMatrixPost')

    return v_


def random_vector(environment, num_elements, rng=random):
    """
    Generates a vector of random numbers in in [0.0, 1.0). Does NOT normalize.
    """
    dataset = environment.generate_sequence(1, num_elements) \
        .set_parallelism(1).zip_with_index()
    dataset = dataset.map(lambda x: (x[0], rng.random())).set_parallelism(1)
    return dataset


class NormalizeVectorGroupReducer(GroupReduceFunction):
    """
    Normalizes a vector in (index, value) format.
    """

    def reduce(self, iterator, collector):
        data = list(iterator)
        mean = 0.0
        mag = 0.0
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


class RngWrapper(object):
    """Compatibility wrapper for different RNG methods."""

    def random(self):
        """Assign something to me."""
        pass


def initialize_rng(seed=None, java=False):
    wrapper = RngWrapper()
    if java:
        import javarandom
        r = javarandom.Random(seed)
        wrapper.random = r.nextDouble
    else:
        r = np.random.RandomState(seed)
        wrapper.random = r.random_sample
    return wrapper


def get_temporary_S_path(m):
    return os.path.join(temporary_directory, 'temp_S.' + str(m))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flictionary Learning',
                                     add_help='How to use',
                                     prog='.../pyflink2.sh R1DL_Flink.py -')

    parser.add_argument("-q", "--quiet", action='store_true',
                        help="Don't print out any verbose information. "
                             "Note that this doesn't cover the output from "
                             "Flink itself. (optional)")
    parser.add_argument("-l", "--local", action='store_true',
                        help="Run script on the local machine "
                             "i.e. NOT a cluster. (optional)")
    parser.add_argument("-j", "--javarand", action='store_true',
                        help="Use an implementation of the Java RNG. "
                             "This option allows you to directly compare "
                             "results between the Java and Python "
                             "implementations. Requires java-random package. "
                             "(optional)")

    # Inputs.
    parser.add_argument("-i", "--input", required=True,
                        help="Input file containing the matrix S.")
    parser.add_argument("-t", "--rows", type=int, required=True,
                        help="Number of rows (observations) in the "
                             "input matrix S.")
    parser.add_argument("-p", "--cols", type=int, required=True,
                        help="Number of columns (features) in the "
                             "input matrix S.")
    parser.add_argument("-r", "--pnonzero", type=float, required=True,
                        help="Percentage of non-zero elements.")
    parser.add_argument("-m", "--mdicatom", type=int, required=True,
                        help="Number of the dictionary atoms.")
    parser.add_argument("-e", "--epsilon", type=float, required=True,
                        help="The value of epsilon.")
    parser.add_argument("-a", "--parallelism", type=int, required=False,
                        help="Parallelism to use.")
    parser.add_argument("-z", "--seed", type=int, required=False,
                        help="Random seed. (optional)")

    # Outputs.
    parser.add_argument("-d", "--dictionary", required=True,
                        help="Output path to dictionary file. (file_D)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output path to z matrix. (file_z)")
    parser.add_argument("-x", "--prefix", required=True,
                        help="Prefix strings to the output files")
    parser.add_argument("-y", "--temporary", required=True,
                        help="Temporary directory (on HDFS if you're using a "
                             "cluster) for storing intermediate files. A suffix"
                             "will be added to the end of this path.")

    args = vars(parser.parse_args())

    ##################################################################
    # Here's where the real fun begins.
    #

    # First, we're going to initialize some variables we'll need for the
    # following operations. Next, we'll start the optimization loops. Finally,
    # we'll perform the stepping and deflation operations until convergence.
    #
    # Sound like fun?
    ##################################################################

    # parallelism to use. Python API has some quirks with parallelism, like not
    # picking up on the default parallelism sometimes, or not using a specified
    # parallelism; therefore we use this parameter.
    parallelism = args['parallelism']
    T = args['rows']
    P = args['cols']
    epsilon = args['epsilon']  # convergence stopping criterion
    M = args['mdicatom']  # dimensionality of the learned dictionary
    R = int(round(args['pnonzero'] * P))  # enforces sparsity

    # seed random number generator
    rng = initialize_rng(args['seed'], args['javarand'])

    # Print out some useful information for the user
    if not args['quiet']:
        print('====================')
        print('Flictionary learning: R1DL in Flink!')
        print('Local mode: {local}'.format(**args))
        print('Input has {rows} rows and {cols} cols.'.format(**args))
        args['R'] = R
        print('epsilon = {epsilon}, M = {mdicatom}, R = {R}'.format(**args))

        if args['javarand']:
            print('Using java-random.')
        if args['seed'] is not None:
            print('Using random seed {seed}.'.format(**args))

        print('====================')
        sys.stdout.flush()

    ##################################################################
    # Apparently Flink doesn't like np.arrays. So for vectors, we use Flink
    # DataSets with indexes corresponding to their position in the vector.
    ##################################################################

    # Information we need.
    max_iterations = int(P * 10)
    file_D = os.path.join(args['dictionary'], "{prefix}_D.txt".format(**args))
    file_z = os.path.join(args['output'], "{prefix}_z.txt".format(**args))

    r1dl_id = '{0:.8f}'.format(random.random())  # used for temporary files
    temporary_directory = os.path.join(args['temporary'],
                                       'r1dl_data_' + r1dl_id)

    # Initialize the Flink environment.
    env_first = get_environment()
    env_first.set_parallelism(parallelism)

    # Read the data and convert it into a data set of lines
    raw_data = env_first.read_text(args['input'])

    # Convert each line to a tuple: (row number, vec pos, value)
    raw_data \
        .zip_with_index() \
        .flat_map(SExploderFlatMapper()) \
        .name('SExploderFlatMapper') \
        .write_csv(get_temporary_S_path(0), write_mode=WriteMode.OVERWRITE)

    env_first.execute(local=args['local'])

    # Start the loop!
    for m in range(M):
        # If seed is set, the RNG will be in the same state with every generated
        # u vector, because the Python API spawns new Python processes for each
        # operation. Thus, if seed is set, skip the number of random numbers
        # that would have generated if the operations had taken place in the
        # same Python process.
        # This might take up some performance time, but is included only when
        # the seed is specified for reproductivity purposes.
        if args['seed']:
            for z in range(m * T):
                rng.random()

        env = get_environment()
        # needed to prevent normalization etc. group reduce functions from
        # improperly running in parallel
        env.set_parallelism(1)

        S = env.read_csv(get_temporary_S_path(m), (INT, INT, FLOAT))

        # Generate a random vector, subtract off its mean, and normalize it.
        u_old = random_vector(env, T, rng)
        u_old = u_old.reduce_group(NormalizeVectorGroupReducer()) \
            .set_parallelism(1) \
            .name('Random u')
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
            .join_with_huge(S).where(0).equal_to(1) \
            .using(lambda v_el, s_el: (s_el[0], s_el[1], s_el[2] * v_el[1])) \
            .set_parallelism(parallelism) \
            .name('MatrixVector')

        # Now, add up all the products to get the dot product result,
        # and remove the vector position field (second field from the left)
        u_new = u_new.group_by(0) \
            .aggregate(Sum, 2) \
            .set_parallelism(parallelism) \
            .project(0, 2) \
            .set_parallelism(parallelism)
        ################################################################

        u_new = u_new.reduce_group(NormalizeVectorGroupReducer()) \
            .set_parallelism(1) \
            .name('NormalizeVector')

        # Update for the next iteration
        delta = u_new.join_with_huge(u_old_it).where(0).equal_to(0) \
            .using(lambda new, old: (new[0], old[1] - new[1])) \
            .set_parallelism(parallelism)
        delta = delta.reduce_group(MagnitudeGroupReducer()) \
            .set_parallelism(1)
        delta = delta.filter(lambda d: d[1] > epsilon) \
            .set_parallelism(parallelism)

        u_new_final = u_old_it.close_with(u_new, delta) \
            .set_parallelism(parallelism)

        # Save the newly-computed u and v to the output files;
        u_new_final.write_csv(file_D + "." + str(m),
                              write_mode=WriteMode.OVERWRITE) \
            .set_parallelism(parallelism)

        # Compute new v from final u
        v_final = get_top_v(R, u_new_final, S)

        # Fill in missing spots with zeroes
        # Fill with 0.0, not 0 or else Flink thinks these are
        # LONGS and NOT doubles!
        v_zeroes = env.from_elements(*[(p, 0.0) for p in range(P)])
        v_final = v_final.union(v_zeroes)
        v_final = v_final.group_by(0).aggregate(Sum, 1) \
            .set_parallelism(parallelism)
        v_final.write_csv(file_z + "." + str(m), write_mode=WriteMode.OVERWRITE) \
            .set_parallelism(parallelism)

        # P4: Deflation step. Update the primary data matrix S.
        # This replaces deflate in the original Spark implementation.
        # We want k, vector - (u[k] * v) for each vector in the original data
        # Our original data is formatted in tuples of (k, pos, value)
        # First, we add u[k] to each tuple
        S_temp = S.join_with_tiny(u_new_final).where(0).equal_to(0) \
            .project_first(0, 1, 2).project_second(1) \
            .set_parallelism(parallelism)

        # Now, we multiply u[k] by v[pos] and get
        # val - (u[r] * v[pos]) for each tuple
        S_temp = S_temp.join_with_tiny(v_final).where(1).equal_to(0) \
            .using(lambda s_el, v_el: (s_el[0],
                                       s_el[1],
                                       s_el[2] - (s_el[3] * v_el[1]))) \
            .set_parallelism(parallelism)

        # Finally, add up everything.
        # Write the new S matrix to a temporary file
        # for use in the next dict atom
        S = S_temp.group_by(0, 1).aggregate(Sum, 1) \
            .set_parallelism(parallelism) \
            .write_csv(get_temporary_S_path(m + 1),
                       write_mode=WriteMode.OVERWRITE) \
            .set_parallelism(parallelism)

        env.execute(local=args['local'])

        # All done! Write out the matrices as text files.
