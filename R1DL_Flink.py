import argparse
import numpy as np
import os.path
import scipy.linalg as sla
import datetime
import os
import psutil

from flink.plan.Environment import get_environment
from flink.plan.Constants import INT, STRING
from flink.functions.MapPartitionFunction import MapPartitionFunction

class ZipWithIndex(MapPartitionFunction):
    def map_partition(self, iterator, collector):
        count = 0
        for s in iterator: count += 1
        collector.collect((count, s))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Flictionary Learning',
        add_help = 'How to use', prog = 'python R1DL_Flink.py <args>')

    # Inputs.
    parser.add_argument("-i", "--input", required = True,
        help = "Input file containing the matrix S.")
    # parser.add_argument("-t", "--rows", type = int, required = True,
    #     help = "Number of rows (observations) in the input matrix S.")
    # parser.add_argument("-p", "--cols", type = int, required = True,
    #     help = "Number of columns (features) in the input matrix S.")
    # parser.add_argument("-r", "--pnonzero", type = float, required = True,
    #     help = "Percentage of non-zero elements.")
    # parser.add_argument("-m", "--mDicatom", type = int, required = True,
    #     help = "Number of the dictionary atoms.")
    # parser.add_argument("-e", "--epsilon", type = float, required = True,
    #     help = "The value of epsilon.")

    # # Outputs.
    # parser.add_argument("-d", "--dictionary", required = True,
    #     help = "Output path to dictionary file.(file_D)")
    # parser.add_argument("-o", "--output", required = True,
    #     help = "Output path to z matrix.(file_z)")
    # parser.add_argument("-prefix", "--prefix", required = True,
    #     help = "Prefix strings to the output files")

    args = vars(parser.parse_args())

    env = get_environment()
    env.set_degree_of_parallelism(1)
    raw_data = env.read_text(args['input'])
    raw_data \
        .map_partition(ZipWithIndex(), [INT, STRING]) \
        .output()

    env.execute(local = True)
