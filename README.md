# PyFlink R1DL

__NOTE:__ Due to issues with parallelism in the Python API, this script eats a lot of RAM and is not production-ready (hopefully it will be soon).

Rank-1 Dictionary Learning in PyFlink, featured in [_Implementing dictionary learning in Apache Flink, Or: How I learned to relax and love iterations_](http://ieeexplore.ieee.org/document/7840869/).

If you like Java or you want something that is probably more stable and doesn't use the incomplete Flink Python API, an implementation is available at [quinngroup/flink-r1dl](https://github.com/quinngroup/flink-r1dl).

Use with [the `new-iterations-with-multiops` branch of GEOFBOT/flink](https://github.com/GEOFBOT/flink/tree/new-iterations-with-multiops) ([some binaries available here](https://github.com/GEOFBOT/flink/releases/tag/iterations-working)) which has the bulk iterations implementation and tweaks needed to run this script.

## Usage
Run `.../pyflink2.sh R1DL_Flink.py` without extra arguments for usage information.

Input files are made up of rows of whitespace-separated numbers (whitespace separating the columns).
