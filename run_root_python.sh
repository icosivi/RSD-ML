#!/bin/bash
# Sets up the PyROOT environment (this ROOT build is compiled against Python 3.11)
# and runs the given python script with it.
#
# Usage: ./run_root_python.sh <script.py> [args...]
set -e
source "/Users/icosivi/Desktop/pip_rooot_py311/root_install/bin/thisroot.sh"
exec /Users/icosivi/.pyenv/versions/3.11.7/bin/python3.11 "$@"
