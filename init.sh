#!/bin/bash
# Make sure the simulation data is clear
rm west.h5 

# Define the arguments for the basis states (used for generating initial 
# states; in this case we only have one), and target states (used for
# knowing when to recycle trajectories). In this example, we recycle
# trajectories as they reach the bound state; we focus on sampling  
# the binding process (and not the unbinding process).

BSTATE_ARGS="--bstate-file bstates/bstates.txt"
TSTATE_ARGS="--tstate unbound,20.0"

# Initialize the simulation, creating the main WESTPA data file (west.h5)
# The "$@" lets us take any arguments that were passed to init.sh at the
# command line and pass them along to w_init.
$WEST_ROOT/bin/w_init "$@" $BSTATE_ARGS $TSTATE_ARGS --segs-per-state 7 \
  --work-manager=processes
