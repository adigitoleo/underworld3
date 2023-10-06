#!/bin/bash

# Default values (if not provided)
res=6
num_loops=100
Ra=10000.0
stoppingTime=10
TDegree=3
qdegree=4
width=1
nsteps=100
speedUp=1
nsteps=100
useSwarm=False

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --res) res="$2"; shift ;;
        --Ra) Ra="$2"; shift ;; 
        --num_loops) num_loops="$2"; shift ;;
        --stoppingTime) stoppingTime="$2"; shift;;
        --TDegree) TDegree="$2"; shift;;
        --qdegree) qdegree="$2"; shift;;
        --width) width="$2"; shift;;
        --nsteps) nsteps="$2"; shift;;
        --speedUp) speedUp="$2"; shift;;
        --useSwarm) useSwarm="$2"; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "starting autorunner"

# Run the initial command
python3 case1General.py --restart=True --res=$res --temperatureIC=1.0 --nsteps=$nsteps --TDegree=$TDegree --Ra=$Ra --save_every=1 --speedUp=$speedUp --stoppingTime=$stoppingTime --width=$width --qdegree=$qdegree --useSwarm=$useSwarm

# Loop for running the command
for (( i=1; i<=$num_loops; i++ ))
do
    python3 case1General.py --restart=False --res=$res --temperatureIC=1.0 --nsteps=$nsteps --TDegree=$TDegree --Ra=$Ra --save_every=1 --speedUp=$speedUp --stoppingTime=$stoppingTime --width=$width --qdegree=$qdegree --useSwarm=$useSwarm
done

echo "ending autorunner"
