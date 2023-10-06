#!/bin/bash

## this script is desgined as an example for how to use this code

## 
#!/bin/bash

CASE_NAME="myExampleCase"

mkdir "$CASE_NAME"
cp case1runner.sh "$CASE_NAME"/case1runner.sh
cp autoanalyser.py "$CASE_NAME"/autoanalyser.py
cp run.py "$CASE_NAME"/run.py
cd "$CASE_NAME"



## you can change the settings for your case by setting the following parameters for your run
RA=10000 ## Rayeligh number of your simulation
NUM_LOOPS=100 ## number of loops the simulation will do. At the end each loop, the program saves and restarts from the ground up
## the number of total timesteps will be NUM_LOOPS * NUM_STEPS
NSTEPS=20 ## number of steps taken in each loop
STOPPING_TIME=100 ## limit on the time in the simulation
T_DEGREE=3 ## Temperature Degree
Q_DEGREE=4 ## quadrature degree
WIDTH=1 ## inverse width of the gaussian used for mask functions when computing NU
SPEEDUP=5 ## how many times faster the program will run, for accurate results use a value of 1

## loop over each of the resolutions
for i in 6 9 12 15 18 21
do
    sh case1runner.sh --res $i --Ra $RA --num_loops $NUM_LOOPS --stoppingTime $STOPPING_TIME --TDegree $T_DEGREE --qdegree $Q_DEGREE --width $WIDTH --speedUp $SPEEDUP --nsteps $NSTEPS&
done
wait ## wait for all proccesses to stop 

## go analyse all of them
python3 autoanalyser.py
