echo 'starting autorunner'
python3 runSwarm.py

for i in {1..100}
do
	python3 runSwarmC.py
done

echo 'ending autorunner'
