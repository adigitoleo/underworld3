echo 'starting autorunner'
python3 runSwarm.py -restart

for i in {1..100}
do
	python3 runSwarm.py
done

echo 'ending autorunner'
