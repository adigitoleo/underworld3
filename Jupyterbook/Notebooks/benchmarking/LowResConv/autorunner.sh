echo 'starting'

python3 runLowRes.py

for i in {1..100}
do
	python3 runLowResC.py
done

echo 'done'
