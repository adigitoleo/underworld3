echo "starting auto runner"

python3 BLFreeStream.py -restart
for i in {1..100}
do
	python3 BLFreeStream.py
done
echo "ending auto runner"
