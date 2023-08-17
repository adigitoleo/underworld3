import matplotlib.pyplot as plt
import pickle

def getTimePlot(path, times, position):
    timeDict = dict()
    for index in range(times):
        with open(path+str(index) + ".pkl", 'rb') as f:
            data = pickle.load(f)

        timeDict[index] = data

    t = []
    vals = []
    for index in range(times):
        s, vs = timeDict[index]

        vals.append(vs[position])
        t.append(index)
    plt.plot(t, vals)
    plt.xlabel("timeStep")
    plt.ylabel("Boundary Layer thickness")
    plt.savefig("behavourInTime.png")

getTimePlot("bl/dataFreeStream", 100, 10)

    
