import matplotlib.pyplot as plt
import math
import pickle
import numpy as np

with open("results/outputmarkers.pkl", 'rb') as f:
    data = pickle.load(f)

time = data[0]
vrms = data[1]

plt.plot(time, vrms)
plt.axhline(42.86, linestyle= "dotted")
plt.axhline(np.mean(vrms[-100:]), linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("vrms")
plt.legend(("vrms", "target "+str(42.86), "computed" + str(np.mean(vrms[-100:]))[:4] ))
plt.title("case 1(a) swarm method T degree = 1, V degree = 2 at 96x96")
plt.savefig("vrmsNice.png")

