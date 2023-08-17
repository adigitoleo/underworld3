import math
import matplotlib.pyplot as plt
import pickle 
import numpy as np
with open('advDiffMethod/LowResCase1/results/outputmarkers.pkl', 'rb') as f:
    m1Data = pickle.load(f)
    m1T = m1Data[0]
    m1V = m1Data[1]

with open('advDiffSwarmMethod/swarm_adv_diff_fast/results/outputmarkers.pkl', 'rb') as f:
    m2Data = pickle.load(f)
    m2T = m2Data[0]
    m2V = m2Data[1]

m1VAverage = np.mean(m1V[-100:])
m2VAverage = np.mean(m2V[-100:])
plt.scatter(m1T, m1V, color = 'r')
plt.scatter(m2T, m2V, color = 'b')
plt.axhline(m1VAverage, color = 'r', linestyle='dashed')
plt.axhline(m2VAverage, color = 'b', linestyle='dotted')
plt.legend(('adv_diff', 'swarm adv_diff', 'adv_diff s.s vrms='+str(m1VAverage)[:5],'swarm adv_dff s.s vrms='+str(m2VAverage)[:4]))
plt.xlabel("Time")
plt.ylabel("vrms")
plt.show()

