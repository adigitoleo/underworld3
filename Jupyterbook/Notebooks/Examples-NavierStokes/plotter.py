import matplotlib.pyplot as plt
import pickle

with open('bl/data4.pkl','rb') as f:
    data = pickle.load(f)
    space = data[0]
    val = data[1]

val2 = [el**2 for el in val]
plt.plot(space, val2)
plt.xlabel("x")
plt.ylabel("layer thickness squared")
plt.show()
