import math
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('size_data.pkl', 'rb') as f:
    data = pickle.load(f)

## what do we want to do here?

term1x = data[0][:,0]
term1y = data[0][:,1]

term2x = data[1][:,0]
term2y = data[1][:,1]

term3x = data[2][:,0]
term3y = data[2][:,1]



for index in range(len(term1x)):
    print( (term2x[index] + term3x[index] - term1x[index])/max(([abs(term1x[index]), abs(term2x[index]), abs(term3x[index]) ] ) ) )

print("averages:")
print("x", np.mean(term1x), np.mean(term2x), np.mean(term3x))
print("y", np.mean(term1y), np.mean(term2y), np.mean(term3y))

print("median:")
print("x", np.median(term1x), np.median(term2x), np.median(term3x))
print("y", np.median(term1y), np.median(term2y), np.median(term3y))
print("maxes:")
print("x", np.max(term1x), np.max(term2x), np.max(term3x))
print("y", np.max(term1y), np.max(term2y), np.max(term3y))
print("mins:")
print("x", np.min(term1x), np.min(term2x), np.min(term3x))
print("y", np.min(term1y), np.min(term2y), np.min(term3y))

# Now, lets plot out our difference


with open('bl/differenceDataFreeStream.pkl', 'rb') as f:
    differenceData = pickle.load(f)

logDifferenceData = []
counter = 0
counters = []
for el in differenceData:
    try:
        nel = math.log(el)
        logDifferenceData.append(nel)
        counters.append(counter)
    except:        
        print("tried taking log of zero")
    counter += 1 
print("here is last difference:", differenceData[-1])
plt.plot(counters, logDifferenceData)
plt.title("log difference Data")
plt.show()
print("done")

