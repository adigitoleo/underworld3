import os
import re
import pickle
import matplotlib.pyplot as plt
import scipy
import pandas as pd

def extract_res_from_directory(dir_name):
    """
    Extracts 'res' value from a directory name if it matches the pattern.
    """
    # This regular expression matches the 'res' pattern at the beginning of a string.
    match = re.search(r'res(\d+)', dir_name)
    if match:
        return int(match.group(1))
    return None
def extract_data(dir_name):
    with open(dir_name +"/tmp/markers.pkl", 'rb') as f:
        data = pickle.load(f)
    return [data[1][-1], data[2][-1]]

base_dir = '.'

all_directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
res_values = {}

for dir_name in all_directories:
    res_value = extract_res_from_directory(dir_name)
    if res_value is not None:
        res_values[dir_name] = res_value

for dir_name, res_value in res_values.items():
    print(f"Directory: {dir_name}, Res Value: {res_value}")


directories = list(res_values.keys())
directories.sort()


resDict = dict()

for x in directories:
    resDict[res_values[x]] = [extract_data(x)[0], extract_data(x)[1]]

reses = list(resDict.keys())
reses.sort()
print(reses)
vrms = [resDict[r][0] for r in reses]
nu = [resDict[r][1] for r in reses]


from scipy.optimize import curve_fit


def model(x, a, n, b):
    return (a*(x)**n) + b


lowerBounds = [-100, -10,10]
upperBounds = [0, 0,100]


plt.scatter(reses, vrms, label='uw3')
params, covariance = curve_fit(model, reses, vrms, maxfev = 1000000, bounds = (lowerBounds, upperBounds))
plt.plot(reses, model(reses, *params), label='Fitted Curve', color='red')
a, n, b = params
vrmsRate = n
plt.axhline(b, linestyle = 'dashed', label='fitted asymptote')
plt.axhline(42.864947, label='benchmark')
plt.legend(('uw3','Fitted Curve','Fitted Asymptote', 'Benchmark'))
plt.xlabel("resolution")
plt.ylabel("vrms")
plt.savefig( "vrmsConvergence.png" ,dpi=300)
plt.show()
plt.clf()



lowerBounds = [-100, -10, 0]
upperBounds = [0, 0, 100]



plt.scatter(reses, nu)
params, covariance = curve_fit(model, reses, nu, maxfev = 1000000, bounds = (lowerBounds, upperBounds))
plt.plot(reses, model(reses, *params), label='Fitted Curve', color='red')

a, n, b = params
nuRate = n
plt.axhline(b, linestyle = 'dashed', label='fitted asymptote')

plt.axhline(4.884409, label='benchmark')
plt.legend(('uw3','Fitted Curve','Fitted Asymptote', 'Benchmark'))
plt.xlabel('resolution')
plt.ylabel("Nusselt number")
plt.savefig( "nuConvergence.png" ,dpi=300)
plt.show()
plt.clf()

## this thing should output a nice file that 
"""
Now, lets output all the data into a nice format
"""
outdir = "convergenceResults"
os.makedirs(outdir, exist_ok = True)

df = pd.DataFrame({
    'Resolutions': reses,
    'vrmsValues': vrms,
    'nuValues': nu
})
print(outdir + '/convergenceData.csv')
df.to_csv(outdir + '/convergenceData.csv', index = False)
df = pd.DataFrame({
    'vrmsConvergenceRate': [vrmsRate],
    'nuConvergenceRate': [nuRate]
})

df.to_csv(outdir+'/rateData.csv', index = False)



    






