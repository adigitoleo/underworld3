import pickle
import matplotlib.pyplot as plt

# Load the data from the pickle file
with open('results/outputmarkers.pkl', 'rb') as file:
    data = pickle.load(file)

# Assuming the first element is time and the second element is vrms
time_values = data[0]
vrms_values = data[1]

# Create the plot
plt.figure(figsize=(10,6))

# Make the line thinner using the linewidth parameter
plt.plot(time_values, vrms_values, label='vrms', linewidth=0.5)

plt.axhline(y=480, color='r', linestyle='--', label='vrms=480')
plt.xlabel('Time')
plt.ylabel('vrms')
plt.title('vrms vs Time')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
