with open('.\AllocationsV1.txt') as f:
    lines = f.readlines()
loadFactors =[]
import numpy as np
import matplotlib.pyplot as plt
for line in lines:
    if 'Fitness' in line:
        loadFactors.append(float(line[11:18]))

print(len(loadFactors))
print(np.mean(loadFactors))
print(np.max(loadFactors))
plt.plot(loadFactors)
plt.show()