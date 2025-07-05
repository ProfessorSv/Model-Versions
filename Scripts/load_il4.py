import pandas as pd

# 1. Read the CSV file
data = pd.read_csv(r'C:\Users\celal\Desktop\Model-Versions\data\il4_hdm.csv', sep=',', header=0)

# 2. Print to check it loaded correctly
print(data)

# 3. Extract time and IL-4 arrays
t_data = data['time'].values   # [ 2,  4,  8, 24]
y_data = data['il4'].values    # [50, 90, 55, 20]

# 4. (Optional) basic plot to visualise
import matplotlib.pyplot as plt
plt.plot(t_data, y_data, 'o-')
plt.xlabel('Time (h post-challenge)')
plt.ylabel('IL-4 (pg/ml)')
plt.title('IL-4 Time Course')
plt.show()
