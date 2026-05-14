import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

data_path = "051331512_20200406_Hands_L_9-_L.mp4FT.csv"
data = pd.read_csv(data_path)

data.head()
# Calculate the distance between thumb and index finger for each frame
data['Distance'] = np.sqrt((data['X_thumb'] - data['X_index'])**2 + (data['Y_thumb'] - data['Y_index'])**2)
data['Distance_Norm'] = (data['Distance']-np.min(data['Distance']))/(np.max(data['Distance'])-np.min(data['Distance']))

# Identify peaks in the normalized distance
peaks, _ = find_peaks(
            data['Distance_Norm'], 
            height=0.2,           # 最小高度閾值
            distance=50,          # peaks之間最小距離（保持你原本的設定）
            prominence=0.15,      # peak必須明顯突出，避免平台誤判
            width=1,              # peak的最小寬度
            plateau_size=[1, 40]  # 允許1-40個frame的平台，超過視為同一個peak
        )

# Calculate the periods and amplitudes of taps
tap_periods = np.diff(data['Frame'].iloc[peaks])
tap_amplitudes = data['Distance'].iloc[peaks]
print(tap_amplitudes)

# The number of taps corresponds to the number of detected peaks in the distance between the thumb and index finger.
num_taps = len(peaks)

# Plot the identified taps on the original plot to verify the detection
plt.figure(figsize=(15, 5))
plt.plot(data['Frame'], data['Distance_Norm'], label='Distance between thumb and index finger')
plt.plot(data['Frame'].iloc[peaks], data['Distance_Norm'].iloc[peaks], 'rx', label='Detected Taps')
plt.xlabel('Frame')
plt.ylabel('Distance_Norm')
plt.title('Finger Tapping Test: Detected Taps Over Time')
plt.legend()
plt.show()

# Output number of taps and characteristics
num_taps, tap_periods, tap_amplitudes