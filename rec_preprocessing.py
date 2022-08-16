import pandas as pd
from mne import io
import numpy as np



number = 29
# edf_fname = '/Users/burek/PycharmProjects/Thesis/EMGs/SUBJ-6-275/SUBJ-6-275_r29_emg.edf'
edf_fname = '/Users/burek/PycharmProjects/Thesis/EMGs/SUBJ-6-275/SUBJ-6-275_r{}_emg.edf'.format(number)
final_name = 'train275_r{}'.format(number)
edf_raw = io.read_raw_edf(edf_fname, preload=True)
hz = 1 / (edf_raw.times[1] - edf_raw.times[0])
print(hz)
hz = int(hz)
print(hz)
# If you wish to get specific channels and time:
# edf_data, times = edf_raw[channels_indices, int(from_t * hz): int(to_t * hz)]


# Or to get all the data:
edf_data, times = edf_raw[:, :]
print(type(edf_data))
print(np.shape(edf_data))
print("EDF_DATA:_________________________________________________")
print(edf_data)
print("__________________________________________________________")

# df = pd.DataFrame(edf_data, columns = ['Hospital_EMG','SensorDot_EMG'])
edf_dataframe = pd.DataFrame()
edf_dataframe['Hospital_EMG'] = edf_data[0]
# edf_dataframe['SensorDot_EMG'] = edf_data[1]



labels = np.zeros(np.size(edf_data[0]), dtype=int)
print(labels)
print(np.size(labels))


# Change labels for active regions
# 10180	10255	Tonic-clonic movements
x = 2201 * hz
y = 2345 * hz
print(x)
print(y)
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1






'''



# 5091	5199	seizure
x = 5091 * hz
y = 5199 * hz
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1


# 7556	7657	seizure
x = 7556 * hz
y = 7657 * hz
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1

# 8799	8895	seizure
x = 8799 * hz
y = 8895 * hz
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1


# 9482	9576	seizure
x = 9482 * hz
y = 9576 * hz
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1

# 9963	10024	seizure
x = 9963 * hz
y = 10024 * hz
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1


# 10501	10594	seizure
x = 10501 * hz
y = 10594 * hz
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1

# 11140	11215	seizure
x = 11140 * hz
y = 11215 * hz
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1

# 11880	11962	seizure
x = 11880 * hz
y = 11962 * hz
x = int(x)
y = int(y)
# Changes zeros labels array from 0 to 1 for active seizure
for i in range(x, y):
  labels[i] = 1
'''








seizure_df = pd.DataFrame()
seizure_df['Sensor Dot'] = edf_data[0]
seizure_df['Seizure label'] = labels

s025_r2 = seizure_df.to_numpy()
print(s025_r2)
print(np.shape(s025_r2))
print("Unique np values:", np.unique(s025_r2[:,1]))

np.save(final_name, s025_r2)




'''
targets = [
'train025_r2.npy',
'train163_r1.npy',
'train177_r4.npy',
'train178_r4.npy',
'train198_r2.npy',
'train203_r8.npy',
'train226_r6.npy',
'train256_r12.npy',
'train256_r14.npy',
'train275_r29.npy',
'train275_r31.npy',
'train276_r5.npy',
'train276_r8.npy',
'train276_r39.npy',
'train291_r15.npy',
'train291_r16.npy',
'train291_r21.npy',
'train291_r23.npy',
'train291_r25.npy',
'train291_r26.npy',
'train307_r1.npy',
'train349_r3.npy',
'train353_r1.npy',
'train353_r6.npy',
'train357_r45.npy',
'train357_r58.npy'
]
'''


'''

'train357_r1.npy',
'train357_r2.npy',
'train357_r3.npy',
'train357_r4.npy',
'train357_r5.npy',
'train357_r6.npy',
'train357_r7.npy',
'train357_r8.npy',
'train357_r9.npy',
'train357_r10.npy',
'train357_r11.npy',
'train357_r12.npy',
'train357_r13.npy',
'train357_r14.npy',
'train357_r15.npy',
'train357_r16.npy',
'train357_r17.npy',
'train357_r18.npy',
'train357_r19.npy',
'train357_r20.npy',
'train357_r21.npy',
'train357_r22.npy',
'train357_r23.npy',
'train357_r24.npy',
'train357_r25.npy',
'train357_r26.npy',
'train357_r27.npy',
'train357_r28.npy',
'train357_r29.npy',
'train357_r30.npy',
'train357_r31.npy',
'train357_r32.npy',
'train357_r33.npy',
'train357_r34.npy',
'train357_r35.npy',
'train357_r36.npy',
'train357_r37.npy',
'train357_r38.npy',
'train357_r39.npy',
'train357_r40.npy',
'train357_r41.npy',
'train357_r42.npy',
'train357_r43.npy',
'train357_r44.npy',
'train357_r45.npy',
'train357_r46.npy',
'train357_r47.npy',
'train357_r48.npy',
'train357_r49.npy',
'train357_r50.npy',
'train357_r51.npy',
'train357_r52.npy',
'train357_r53.npy',
'train357_r54.npy',
'train357_r55.npy',
'train357_r56.npy',
'train357_r57.npy',
'train357_r58.npy',
'train357_r59.npy',
'train357_r60.npy',
'train357_r61.npy',
'train357_r62.npy',
'train357_r63.npy',
'train357_r64.npy',
'train357_r65.npy'
'''






