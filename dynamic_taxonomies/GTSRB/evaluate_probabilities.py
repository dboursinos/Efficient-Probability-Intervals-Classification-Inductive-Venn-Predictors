import numpy as np
import pandas as pd
import sys
import pickle
import matplotlib.pyplot as plt
import csv
import config

sys.setrecursionlimit(40000)
C = config.Config()

with open(C.preprocessed_data, "rb") as f:
    data = pickle.load(f)

with open(C.calibration_probs_path, "rb") as f:
    calibration_probs = pickle.load(f)
with open(C.test_probs_path, "rb") as f:
    test_probs = pickle.load(f)

c_probs=np.max(calibration_probs,axis=1)
c_pred=np.argmax(calibration_probs,axis=1)
t_probs=np.max(test_probs,axis=1)
t_pred=np.argmax(test_probs,axis=1)

print(c_probs)

CE=np.zeros(len(data['y_test'])+1)
EP=np.zeros(len(data['y_test'])+1)

for i in range(len(data['y_test'])):
    CE[i+1]=CE[i]+(t_pred[i]!=data['y_test'][i])
    EP[i+1]=EP[i]+(1-t_probs[i])
print(CE[-1],EP[-1])

downsample_idxs=np.logical_not(np.arange(len(CE))%50)
error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'EP':EP[downsample_idxs]}
df=pd.DataFrame(error_data,columns=['sample', 'CE', 'EP'])
df.to_csv(C.classifier_calibration['data'], index=False, header=True) 

fig = plt.figure(figsize=(7, 7))
plt.plot(CE)
plt.plot(EP)
fig.savefig(C.classifier_calibration['plot'])
