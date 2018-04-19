import os
import numpy as np
import cv2
import random
import pandas as pd

# Arguments
dir_path = 'predicted_lanes/.'

numSamples = 500
random.seed(1)

randNums = np.random.choice(np.arange(6,len(np.sort(os.listdir(dir_path)))), 500, replace = False)

for i in randNums:
    currImg = cv2.imread('predicted_lanes/img'+str(i).zfill(4)+'_lanes.jpg')
    cv2.imwrite('sample_lanes/img'+str(i).zfill(4)+'_lanes.jpg', currImg)
    
# create pandas df to save
df = pd.DataFrame(index=np.sort(randNums), columns = ['TP','FP','FN'])
df.to_csv('accuracyResults.csv',sep = ',')