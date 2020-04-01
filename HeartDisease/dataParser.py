import numpy as np
import pandas as pd

# Read file
file = open(r'cleveland.data',  encoding='latin1')
raw_text = file.readlines()
file.close()

# Parse data
lines = [line.split() for line in raw_text]
new_line = []
new_list = []
for i in range(0, len(lines)-1):
   line = lines[i] 
   if line[len(line)-1] == 'name':
       line.remove('name')
       new_line.extend(line)
       new_list.append(new_line)
       new_line = []
   else:    
       new_line.extend(line)
       
heart_data = pd.DataFrame(new_list)

#Drop last rows
heart_data = heart_data.drop(heart_data.iloc[:, 282:293], axis = 0)

# Write out to CSV
heart_data.to_csv('goodData.csv', encoding='utf-8', index=False)