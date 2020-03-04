'''
Hepatitis Data set
155 records | 19 attributes + 1 for variable live or die

Project 2
Summarize each attribute
Load and preprocess data
Develop model based on decision tree
Learn exactly how it works and apply
Approximately 100 records for learning and 50 for testing
Look at python decision tree id3

Attribute Information:

1. Class: DIE, LIVE
2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
3. SEX: male, female
4. STEROID: no, yes
5. ANTIVIRALS: no, yes
6. FATIGUE: no, yes
7. MALAISE: no, yes
8. ANOREXIA: no, yes
9. LIVER BIG: no, yes
10. LIVER FIRM: no, yes
11. SPLEEN PALPABLE: no, yes
12. SPIDERS: no, yes
13. ASCITES: no, yes
14. VARICES: no, yes
15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
-- see the note below
16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
17. SGOT: 13, 100, 200, 300, 400, 500,
18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
20. HISTOLOGY: no, yes 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.api.types import is_numeric_dtype
import seaborn as sns

''' READ & PREP DATA '''

col_names = ['class', 'age', 'sex', 'steriod', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 
'spleen_palpable', 'spiders', 'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology']

hepatitis_data = pd.read_csv('hepatitis.data', na_values = ['?'], names = col_names, header = None)

print('Initial dataframe:\n', hepatitis_data)

# Number of NaN values in each column
print('Number of NaN values in each column:\n', hepatitis_data.isnull().sum())

# Print datatypes for each column
print('Datatypes for each column\n', hepatitis_data.dtypes, '\n')


''' STATS '''

# Statistical characteristics for each column
for col in hepatitis_data.columns:
    if is_numeric_dtype(hepatitis_data[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % hepatitis_data[col].mean())
        print('\t Median = %.2f' % hepatitis_data[col].median())
        print('\t Standard deviation = %.2f' % hepatitis_data[col].std())
        print('\t Minimum = %.2f' % hepatitis_data[col].min())
        print('\t Maximum = %.2f' % hepatitis_data[col].max())

# Print number of records in class that are either 1 (died) or 2 (lived)
print('Value counts for class, 1 (died) or 2 (lived):\n', hepatitis_data['class'].value_counts())