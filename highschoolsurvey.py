# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:01:29 2020

@author: LabUser
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in data
survey_data = pd.read_csv('HighSchoolSurvey.csv', dtype="str", encoding='cp1252')
print(survey_data.head())
print(survey_data['HousingInsecure'])

#cols_to_drop = std[std==0].index
#print(cols_to_drop)
#heart_data.drop(cols_to_drop, inplace = True, axis=1)
#print(heart_data)

print(survey_data.dtypes)
unique = survey_data['1.How old are you?'].unique()
print(unique)
survey_data['1.How old are you?'] = survey_data['1.How old are you?'].str.replace('14 years old or younger', '14').replace('15 years old', '15').replace('16 years old', '16').replace('17 years old', '17').replace('18 years old or older', '18')
#survey_data['1.How old are you?'] = survey_data['1.How old are you?'].str.replace('15 years old', '15')
#survey_data['1.How old are you?'] = survey_data['1.How old are you?'].str.replace('16 years old', '16')
#survey_data['1.How old are you?'] = survey_data['1.How old are you?'].str.replace('17 years old', '17')
#survey_data['1.How old are you?'] = survey_data['1.How old are you?'].str.replace('18 years old or older', '18')
print(survey_data['1.How old are you?'])