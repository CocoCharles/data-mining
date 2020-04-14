# -*- coding: utf-8 -*-
"""
Preprocessing data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn import tree
from sklearn.svm import SVC
import pydotplus


# Read in data
data_original = pd.read_csv('DCYA2018.csv', na_values=[' '])
# print('First 5 rows of initial data:\n', data.head(), '\n')

# Drop columns
data = data_original.drop(data_original.iloc[:, 266:316], axis = 1)

# Drop open ended question (string answers)
# 'DontFitIn', 'Rules', 'CloseToPeople', 'FeelSafe', 'TreatedFairly', 'AdultsToTalkTo', 'IBelong', 'DateSexRecode'
data = data.drop(['Year', 'Schoolcode', 'RespondentID', 'Weighting2', 'OtherProblems', 'Homeless2015', 'HeightInch', 'HeightFeet', 'Weight'], axis=1)

print(data.shape)
# print('First 5 rows of cleaned** data:\n', data.head(), '\n')
# print('Numer of instances = %d' %data.shape[0])
# print('Numer of attributes = %d' %data.shape[1])

# Add values missing from NoIntercourse
data.loc[(data['Abstinence'] == 15), 'NoIntercourse'] = 15
print('\nNumber of each value in NoIntercourse:\n', data['NoIntercourse'].value_counts())

# Recode columns with binary values to 0(no) and 1(yes)
for col in data.columns:
    if 0 in data[col].unique():
        data.loc[(data[col] > 0), col] = 1


# Change types from 64 to 32
for col in data.columns:
    if data.dtypes[col] == 'int64':
        data[col] = data[col].astype('int32')
    if data.dtypes[col] == 'float64':
        data[col] = data[col].astype('float32')


# Add a column that converts 'IBelong' to binary
# With 1 being agree and 2 being disagree
belong_map = {1.0:1,
              2.0:1,
              3.0:2,
              4.0:2}
                 
data['belong'] = data['IBelong'].map(belong_map)
# data['belong'] = data.belong.astype('float32')
print('\nNumber of each value in belong:\n', data['belong'].value_counts())

# Recode attributes of the 'protected sex' question to binary 0 and 1
sex_protection_map = {1.0:1, 2.0:1, 3.0:1, 4.0:1, 5.0:1, 6.0:1, 7.0:1, 8.0:1, 9.0:1, 10.0:1, 11.0:1, 12.0:1, 15.0:0}
for col in data.columns:
    if 15 in data[col].unique():
        data[col] = data[col].map(sex_protection_map)
print('\nNumber of each value in IUD:\n', data['IUD'].value_counts())

print('\nCleaned data with NaNs:\n', data)




# Imputing missing values using K Nearest Neighbor Imputer

# Column names
data_columns = ['Age','Grade','Race','Sex','SexualOrientation','GenderID','Transgender','LivingSituation','RunAway','FamHomeless','Homeless','FosterCare',
'KickedOut','ChangedSchools','FamFinacial','Employment','SuppoprtFam','Volunteerting','Sports','HomeWorkClub','DramaMusic','StudentLeadership','Religious',
'AfterSchoolClubs','PhysicallyActive','Exercise','NoLimits','NoTime','DontLike','HealthProbs','NoSkills','NoProgram','NoSignUp','CostsTooMuch','NoTransport',
'Other','Sleep','SleepPhone','DriveText','DoAboutWeight','FruitsVeggies','FamilyMeals','Breakfast','SkipMealsNo$$','SugaryDrink','CaffeineDrink',
'DisabilityEducation','DisabilityPhysical','DentistCheckup','DentalPain','WeatSeatBelt','GunAccess','ParentMentalHealth','ParentJail','ParentHit',
'ParentDrunk','ParentPrescptDrugs','ParentsFight','ParentsKnowWhereIAm','ParentsRules','ParentsFuturePlans','ParentsConsquences','ParentsEncourage',
'ParentsMonitorSchool','ParentsTalkProblems','FeltAnxious','CantStopWorry','PilingUp','TooMuchWorry','CantRelax','Restless','Annoyed','Afraid', 
'NoAnxiety', 'PressureThings', 'TooMuchHomework', 'SocialPressure', 'SchoolPressure', 'DontFitIn', 'CollegePressure', 'HomeProblems', 'WorldProblems', 
'Deportation', 'OKtoGetMentalHealthCare', 'NotMakeBigDeal', 'TendtoObsess', 'CareINeed', 'DontBlockFeelings', 'Shortcomings', 'LackTolerance', 
'NoEmotionalHealthProbs', 'Depression2', 'Anxiety2', 'EatingDis', 'ADHD', 'OtherEmotionalHealthProb', 'Depression', 'MentalHealthServices', 'SelfHarm', 
'SuicideIdeation', 'Suicide', 'SexualContact', 'ReasonsNoSex', 'SexualIntercourse', 'AgeIntercourse', 'NumberSexPartners', 'SexUnderInfluence', 
'SexPartnerGender', 'BirthControl', 'NoIntercourse', 'Abstinence', 'Pills', 'Implants', 'Patch', 'NuvaRing', 'Condom', 'Shots', 'IUD', 'Withdrawal', 
'OtherMethod', 'Nothing', 'UseCondomProtection', 'WhereToGetBirthcontrol', 'WhereToGetSTITest', 'EverTestSTI', 'WhyNeverTested', 'HowManyHaveSex', 
'ParentTalkSex', 'ParentTalkDating', 'ParentTalkBirthControl', 'ParentTalkSTI', 'Vaping', 'Cigarettes', 'RollYourOwn', 'Flavored', 'Clove', 
'LittleCigars', 'Chewing', 'Hookah', 'TobaccoSource', 'Alcohol12Mo', 'ReasonsNoDrink', 'Alcohol30days', 'Binge30days', 'AlcoholDrive', 'AlcoholSource', 
'ParentsKnow', 'ParentsProvide', 'Marijuana12Mo', 'Marijuana30days', 'MarijuanaDriving', 'PrescriptDrugs', 'OverCounterDrugs', 'Cocaine', 'Inhale', 
'SpeedMeth', 'Heroin', 'Ecstasy', 'BathSalts', 'K2', 'Steroids', 'PresriptsHome', 'WherePrescriptDrugs', 'AODtoRelax', 'AODDWI', 'AODAlone', 'AODForget', 
'AODFriendsSTop', 'AODinTrouble', 'RiskPackofCigs', 'RiskBingeDrinking', 'RiskMarijuana', 'RiskPrescriptDrugs', 'RiskEcigs', 'ParentsWrongMarijunana', 
'ParentsWrongTobacco', 'ParentsWrongAlcohol', 'ParentsWrongPrescrptDrugs', 'ParentsWrongAlcoholDaily', 'YourAgeAlcohol', 'YourAgeMarijuana', 
'YourAgeTobacco', 'YourAgePrescrptDrugs', 'FriendsWrongAlcohol', 'FriendsWrongTobacco', 'FriendsWrongMarijuana', 'FriendsWrongPrescriptDrugs', 
'FriendsWrongDailyAlcohol', 'GradesInSchool', 'FinishHighSchool', 'GoToTechSchool', 'GoToCollege', 'LookForJob', 'JoinMilitary', 'CareerILike', 
'IEP', 'FreeReduced', 'NeedSchoolMeals', 'Rules', 'CloseToPeople', 'FeelSafe', 'TreatedFairly', 'AdultsToTalkTo', 'IBelong', 'StruggleHomeWork', 
'TooMuchHomeWork2', 'HomeWorkTooHard', 'TooMuchToDo', 'CantGetHelp', 'TooTired', 'APClasses', 'CutClass', 'InSchoolSuspension', 'OutOfSchoolSuspension', 
'SeenAtSchoolDrugUse', 'SeenAtSchoolSellingDrugs', 'SeenAtSchoolWeapons', 'SeenAtSchoolGangs', 'SeenAtSchoolBullying', 'SeenAtSchoolNameCalling', 
'SeenAtSchoolStudentPutDowns', 'InaGang', 'FriendsInGang', 'HeardSexualComments', 'GotHitPushed', 'MadeFunofOthers', 'GotPickedOn', 'UpsetOthers', 
'StartedArguments', 'SpreadRumors', 'ToldSomeoneToStop', 'MadeFunofMe', 'ExcludedOthers', 'GotCalledNames', 'HelpedHarass', 'BulliedHarassedElectronic', 
'BulliedHarassedGLBT', 'BulliedHarassedRace', 'BulliedHarassedImmigrant', 'BulliedHarassedPolitics', 'BulliedHarassedLooks', 'DatingViolence', 
'DatingPutDowns', 'DatingControl', 'DatingThreats', 'DatingForcedSex', 'DatingNoControlViolence', 'PerpDatingViolence', 'ForcedSex', 'SexTrafficking',
'AskNeighborsForHelp', 'WouldMissNeighborhood', 'SafeInNeighborhood', 'NeighborsFriendly', 'CountOnPolice', 'HelpMyNeighbors', 'AdultsIRelyOn', 'belong']



# Split first 1/16 (1018 records)
remaining, current = train_test_split(data, test_size = 1018)
print('Remaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)
print(fill_data)

# Round to nearest integer
fill_data = np.round(fill_data)
print(fill_data)

# Convert to dataframe
data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')
print(data)


# Split 2
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 3
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 4
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 5
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 6
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 7
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 8
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 9
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 10
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 11
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 12
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 13
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 14
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 15
remaining, current = train_test_split(remaining, test_size = 1018)
print('\nRemaining shape: ', remaining.shape, '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)


# Split 16
current = remaining
print('\nRemaining shape: 0', '\nCurrent shape: ', current.shape)

# Imputation of missing values (returns array)
imputer = KNNImputer(n_neighbors=31)
fill_data = imputer.fit_transform(current)

# Round to nearest integer
fill_data = np.round(fill_data)

# Convert to dataframe
add_data = pd.DataFrame(fill_data, columns=data_columns, dtype='int32')

# Concatenate frames
data = pd.concat([data, add_data], ignore_index=True)

print(data)
data.to_csv('DCYApreprocessed.csv')
print('isna ', data.isna().sum())
