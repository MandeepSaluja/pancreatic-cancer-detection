# Installing/upgrading scikit-learn (assuming it's not already installed)
!pip install --upgrade scikit-learn

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the dataset
df = pd.read_csv('/content/Debernardi et al 2020 data.csv')
df

# Printing unique values in the 'sample_origin' column
unique_values = []
for value in df['sample_origin']:
    if value not in unique_values:
        unique_values.append(value)
print(unique_values)

# Dropping the 'sample_id' column as it seems unnecessary
df.drop(['sample_id'], axis=1, inplace=True)

# Displaying the shape and info of the DataFrame
df.shape
df.info()

# Descriptive statistics of the DataFrame
df.describe()

# Checking for missing values
df.isnull().sum()

# Visualizing the count of diagnosis based on sex
sns.countplot(x='diagnosis', hue='sex', data=df)

# Converting gender to numerical values
def gender(sex):
    if sex == 'M':
        return 0
    elif sex == 'F':
        return 1
    else:
        return -1

df['sex'] = df['sex'].apply(gender)

# Calculating correlation between age and diagnosis
corr1 = df['age'].corr(df['diagnosis'])
print("Correlation coefficient between age and diagnosis:", corr1)

# Converting age into categorical groups
def new_age(age):
    if 25 <= age <= 35:
        return 0
    elif 35 <= age <= 50:
        return 1
    elif 50 <= age <= 90:
        return 2
    else:
        return -1

df['age'] = df['age'].apply(new_age)

# Visualizing the count of diagnosis based on age
sns.countplot(x='diagnosis', hue='age', data=df)

# Visualizing the count of diagnosis based on patient cohort
sns.countplot(x='diagnosis', hue='patient_cohort', data=df)

# Converting patient cohort into numerical values
def cohort(patient_cohort):
    if patient_cohort == 'Cohort1':
        return 0
    elif patient_cohort == 'Cohort2':
        return 1
    else:
        return -1

df['patient_cohort'] = df['patient_cohort'].apply(cohort)

# Visualizing the count of diagnosis based on sample origin
sns.countplot(x='diagnosis', hue='sample_origin', data=df)

# Converting sample origin into numerical values
def origin(sample_origin):
    if sample_origin == 'BPTB':
        return 0
    else:
        return 1

df['sample_origin'] = df['sample_origin'].apply(origin)

# Calculating correlation between plasma_CA19_9 and diagnosis
corr2 = df['plasma_CA19_9'].corr(df['diagnosis'])
corr2

# Calculating correlation between creatinine and diagnosis
corr3 = df['creatinine'].corr(df['diagnosis'])
corr3

# Scatter plot of creatinine against diagnosis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='creatinine', y='diagnosis', data=df)

# Converting creatinine into categorical groups
def cr(creatinine):
    if creatinine <= 0.37320:
        return 0
    elif 0.38 <= creatinine <= 1.139:
        return 1
    elif creatinine > 1.14:
        return 2
    else:
        return -1

sns.countplot(x='diagnosis', hue='creatinine', data=df)

df['creatinine'] = df['creatinine'].apply(cr)

# Calculating correlation between LYVE1 and diagnosis
corr4 = df['LYVE1'].corr(df['diagnosis'])
corr4

# Calculating correlation between REG1B and diagnosis
corr5 = df['REG1B'].corr(df['diagnosis'])
corr5

# Calculating correlation between TFF1 and diagnosis
corr6 = df['TFF1'].corr(df['diagnosis'])
corr6

# Calculating correlation between REG1A and diagnosis
corr7 = df['REG1A'].corr(df['diagnosis'])
corr7

# Dropping unnecessary columns for training the model
df_tr = df.drop(['stage', 'benign_sample_diagnosis', 'plasma_CA19_9', 'REG1A', 'diagnosis'], axis=1)
df_tr

# Checking for missing values in the transformed DataFrame
df_tr.isnull().sum()

# Displaying the 'diagnosis' column
df['diagnosis']
