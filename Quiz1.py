import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Quiz1\Heart.csv")

# Print first 5 rows
df.head()

# Count target value
df.target.value_counts()
sns.countplot(x="target", data=df, palette="bwr")
plt.show()

# Counting patient percentage who have heart disease / no disease
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format(
    (countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format(
    (countHaveDisease / (len(df.target))*100)))
sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

# Male and Female Percentage
countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format(
    (countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format(
    (countMale / (len(df.sex))*100)))

# Grouping with Mean
df.groupby('target').mean()

# Showing plot for heart diesease age
pd.crosstab(df.age, df.target).plot(kind="bar", figsize=(20, 6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()

# Showing plot for have or havent have heart disease
pd.crosstab(df.sex, df.target).plot(
    kind="bar", figsize=(15, 6), color=['#1CA53B', '#AA1111'])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()
