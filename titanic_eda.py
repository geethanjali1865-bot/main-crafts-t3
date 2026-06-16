# ==============================
# STEP 1: IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# STEP 2: LOAD DATASET
# ==============================
df = pd.read_csv("titanic.csv")   # rename file to this

print(df.head())

# ==============================
# STEP 3: DATA CLEANING
# ==============================

# Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column
df = df.drop(columns=['Cabin'])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ==============================
# STEP 4: FEATURE ENGINEERING
# ==============================

# Create Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create Age Groups
bins = [0, 12, 20, 40, 60, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle Age', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# ==============================
# STEP 5: ANALYSIS
# ==============================

print("\nSurvival by Age Group:")
print(df.groupby('AgeGroup')['Survived'].mean())

print("\nSurvival by Embarked:")
print(df.groupby('Embarked')['Survived'].mean())

print("\nSurvival by Family Size:")
print(df.groupby('FamilySize')['Survived'].mean())

# ==============================
# STEP 6: VISUALIZATION
# ==============================

# Age Distribution
plt.figure()
sns.histplot(df['Age'], bins=30)
plt.title("Age Distribution")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Survival by Family Size
plt.figure()
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title("Survival by Family Size")
plt.show()

# Survival by Age Group (Bonus)
plt.figure()
sns.barplot(x='AgeGroup', y='Survived', data=df)
plt.title("Survival by Age Group")
plt.show()