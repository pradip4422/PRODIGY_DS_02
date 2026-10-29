
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN INFORMATION")
print(df.info())

print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Clean the dataset
df = df.drop_duplicates().copy()

if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

if "Cabin" in df.columns:
    df["Cabin"] = df["Cabin"].fillna("Unknown")

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("outputs/titanic_cleaned.csv", index=False)

# Survival counts
if "Survived" in df.columns:
    print("\nSURVIVAL COUNTS")
    print(df["Survived"].value_counts())

# 1. Survival distribution
if "Survived" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Survived")
    plt.title("Titanic Passenger Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig("outputs/survival_count.png", dpi=300)
    plt.show()

# 2. Survival by gender
if {"Sex", "Survived"}.issubset(df.columns):
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="Sex", y="Survived")
    plt.title("Survival Rate by Gender")
    plt.ylabel("Average Survival Rate")
    plt.tight_layout()
    plt.savefig("outputs/survival_by_gender.png", dpi=300)
    plt.show()

    print("\nSURVIVAL RATE BY GENDER")
    print(df.groupby("Sex")["Survived"].mean().round(3))

# 3. Survival by passenger class
if {"Pclass", "Survived"}.issubset(df.columns):
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="Pclass", y="Survived")
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Average Survival Rate")
    plt.tight_layout()
    plt.savefig("outputs/survival_by_class.png", dpi=300)
    plt.show()

    print("\nSURVIVAL RATE BY CLASS")
    print(df.groupby("Pclass")["Survived"].mean().round(3))

# 4. Age distribution
if "Age" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=20, kde=True)
    plt.title("Distribution of Passenger Ages")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/age_distribution.png", dpi=300)
    plt.show()

# 5. Correlation heatmap
numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=300)
    plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Cleaned data and charts are saved in the outputs folder.")
