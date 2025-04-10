# # C:\Users\DEEP NARAYAN\Desktop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading Dataset
df = pd.read_excel(r"C:\Users\DEEP NARAYAN\Desktop\Public_Libraries.xlsx")

#Some basic EDA Process

print(df.head())
print("Shape of dataset:", df.shape)
print(df.info())  

# Handling Missing Values
df.drop(columns=['Registrations Per Capita Served'], inplace=True)

num_cols = [
    'AENGLC Rank',
    'Population of Service Area',
    'Total Registered Borrowers',
    'Percent of Residents with Library Cards',
    'Reference Questions',
    'Reference Questions Per Capita Served',
    'Total Circulation',
    'Circulation Per Capita Served',
    'Total Programs (Synchronous + Prerecorded)',
    'Total Program Attendance & Views Per Capita Served',
    'Total Collection',
    'Collection Per Capita Served',
    'Operating Income Per Capita',
    'Town Tax Appropriation for Library',
    'Tax Appropriation Per Capita Served',
    'Library Materials Expenditures',
    'Wages & Salaries Expenditures',
    'Operating Expenditures',
    'Operating Expenditures Per Capita',
        'Total Library Visits',
    'Library Visits Per Capita Served',
    'Total Program Attendance & Views',
    'Total Operating Income'
]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df['Use of Public Internet Computers'] = df['Use of Public Internet Computers'].fillna(0)
df['Principal Public?'] = df['Principal Public?'].fillna(df['Principal Public?'].mode()[0])

#Checking missing values
missing = df.isnull().sum()
print("Missing Values:\n", missing[missing > 0])


print(df.describe())
 
#Correlation Between Key Numeric Columns
correlation = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


### Objective 1: Analyze Year-wise Growth in Library Visits

df.groupby('Fiscal Year')['Total Library Visits'].sum().plot(kind='line', marker='o')
plt.title('Year-wise Total Library Visits')
plt.ylabel('Total Visits')
plt.xlabel('Fiscal Year')
plt.grid(True)
plt.show()


###Objective 2: Find Libraries with Unusual Program Attendance and Operating Income

def find_unusual(data, column):
    Q1 = data[column].quantile(0.25) 
    Q3 = data[column].quantile(0.75) 
    IQR = Q3 - Q1                  
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_limit) | (data[column] > upper_limit)]
    return outliers

outliers_attendance = find_unusual(df, 'Total Program Attendance & Views')
outliers_income = find_unusual(df, 'Total Operating Income')

print("Libraries with Unusual Program Attendance:")
print(outliers_attendance[['Total Program Attendance & Views']])

print("\nLibraries with Unusual Operating Income:")
print(outliers_income[['Total Operating Income']])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Total Program Attendance & Views'], color='skyblue')
plt.title("Program Attendance (Outliers shown)")
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Total Operating Income'], color='lightgreen')
plt.title("Operating Income (Outliers shown)")
plt.tight_layout()
plt.show()


##Objective 3: Quantify Relationships Between Library Funding, Usage, and Service Output Using Statistical Measures

corr_matrix = df[['Total Operating Income', 'Total Circulation',
                  'Library Visits Per Capita Served', 
                  'Town Tax Appropriation for Library',
                  'Total Programs (Synchronous + Prerecorded)']].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Key Metrics')
plt.show()
cov_matrix = df[['Town Tax Appropriation for Library', 
                 'Total Programs (Synchronous + Prerecorded)',
                 'Total Circulation']].cov()
print("\nCovariance Matrix:")
print(cov_matrix)

print("\nYear-wise Covariance Trends:")
print(df.groupby('Fiscal Year')[['Total Operating Income', 'Total Circulation']].cov())


##Objective 4: Testing Normality and Distribution Fit of Library Visit Rates


# Step 2: Calculate Skewness and Kurtosis

visit_data = df['Library Visits Per Capita Served'].dropna()
skewness = visit_data.skew()
kurtosis = visit_data.kurt()
print("Skewness of Visit Rates:", skewness)
print("Kurtosis of Visit Rates:", kurtosis)
if abs(skewness) < 0.5:
    print("Data is approximately symmetric (low skewness).")
else:
    print("Data is skewed.")

if kurtosis > 0:
    print("Data is leptokurtic (peaked).")
elif kurtosis < 0:
    print("Data is platykurtic (flat).")
else:
    print("Data is mesokurtic (normal-like).")

plt.figure(figsize=(8, 4)) #Visualization - Histogram + KDE
sns.histplot(visit_data, kde=True, color='skyblue')
plt.title('Distribution of Library Visits Per Capita')
plt.xlabel('Visits Per Capita')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


##Objective 5:Investigating the Impact of Program Engagement on Library Usage through Correlation Analysis and Group Comparison
 
correlation = df['Total Program Attendance & Views'].corr(df['Total Library Visits'])
print(f"Pearson Correlation (using pandas): {correlation:.2f}")
if abs(correlation) > 0.5:
    print("Moderate to strong correlation between program attendance and library visits.")
else:
    print("Weak or no significant correlation.")
median_attendance = df['Total Program Attendance & Views'].median()
df['Attendance_Level'] = np.where(df['Total Program Attendance & Views'] >= median_attendance, 'High', 'Low')
group_means = df.groupby('Attendance_Level')['Total Library Visits'].mean()
print("\nAverage Total Library Visits:")
print(group_means)
diff = group_means['High'] - group_means['Low'] # Difference between means (manual check in place of T-test)
print(f"\nDifference in mean visits: {diff:.2f}")
if diff > 0:
    print("Higher attendance group has more visits (indicates positive relationship).")
else:
    print("No clear increase in visits with higher attendance.")
 
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Attendance_Level', y='Total Library Visits', palette='coolwarm')
plt.title("Library Visits vs. Program Attendance Level")
plt.xlabel("Program Attendance Level")
plt.ylabel("Total Library Visits")
plt.tight_layout()
plt.grid(True)
plt.show()

##Objective 6:Understanding the Relationship Between Internet Usage and Program Engagement in Public Libraries

internet_program_corr = df[["Use of Public Internet Computers", "Total Program Attendance & Views"]].corr().iloc[0, 1]
print(f"Correlation Between Internet Usage and Program Attendance: {internet_program_corr:.4f}")
sns.scatterplot(data=df, x="Use of Public Internet Computers", y="Total Program Attendance & Views", alpha=0.6)
plt.title("Internet Usage vs. Program Attendance")
plt.xlabel("Use of Public Internet Computers")
plt.ylabel("Program Attendance & Views")
plt.grid(True)
plt.show()





















 
