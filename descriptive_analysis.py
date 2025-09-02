import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 


#load netflix dataset 
netflix_data = pd.read_csv('netflix_customer_churn.csv')

#Descriptive Statistics 
print("\nSummary Statistics for Numeric Colomns:")
print(netflix_data.describe())

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(netflix_data.corr(numeric_only=True), cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Check churn distribution 
if "churned" in netflix_data.columns:
    plt.figure(figsize=(5,5))
    sns.countplot(x="churned", data=netflix_data)
    plt.title("Churn Distribution")
    plt.show()

#Categorical Analysis---------------------------------------------------------------------------------------------------------------

# Example categorical columns
categorical_cols = ['gender', 'region', 'subscription_type', 'churned']

# Value counts
for col in categorical_cols:
    print(netflix_data[col].value_counts())
    print("\n")

# Bar plot for each categorical variable
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=netflix_data, x=col)
    plt.title(f'Distribution of {col}')
    plt.show()
    
# Pie chart example for one variable
netflix_data['churned'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6,6), colors=['skyblue','lightgreen'])
plt.title('Churn Distribution')
plt.ylabel('')
plt.show()



#Numerical Analysis (Distribution of Numeric Variables)-----------------------------------------------------------------------


 # Example numeric columns
numeric_cols = ['age', 'monthly_fee']


# Histograms
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(netflix_data[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
    plt.show()

# Boxplots (to check outliers)
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(netflix_data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Density plot (smooth distribution)
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.kdeplot(netflix_data[col], fill=True)
    plt.title(f'Density Plot of {col}')
    plt.show()


#Bivariate Analysis (Feature vs Target)-----------------------------------------------------------------------------------

    #Goal: See how each feature relates to your target variable (e.g., Churned).

                  #Categorical vs Target (Churn )
            

    # Count plot with hue
for col in ['gender', 'region', 'subscription_type']:
    plt.figure(figsize=(6,4))
    sns.countplot(data=netflix_data, x=col, hue='churned')
    plt.title(f'{col} vs churned')
    plt.show()

                  #Numerical vs Target (Churned)

    # Boxplots to see distribution of numeric variables by target
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='churned', y=col, data=netflix_data)
    plt.title(f'{col} by churned')
    plt.show()

# Violin plots (optional, more detailed distribution)
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.violinplot(x='churned', y=col, data=netflix_data)
    plt.title(f'{col} by churned')
    plt.show()
            
            
           


    