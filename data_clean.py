import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 


#load netflix dataset 
netflix_data = pd.read_csv('netflix_customer_churn.csv')

#Display basic dataset information 
print("\n Dataset Information:")
print(netflix_data.info())

#Identify missing values in each colums
print("\n Missing values count:")
print(netflix_data.isnull().sum())

#Visualisation missing values using a heatmap
plt.figure(figsize=(10,6))
sns.heatmap(netflix_data.isnull(), cmap="coolwarm", cbar=False)
plt.title("Missing values heatmap in Netflix Dataset")
plt.show() 

netflix_data.to_csv('Netflix_Feature_Engineered.csv', index=False)   




