import pandas as pd
from sklearn.preprocessing import LabelEncoder

#load netflix dataset 
netflix_data = pd.read_csv('netflix_customer_churn.csv')


# ---------------------------
# 1 Create new features
# ---------------------------

# Average watch hours per month (normalize watch_hours by account age in months)
netflix_data['avg_watch_hours_per_month'] = netflix_data['watch_hours'] / ((netflix_data['last_login_days'] + 1)/30)
# Explanation: This tells us roughly how much the customer watches per month. Adding +1 avoids division by zero.

# ---------------------------
#  2 Engagement Score
# ---------------------------
# Multiply average daily watch time by number of profiles to measure overall engagement
netflix_data['engagement_score'] = netflix_data['avg_watch_time_per_day'] * netflix_data['number_of_profiles']

# ---------------------------
# 3 High-Value Customer
# ---------------------------
# Calculate median of monthly fees
median_fee = netflix_data['monthly_fee'].median()
# Create a new column: 1 if customer's monthly fee > median, else 0
netflix_data['high_value_customer'] = netflix_data['monthly_fee'].apply(lambda x: 1 if x > median_fee else 0)

# ---------------------------
# 4 Recent Login
# ---------------------------
# Create a new column: 1 if last login was within 30 days, else 0
netflix_data['recent_login'] = netflix_data['last_login_days'].apply(lambda x: 1 if x <= 30 else 0)

# ---------------------------
#  Encode categorical columns
# ---------------------------

categorical_cols = ['gender', 'subscription_type', 'region', 'device', 'payment_method']

le = LabelEncoder()
for col in categorical_cols:
    netflix_data[col] = le.fit_transform(netflix_data[col])
# Explanation: Converts text categories into numbers so models can understand them.

# Optional: One-Hot Encoding for favorite_genre
netflix_data = pd.get_dummies(netflix_data, columns=['favorite_genre'], drop_first=True)
# Explanation: Converts each genre into a separate column (1 if favorite, 0 if not).

# ---------------------------
# 3 Drop irrelevant column
netflix_data.drop(['customer_id'], axis=1, inplace=True)
# Explanation: customer_id is unique for each row and doesn’t help the model.


print(netflix_data.head())
print(netflix_data.columns)
print(netflix_data.dtypes)
print(netflix_data['churned'].value_counts())

netflix_data.to_csv('Netflix_Feature_Engineered.csv', index=False)   

