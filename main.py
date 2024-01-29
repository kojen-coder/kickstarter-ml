## Developing the model ###

# Load Libraries
import pandas as pd

import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import warnings
import holidays


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


######################################
#### Classification Model Section ####
######################################

# Functions
def get_holidays(years, country='US'):
    all_holidays = set()
    for year in years:
        for date in holidays.CountryHoliday(country, years=year).keys():
            all_holidays.add((date.month, date.day))
    return all_holidays

def is_near_holiday(month, day, holidays, days=7):
    for holiday_month, holiday_day in holidays:
        if abs(pd.Timestamp(2000, month, day) - pd.Timestamp(2000, holiday_month, holiday_day)) <= pd.Timedelta(days=days):
            return 1
    return 0

def preprocessing(input_df, input_var):
    input_df = input_df[input_df[label].isin(['failed', 'successful'])].reset_index(drop=True)
    input_df.loc[input_df['state']=='successful', 'state'] = 1
    input_df.loc[input_df['state']=='failed', 'state'] = 0
    input_df['state'] = input_df['state'].astype(int)
    input_df['goal_usd'] = input_df['goal'] * input_df['static_usd_rate']
    df = input_df[input_var].reset_index(drop=True)
    df['category'].fillna('Others', inplace = True)

    df['launch_season'] = df['launched_at_month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                                        3: 'Spring', 4: 'Spring', 5: 'Spring',
                                                        6: 'Summer', 7: 'Summer', 8: 'Summer',
                                                        9: 'Fall', 10: 'Fall', 11: 'Fall'})
    df['launched_weekend'] = df['launched_at_weekday'].isin(['Saturday', 'Sunday']).astype(int)
    df['deadline_weekend'] = df['deadline_weekday'].isin(['Saturday', 'Sunday']).astype(int)
    df['name_word_density'] = df['name_len'] / df['name_len_clean']
    df['blurb_word_density'] = df['blurb_len'] / df['blurb_len_clean']
    df['is_us'] = df['country'].apply(lambda x: 1 if x == 'US' else 0)
    df['created_at_time_of_day'] = pd.cut(df['created_at_hr'], bins=[0, 6, 12, 18, 24], 
                                          labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df['launched_at_time_of_day'] = pd.cut(df['launched_at_hr'], bins=[0, 6, 12, 18, 24], 
                                           labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df['deadline_time_of_day'] = pd.cut(df['deadline_hr'], bins=[0, 6, 12, 18, 24], 
                                        labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    us_holidays = get_holidays(range(2009, 2017), 'US')
    df['launched_near_holiday'] = df.apply(lambda x: is_near_holiday(x['launched_at_month'], x['launched_at_day'], us_holidays), axis=1)
    df['deadline_near_holiday'] = df.apply(lambda x: is_near_holiday(x['deadline_month'], x['deadline_day'], us_holidays), axis=1)
    return df

from sklearn.preprocessing import OneHotEncoder

class DataPreprocessor:
    def __init__(self, df, num_list, cat_list):
        self.df = df
        self.num_list = num_list
        self.cat_list = cat_list
    
    def imputation(self, method_list, df=None, num_list=None):
        if not df:
            df = self.df
        if not num_list:
            num_list = self.num_list
        imputation_map = {}
        for num_col, method in zip(num_list, method_list):
            impute_val = self.col_imputation(df, num_col, method)
            imputation_map[num_col] = impute_val
            df[num_col] = df[num_col].astype(float)
            df.loc[df[num_col].isnull(), num_col] = impute_val
        self.df = df
        self.imputation_map = imputation_map
    
    def col_imputation(self, df, col_name, method):
        if method == 'mean':
            val = df[col_name].mean()
        elif method == 'median':
            val = df[col_name].median()
        return val
    
    def cat_encoding(self, df=None, cat_list=None):
        if not df:
            df = self.df
        if not cat_list:
            cat_list = self.cat_list
        for name in cat_list:
            df.loc[:,name] = df[name].astype(str) 
            df.loc[df[name].isnull(), name] = 'null'
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[cat_list]).toarray()
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        self.encoder = encoder
        self.encode_df = pd.concat([df.drop(cat_list, axis=1), encoded_df], axis=1)
        return self.encode_df

    def apply_imputation(self, df):
        for name in self.num_list:
            val = self.imputation_map[name]
            df[name] = df[name].astype(float).fillna(val)
        return df

    def apply_one_hot_encoding(self, df):
        temp = self.encoder.transform(df[self.cat_list]).toarray()
        encoded_df = pd.DataFrame(temp, columns=self.encoder.get_feature_names_out())
        encode_df = pd.concat([df.drop(cat_list, axis=1), encoded_df], axis=1)
        return encode_df


# Declare Variables
input_var = ['state','goal_usd', 'country', 'currency', 'category', 
       'name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean', 
       'deadline_weekday', 'created_at_weekday', 'launched_at_weekday', 
       'deadline_month', 'deadline_day', 'deadline_hr', 'created_at_month', 
       'created_at_day', 'created_at_hr', 'launched_at_month', 'launched_at_day', 
       'launched_at_hr', 'create_to_launch_days', 'launch_to_deadline_days']

numerical_list = ['goal_usd', 'name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean', 'deadline_month', 'deadline_day', 
                  'deadline_hr', 'created_at_month', 'created_at_day', 'created_at_hr', 'launched_at_month', 'launched_at_day', 
                  'launched_at_hr', 'create_to_launch_days', 'launch_to_deadline_days', 'launched_weekend', 'deadline_weekend',
                  'name_word_density', 'blurb_word_density', 'is_us', 'launched_near_holiday', 'deadline_near_holiday']
imputation_method = ["mean" for i in numerical_list]
cat_list = ['country', 'currency', 'category', 'deadline_weekday', 'created_at_weekday', 
                'launched_at_weekday', 'created_at_time_of_day', 'launched_at_time_of_day', 
                 'deadline_time_of_day', 'launch_season']
label = 'state'



# Import data
ks_df = pd.read_excel('Kickstarter.xlsx')

# Pre-Processing
pre_ks = preprocessing(ks_df, input_var)
dp = DataPreprocessor(pre_ks, numerical_list, cat_list)
dp.imputation(imputation_method)
encode_df = dp.cat_encoding().reset_index()

# setup variable & spliting
model_var = [i for i in encode_df.columns if i not in ['state', 'index']]
train_df = encode_df.sample(frac=0.8).reset_index(drop=True)
test_df = encode_df[~encode_df["index"].isin(train_df['index'])].reset_index(drop=True)
train_x, test_x  = train_df[model_var], test_df[model_var]
train_y, test_y  = train_df['state'], test_df['state']

from sklearn.ensemble import GradientBoostingClassifier
# Define the parameter grid
param_grid = {
    'n_estimators': [200, 300],  # Number of boosting stages to be run
    'learning_rate': [0.01, 0.07],  # Shrinks the contribution of each tree by learning_rate
    'max_depth': [4, 5],  # Maximum depth of the individual regression estimators
    'min_samples_split': [4, 6],  # The minimum number of samples required to split an internal node
    'min_samples_leaf': [2, 3],  # The minimum number of samples required to be at a leaf node
    'subsample': [0.8, 1.0],  # The fraction of samples to be used for fitting the individual base learners
    'max_features': [None, 'sqrt', 'log2']  # The number of features to consider when looking for the best split
}

# Initialize the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
cv = GridSearchCV(estimator=gbc, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
cv.fit(train_x, train_y)

# Using the model to predict the results based on the test dataset
test_pred = cv.predict(test_x)

# Calculate the mean squared error of the prediction
from sklearn.metrics import accuracy_score
print(accuracy_score(test_y, test_pred))


## Grading ##

# Import Grading Data
print("model inference")
ks_df2 = pd.read_excel('Kickstarter-Grading.xlsx')
# ks_df2 = pd.read_excel('Kickstarter-Grading-Sample.xlsx')


# Pre-Process Grading Data
pre_ks2 = preprocessing(ks_df2, input_var)
impute_ks2 = dp.apply_imputation(pre_ks2)
encode_ks2 = dp.apply_one_hot_encoding(impute_ks2).reset_index()


# Apply the model previously trained to the grading data
test_encode_ks2 = encode_ks2[model_var]
test_encode_ks2_pred = cv.predict(test_encode_ks2)
test_encode_ks2 = encode_ks2[model_var]

# Calculate the accuracy score
print(accuracy_score(encode_ks2[label], test_encode_ks2_pred))
