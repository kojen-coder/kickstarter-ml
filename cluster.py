
##################################
#### Clustering Model Section ####
##################################
ks_df = pd.read_excel('Kickstarter.xlsx')

ks_df = ks_df[ks_df['state'].isin(['failed', 'successful'])].reset_index(drop=True)
ks_df['category'].fillna('Others', inplace = True)  # Change category "NaN" to "Others"
ks_df['goal_usd'] = ks_df['goal'] * ks_df['static_usd_rate'] # goal convert to usd
df = ks_df
df.loc[df['state']=='successful', 'state'] = 1
df.loc[df['state']=='failed', 'state'] = 0
df['state'] = df['state'].astype(int)

#### Optimized Cohort Feature Engineering ####
## Add Predictors

# -- Add seasonality --
df['launch_season'] = df['launched_at_month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                                                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                                                    9: 'Fall', 10: 'Fall', 11: 'Fall'})
# -- Add weekend flag or not --
df['launched_weekend'] = df['launched_at_weekday'].isin(['Saturday', 'Sunday']).astype(int)
df['deadline_weekend'] = df['deadline_weekday'].isin(['Saturday', 'Sunday']).astype(int)

# -- Text Features to see word density --
df['name_word_density'] = df['name_len'] / df['name_len_clean']
df['blurb_word_density'] = df['blurb_len'] / df['blurb_len_clean']

# -- Country-Specific Indicator --
df['is_us'] = df['country'].apply(lambda x: 1 if x == 'US' else 0)

# -- Hourly Patterns (morning, afternoon, evening) --
df['created_at_time_of_day'] = pd.cut(df['created_at_hr'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
df['launched_at_time_of_day'] = pd.cut(df['launched_at_hr'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
df['deadline_time_of_day'] = pd.cut(df['deadline_hr'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

# -- Holiday period nor not --
# Define a function to create a set of holidays for a range of years
def get_holidays(years, country='US'):
    all_holidays = set()
    for year in years:
        for date in holidays.CountryHoliday(country, years=year).keys():
            all_holidays.add((date.month, date.day))
    return all_holidays
us_holidays = get_holidays(range(2009, 2017), 'US')

# Function to check if a month and day are near a holiday
def is_near_holiday(month, day, holidays, days=7):
    for holiday_month, holiday_day in holidays:
        if abs(pd.Timestamp(2000, month, day) - pd.Timestamp(2000, holiday_month, holiday_day)) <= pd.Timedelta(days=days):
            return 1
    return 0

# Apply the function to your month and day columns
df['launched_near_holiday'] = df.apply(lambda x: is_near_holiday(x['launched_at_month'], x['launched_at_day'], us_holidays), axis=1)
df['deadline_near_holiday'] = df.apply(lambda x: is_near_holiday(x['deadline_month'], x['deadline_day'], us_holidays), axis=1)

df2 = df # df2 will be the optimized cohort

#### Optimized Cohort One Hot Encoding ####
encoder = OneHotEncoder(sparse=False)
encoder_list = ['country', 'currency', 'category', 'deadline_weekday', 'created_at_weekday',
                'launched_at_weekday', 'created_at_time_of_day', 'launched_at_time_of_day',
                 'deadline_time_of_day', 'launch_season']

encoded_data = encoder.fit_transform(df2[encoder_list])
columns = encoder.get_feature_names_out(encoder_list)

one_hot_encoded_df = pd.DataFrame(encoded_data, columns=columns, index=df2.index)
df2_final = pd.concat([df2, one_hot_encoded_df], axis=1)

#### Final Optimized Cohort ####
df2_final.drop(encoder_list, axis=1, inplace=True)

#### Select Clustering Features ####
from sklearn.preprocessing import StandardScaler
clustering_features = ['goal_usd', 'blurb_word_density', 'launch_to_deadline_days',
                       'created_at_hr', 'is_us', 'launched_at_day', 'create_to_launch_days']

df_clustering = pd.concat([df2[clustering_features], one_hot_encoded_df], axis=1)
df_clustering = df2[clustering_features]
scaler = StandardScaler()
X_std = scaler.fit_transform(df_clustering)

#### Use Silhouette Score: Find the Best n_clusters ####
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Fit PCA for dimensionality reduction for visualization purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Define a variable to hold the best silhouette score and the corresponding number of clusters
best_silhouette_avg = -1
best_n_clusters = 0

# Try different numbers of clusters with K-Means
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
    cluster_labels = kmeans.fit_predict(X_std)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_std, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg}")

    # Check if the silhouette score is the best and if so, update the best_n_clusters
    if silhouette_avg > best_silhouette_avg:
        best_silhouette_avg = silhouette_avg
        best_n_clusters = n_clusters

# Confirm the best number of clusters
print(f"Best number of clusters based on silhouette score: {best_n_clusters}")

# Fit KMeans with the best number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(X_std)

# Add the cluster labels to the original dataframe df2
df2['cluster'] = labels

# Visualize the final clusters using PCA components
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='.')
plt.title('K-Means Clustering with PCA Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Print the value counts (sizes) of each cluster
print(f"Cluster Sizes for {best_n_clusters} clusters: \n{df2['cluster'].value_counts()}")

# Display the first few rows of df2 to confirm cluster labels are added
df2.head()


#### Plot Clustering Results Using TSNE ####
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=42)

# Transform the data
X_tsne = tsne.fit_transform(X_std)

# Assuming 'labels' are the cluster labels from your KMeans clustering
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', marker='.')
plt.title('K-Means Clustering with t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

#### Elbow Method - Another Way to Identify Optimial Number ####
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
for i in range(1, 10): # Trying different numbers of clusters
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_std)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show() # It is difficult to find an elbow - hard to identify the optimal number of clusters


#### Groupby Clusters to Gather Insights in Each Cluster ####
# Select only numeric columns for calculating mean and median
numeric_cols = df2.select_dtypes(include=['float64', 'int64'])

# Group by cluster and calculate mean and median for numeric columns
cluster_groups = numeric_cols.groupby(df2['cluster'])

# Display the mean for each feature in each cluster
cluster_means = cluster_groups.mean()
print(cluster_means)

# Display the median for each feature in each cluster
cluster_medians = cluster_groups.median()
print(cluster_medians)

# Visualize the distributions using boxplots for numerical features
import seaborn as sns

numerical_list = ['goal_usd', 'name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean',
       'deadline_month', 'deadline_day', 'deadline_hr', 'created_at_month',
       'created_at_day', 'created_at_hr', 'launched_at_month', 'launched_at_day',
       'launched_at_hr', 'create_to_launch_days', 'launch_to_deadline_days']
label_list = ['state']
id_list = ['id', 'name']
add_numerical_list = ['launched_weekend', 'deadline_weekend', 'name_word_density', 'blurb_word_density',
                      'is_us', 'launched_near_holiday', 'deadline_near_holiday']
future_num_list = ['pledged', 'backers_count', 'deadline_yr', 'created_at_yr', 'state_changed_at_month',
                   'launched_at_yr', 'launch_to_state_change_days']
future_cat_list = ['disable_communication', 'staff_pick', 'spotlight', 'state_changed_at_weekday', ]
category_list = ['country', 'currency', 'category', 'deadline_weekday', 'created_at_weekday',
                'launched_at_weekday', 'created_at_time_of_day', 'launched_at_time_of_day',
                 'deadline_time_of_day', 'launch_season']

for feature in future_num_list:  # numerical feature
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=feature, data=df2)
    plt.title(f'Distribution of {feature} by cluster')
    plt.show()

for feature in future_cat_list:  # categorical feature
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='cluster', data=df2)
    plt.title(f'Distribution of {feature} by cluster')
    plt.show()

for feature in label_list:  # categorical feature - State 1 or 0
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='cluster', data=df2)
    plt.title(f'Distribution of {feature} by cluster')
    plt.show()

for feature in numerical_list:  # numerical feature
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=feature, data=df2)
    plt.title(f'Distribution of {feature} by cluster')
    plt.show()

# Set the palette to a simpler one with distinct colors
palette = sns.color_palette("tab10", n_colors=df2['country'].nunique())

# Create a larger count plot for the 'country' feature
plt.figure(figsize=(20, 10))
sns.countplot(x='category', hue='cluster', data=df2, palette=palette)
plt.title('Distribution of Country by Cluster')
plt.xticks(rotation=90)  # Rotate labels for better readability
plt.show()

# Analyzing the distribution of categorical features within each cluster
for feature in category_list:
    cluster_crosstab = pd.crosstab(df2['cluster'], df2[feature], normalize='index') * 100  # Normalize by index to get percentages
    print(f"Distribution of {feature} by cluster (in %):")
    print(cluster_crosstab)

    # Visualizing the distribution of categorical features within each cluster
    cluster_crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Distribution of {feature} by cluster')
    plt.ylabel('Percentage')
    plt.show()
