## Any feature with strong correlation with target
### But they are highly correlated with each other as we've saw on previous analysis
### So, lets start creating a list of all highly correlated variables
train = pd.read_parquet("../data/processed/train.parquet")
# Create correlation matrix
corr_matrix = train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

# Drop features 
#df.drop(to_drop, axis=1, inplace=True)
## Feature selection with Boruta
x= train.drop(['customer_ID','S_2','target'], axis=1)
y= train.target

x = x.fillna(-127)

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(
    n_jobs = -1,
    max_depth=5
    )
boruta = BorutaPy(
    estimator=forest,
    n_estimators='auto',
    max_iter=100,

)

boruta.fit(x.values, y.values)

## Spearman correlation 
* Same issue as boruta, no computer power available to deal with the problem
features= train.columns.to_list()[2:-1]

X= train[features]
y= train.target
X= X.fillna(-127)


from scipy.stats import spearmanr


df_spearman= train.copy()
df_spearman= df_spearman.fillna(-127)
df_spearman.drop(["customer_ID",'S_2'], inplace=True, axis=1)


import scipy

df = pd.DataFrame()
feat1s=[]
feat2s=[]
corrs=[]
p_values=[]

for feat1 in df_spearman.columns:
    for feat2 in df_spearman.columns:
        if feat1 != feat2:
            feat1s.append(feat1)
            feat2s.append(feat2)
            corr, p_value = spearmanr(df_spearman[feat1], df_spearman[feat2])
            corrs.append(corr)
            p_values.append(p_value) 

df['Feature_1'] = feat1s
df['Feature_2'] = feat2s
df['Correlation'] = corrs
df['p_value'] = p_values
df

df.to_csv("pearson.csv")
## Create dataframe based on the corr matrix
corr_features = pd.read_csv("../reports/high_correlated_features.txt", header=0,names=["features"])
features_to_drop = corr_features.features.to_list()
train_fs = train.drop(labels= features_to_drop, axis= 1)
train_fs.to_parquet("../data/processed/train_fs.parquet")