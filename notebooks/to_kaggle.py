import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


cat_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

train = pd.read_parquet("../data/processed/train_withfeatures.parquet")


# VERSION NAME FOR SAVED MODEL FILES
VER = '03'

# TRAIN RANDOM SEED
SEED = 42

# FILL NAN VALUE
NAN_VALUE = -127 # will fit in int8

# FOLDS PER MODEL
FOLDS = 5

# TRAIN FOLD
TRAIN_PATH = "../data/processed/train.parquet"
encoder = LabelEncoder()
for col in cat_cols:
    train[col]= encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])

train = train.fillna(NAN_VALUE)

del train["S_2"]
train.head()

train.target = train.target.astype('int8')

### Competition metric
def amex_metric(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)
### Model training
# FEATURES
FEATURES = train.columns[1:-1]
print(f'There are {len(FEATURES)} features!')
xgb_params = {
    'max_depth':4,
    'booster': 'dart',
    'lambda': 30,
    'alpha': 0.1,
    'learning_rate':0.035,
    'subsample': 0.8,
    'colsample_bytree':0.6,
    'eval_metric': 'logloss',
    'objective':'binary:logistic',
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'random_state':42
}

''''### Version 02 - Kaggle Metric - 0.7466903183087572
* Dropped features from spearman correlation

### Version 03 - Kaggle Metric - 0.7399987922099773
* Drop features from WOE and IV 
* Feature drop with no knowledge about feature meaning is a problem.
''''
### Version 04 - Kaggle Metric - 
#* Force features
importances = []
oof = []
TRAIN_SUBSAMPLE = 1.0

skf = KFold(n_splits = FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train.target)):

    if TRAIN_SUBSAMPLE<1.0:
        np.random.seed(SEED)
        train_idx = np.random.choice(train_idx, 
                       int(len(train_idx)*TRAIN_SUBSAMPLE), replace=False)
        np.random.seed(None)
        
    print('#'*25)
    print('### Fold',fold+1)
    print('### Train size',len(train_idx),'Valid size',len(valid_idx))
    print(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
    print('#'*25)

    X_train = train.loc[train_idx, FEATURES]
    y_train = train.loc[train_idx, 'target']
    X_valid = train.loc[valid_idx, FEATURES]
    y_valid = train.loc[valid_idx, 'target']

    dtrain = xgb.DMatrix(data= X_train, label=y_train)
    dvalid= xgb.DMatrix(data= X_valid, label= y_valid)
    
    model = xgb.train(
                    xgb_params,
                    dtrain=dtrain,
                    evals=[(dtrain, 'train'), (dvalid, 'valid')],
                    num_boost_round= 9999,
                    early_stopping_rounds = 100,
                    verbose_eval= 100
                    )

    model.save_model(f'../models/XGB_V{VER}_fold{fold}.xgb')

    dd = model.get_score(importance_type='weight')
    df= pd.DataFrame({'feature':dd.keys(), f'importance_{fold}':dd.values()})
    importances.append(df)

    oof_preds = model.predict(dvalid)
    acc = amex_metric(y_valid.values, oof_preds)
    print("Kaggle Metric=", acc,'\n')

    df = train.loc[valid_idx, ['customer_ID', 'target']].copy()
    df['oof_pred']= oof_preds
    oof.append(df)

    del dtrain, X_train, y_train, dd, df
    del X_valid, y_valid, dvalid, model

print('#'*25)
oof = pd.concat(oof, axis=0, ignore_index=True).set_index('customer_ID')
acc= amex_metric(oof.target.values, oof.oof_pred.values)
print('OVERAL CV Kaggle Metric = ', acc)

## Import up sound alert dependencies Play after long run
import IPython

