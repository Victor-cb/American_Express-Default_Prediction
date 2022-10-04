def create_bins(data, target, bins=10, show_woe=False, show_iv= False):
    cols = data.columns
    prefix = "_bins"
    for ivars in cols[~cols.isin([target])]:
            
            if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
                binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
                data[ivars] = binned_x
                #d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
            data[ivars] = data[ivars].astype(str)
            # d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
            # d.columns = ['Cutoff', 'N', 'Events']
            
    return data
test= train.copy()
train = create_bins(train, "target")
### Check later

def iv_woe(data, target, bins=10, show_woe=False, show_iv= False, split_max= False):
    import re


    iv_relevance_dict={"not_useful":[],
                       "useful":[],
                      }
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    lst=[]
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d.insert(loc=0, column='Variable', value=ivars)
    
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])

        
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)
        
        #Show IV_values:
        if show_iv:
            print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))

        #Show WOE Table
        if show_woe == True:
            print(d)
        
        
    
    #Creating a list of usefol and not useful features
    for i,v in newDF.iterrows():
        check = v["IV"]
        if check < 0.02:
            iv_relevance_dict["not_useful"].append(v[i])
        elif 0.02 < check < 0.1:
            iv_relevance_dict["useful"].append(v[i])
        elif 0.01 <= check < 0.3:
            iv_relevance_dict["useful"].append(v[i])
        elif 0.03 <= check < 0.5:
            iv_relevance_dict["useful"].append(v[i])
        else:
            iv_relevance_dict["not_useful"].append(v[i])

    iv_relevance_dict["useful"].append("target")
    # creating a parameter to update train df
    if split_max:
        import re
        def split_it(year):
            return pd.Series(re.findall('(\s\d{1,}\.\d{1,})', year))
        def sec_split(year):
            return pd.Series(re.findall('(^[-+]?\d*$)', year))

        woeDF["max"] = woeDF['Cutoff'].apply(split_it)
        woeDF["max"] = pd.to_numeric(woeDF["max"])
        woeDF["max"] = woeDF["max"].replace({"NaN":np.NaN})

        woeDF["test"] = woeDF['Cutoff'].apply(sec_split)
        woeDF["test"] = pd.to_numeric(woeDF["test"])
        woeDF["test"] = woeDF["test"].replace({"NaN":np.NaN})

        woeDF["var_max"]= woeDF[["max", "test"]].sum(axis=1, min_count=1)
        woeDF.drop(columns=["max", "test"], inplace= True)   
    return newDF, woeDF, iv_relevance_dict
   

iv_values, woeDF, iv_relevance_dict = iv_woe(train[feats], 'target', bins=10, show_woe=False)




## Reg Log && Random Forest

reg_log_params={
        'penalty': 'l2',
        'max_iter': 200,
        'warm_start': True,
        'n_jobs': 1,
            
                }
fores_params={
        'bootstrap': True,
        'criterion': 'gini',
        'max_depth': 20,
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.01,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 10,
        'n_jobs': -1,
        'oob_score': False,
        'random_state': 42,
        'verbose': 0,
        'warm_start': False
            }# Setting MLFlow
experiment_name = "RegLog, Forest raw dataset"
try:
    exp_id = mlflow.create_experiment(name=experiment_name)
except Exception as e:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id 
mlflow.autolog()

modelclasses = [
        #["reg_log",LogisticRegression,reg_log_params],
        ["forest", RandomForestClassifier, fores_params]
                ]

# TRAIN RANDOM SEED
SEED = 42

# FILL NAN VALUE
NAN_VALUE = -127 # will fit in int8

# FOLDS PER MODEL
FOLDS = 5


importances = []
oof = []
TRAIN_SUBSAMPLE = 1.0

skf = KFold(n_splits = FOLDS, shuffle=True, random_state=42)
with mlflow.start_run(experiment_id=exp_id):
        for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train.target)):
                

                if TRAIN_SUBSAMPLE<1.0:
                        np.random.seed(SEED)
                        train_idx = np.random.choice(train_idx, 
                                        int(len(train_idx)*TRAIN_SUBSAMPLE), replace=False)
                        np.random.seed(None)
                
                X_train = train.loc[train_idx, FEATURES]
                y_train = train.loc[train_idx, 'target']
                X_valid = train.loc[valid_idx, FEATURES]
                y_valid = train.loc[valid_idx, 'target']

                for modelname, Model, param_list in modelclasses:
                        print('#'*25)
                        print('### Fold',fold+1)
                        print('### Train size',len(train_idx),'Valid size',len(valid_idx))
                        print(f'### Training model {modelname.upper()}')
                        print(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
                        print('#'*25)

                        model = Model(**param_list)
                        #print(model)
                        model.fit(X_train,y_train)

                        pickle.dump(model, open(f'../models/{modelname}_fold{fold}.pkl','wb'))

                        oof_preds = model.predict(X_valid)
                        acc = amex_metric(y_valid.values, oof_preds)
                        print("Kaggle Metric=", acc,'\n')
                        mlflow.log_metric(f"Kaggle Metric for {modelname}", acc)
                        mlflow.sklearn.log_model(model, f"{Model}")
                        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
                        
                        mlflow.sklearn.log_model(model, f"{Model}")
                        df = train.loc[valid_idx, ['customer_ID', 'target']].copy()
                        df['oof_pred']= oof_preds
                        df['model_name'] = modelname
                        oof.append(df)

                        del df, model

                del X_train, y_train
                del X_valid, y_valid

        
print('#'*25)
oof = pd.concat(oof, axis=0, ignore_index=True).set_index('customer_ID')
for n in range(len(modelclasses)):
    target = oof.loc[oof['model_name'] ==modelclasses[n][0], ['target'] ].reset_index()
    preds = oof.loc[oof['model_name'] == modelclasses[n][0], ['oof_pred']].reset_index()
    acc= amex_metric(target.target.values, preds.oof_pred.values)
    print(f'OVERAL CV Kaggle Metric for {modelclasses[n][0]} = {acc}')

# LGBM
# Setting MLFlow
experiment_name = "LightGBM"
try:
    exp_id = mlflow.create_experiment(name=experiment_name)
except Exception as e:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id 
def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True


params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'dart',
    'seed': 42,
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.20,
    'bagging_freq': 10,
    'bagging_fraction': 0.50,
    'n_jobs': -1,
    'lambda_l2': 2,
    'min_data_in_leaf': 40,
    'device_type': 'gpu',
    'max_bin': 64,

    }
# Create a numpy array to store test predictions
#test_predictions = np.zeros(len(test))
# Create a numpy array to store out of folds predictions
oof_predictions = np.zeros(len(train))
skf = KFold(n_splits = 5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train.target)):
    print(' ')
    print('-'*50)
    print(f'Training fold {fold} with {len(FEATURES)} features...')
    x_train, x_val = train[FEATURES].iloc[train_idx], train[FEATURES].iloc[valid_idx]
    y_train, y_val = train['target'].iloc[train_idx], train['target'].iloc[valid_idx]
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_val, y_val)
    model = lgb.train(
        params = params,
        train_set = lgb_train,
        num_boost_round = 10500,
        valid_sets = [lgb_train, lgb_valid],
        early_stopping_rounds = 1500,
        verbose_eval = 500,
        feval = lgb_amex_metric
        )
    # Save best model
    pickle.dump(model, open(f'../models/LGBM_fold{fold}.pkl','wb'))
    # Predict validation
    val_pred = model.predict(x_val)
    # Add to out of folds array
    oof_predictions[valid_idx] = val_pred
    # Predict the test set
    #test_pred = model.predict(test[FEATURES])
    #test_predictions += test_pred / 5
    # Compute fold metric
    score = amex_metric(y_val, val_pred)
    print(f'Our fold {fold} CV score is {score}')
    mlflow.log_metric("Kaggle Metric for LightGbm", acc)
    mlflow.lightgbm.log_model(model, f"{Model}")
    del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
    gc.collect()
# Compute out of folds metric
score = amex_metric(train[target], oof_predictions)
print(f'Our out of folds CV score is {score}')
# Create a dataframe to store out of folds predictions
oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[target], 'prediction': oof_predictions})
oof_df.to_csv(f'/content/drive/MyDrive/Amex/OOF/oof_lgbm_dart_baseline_5fold_seed42.csv', index = False)
# Create a dataframe to store test prediction
# test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
# test_df.to_csv(f'/content/drive/MyDrive/Amex/Predictions/test_lgbm_dart_baseline_fold_5_seed42.csv', index = False)

