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

