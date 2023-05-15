import numpy as np
import pandas as pd


class Woe_Iv:
    def __init__(self, data, target, bins):
        self.data = data
        self.target = target
        self.bins = bins
        self.newDF = pd.DataFrame()
        self.woeDF = pd.DataFrame()
        self.iv_relevance_dict = {
            "not_useful": [],
            "useful": [],
        }
        self.cols = self.data.columns
        self.lst = list()

    def woe_df_iv_df(self):

        for ivars in self.cols[~self.cols.isin([self.target])]:

            if (self.data[ivars].dtype.kind in "bifc") and (
                len(np.unique(self.data[ivars])) > 10
            ):
                binned_x = pd.qcut(self.data[ivars], self.bins, duplicates="drop")
                d0 = pd.DataFrame({"x": binned_x, "y": self.data[self.target]})
            else:
                d0 = pd.DataFrame({"x": self.data[ivars], "y": self.data[self.target]})
            d0 = d0.astype({"x": str})
            d = d0.groupby("x", as_index=False, dropna=False).agg(
                {"y": ["count", "sum"]}
            )
            d.columns = ["Cutoff", "N", "Events"]
            d.insert(loc=0, column="Variable", value=ivars)

            d["% of Events"] = np.maximum(d["Events"], 0.5) / d["Events"].sum()
            d["Non-Events"] = d["N"] - d["Events"]
            d["% of Non-Events"] = (
                np.maximum(d["Non-Events"], 0.5) / d["Non-Events"].sum()
            )
            d["WoE"] = np.log(d["% of Non-Events"] / d["% of Events"])
            d["IV"] = d["WoE"] * (d["% of Non-Events"] - d["% of Events"])

            temp = pd.DataFrame(
                {"Variable": [ivars], "IV": [d["IV"].sum()]}, columns=["Variable", "IV"]
            )
            self.newDF = pd.concat([self.newDF, temp], axis=0)
            self.woeDF = pd.concat([self.woeDF, d], axis=0)

    def create_iv_features(self):

        for i, v in self.newDF.iterrows():
            check = v["IV"]
            if check < 0.02:
                self.iv_relevance_dict["not_useful"].append(v[i])
            elif 0.02 < check < 0.1:
                self.iv_relevance_dict["useful"].append(v[i])
            elif 0.01 <= check < 0.3:
                self.iv_relevance_dict["useful"].append(v[i])
            elif 0.03 <= check < 0.5:
                self.iv_relevance_dict["useful"].append(v[i])
            else:
                self.iv_relevance_dict["not_useful"].append(v[i])

        self.iv_relevance_dict["useful"].append("target")
        # creating a parameter to update train df

    def split_max(self):
        import re

        def split_it(year):
            return pd.Series(re.findall("(\s\d{1,}\.\d{1,})", year))

        def sec_split(year):
            return pd.Series(re.findall("(^[-+]?\d*$)", year))

        self.woeDF["max"] = self.woeDF["Cutoff"].apply(split_it)
        self.woeDF["max"] = pd.to_numeric(self.woeDF["max"])
        self.woeDF["max"] = self.woeDF["max"].replace({"NaN": np.NaN})

        self.woeDF["test"] = self.woeDF["Cutoff"].apply(sec_split)
        self.woeDF["test"] = pd.to_numeric(self.woeDF["test"])
        self.woeDF["test"] = self.woeDF["test"].replace({"NaN": np.NaN})

        self.woeDF["var_max"] = self.woeDF[["max", "test"]].sum(axis=1, min_count=1)
        self.woeDF.drop(columns=["max", "test"], inplace=True)
