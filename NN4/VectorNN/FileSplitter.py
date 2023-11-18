import pandas as pd

substring = "Test"
data = pd.read_csv('./ASL_Alph_Vectorized.csv')
data["ImagePath"] = data["ImagePath"].fillna("")

filteredData = data[data["ImagePath"].str.contains(substring, case=False)]
filteredData.to_csv('./ASL_Alph_Test.csv', index=False)
