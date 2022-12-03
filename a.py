import pandas as pd

df = pd.DataFrame(
    {
    "Name": ["Braund, Mr. Owen Harris",
             "Allen, Mr. William Henry",
             "Bonnell, Miss. Elizabeth"],
    "Sex": ["male", "male", "female"]
    }
)
df1 = pd.DataFrame(
    {
    "Name": ["Braund, Mr. Owen Harris",
             "Allen, Mr. William Henry",
             "Bonnell, Miss. Elizabeth"],
    "Age": [22, 35, 58]
    }
)
df2 = pd.merge(df, df1, left_on='Name', right_on='Name')
df2 = df2[['Age', 'Sex']]
print(df2)