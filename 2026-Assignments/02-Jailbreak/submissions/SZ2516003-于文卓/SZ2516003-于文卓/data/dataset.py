import pandas as pd

df = pd.read_csv("harmful_behaviors.csv")

print(df.head())

goals = df["goal"].tolist()

print(goals[:5])