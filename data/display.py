import pandas as pd

df = pd.read_csv('train.csv')

while True:
    row = df.iloc[0]
    print(row['Prompt'])
    print(row['Answer'])
    print(row['Target'])
    input()
    df = df[1:]