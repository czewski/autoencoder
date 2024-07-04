## Imports
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

print('-'*15)
print('starting preprocess!')
print('-'*15)

df = pd.read_csv('data/train-item-views.csv',  sep=';')

## Drop timeframe, userId, eventdate
df = df.drop('timeframe', axis=1)
df = df.drop('userId', axis=1)
df = df.drop('eventdate', axis=1)

## Group by sessionId
agg_df = df.groupby('sessionId').agg({
    'itemId': lambda x: list(x)
}).reset_index()
agg_df.head()

## Session length == 6 
def split_rows(row, max_items=6):
    n = len(row['itemId'])
    split_data = []
    for i in range(0, n, max_items):
        split_data.append({
            'sessionId': row['sessionId'],
            'itemId': row['itemId'][i:i+max_items]
        })
    return pd.DataFrame(split_data)

# Apply the split function to rows with more than 6 items
split_df = pd.concat([split_rows(row) if len(row['itemId']) > 6 else pd.DataFrame([row]) for idx, row in agg_df.iterrows()], ignore_index=True)

## Remove sessions of length 1
split_df = split_df[split_df['itemId'].apply(lambda x: len(x) > 1)]

## Drop sessionId?
df_split = split_df.drop('sessionId', axis=1)

## Remap itens (0,num_items)
# Flatten the lists into a single list
all_items = [item for sublist in df_split['itemId'] for item in sublist]
# Find the number of unique items
unique_items = set(all_items)
num_unique_items = len(unique_items)
print(f'Number of unique items: {num_unique_items}')
# Mapping here # Create set from all_items
item_map = {item: idx for idx, item in enumerate(unique_items)}
df_split['itemId'] = df_split['itemId'].apply(lambda x: [item_map[item] for item in x])

all_sessions = df_split.copy()
all_sessions.to_csv('data/diginetica/full.csv', index=False)

## Last itemId as target
df_split['target'] = df_split['itemId'].apply(lambda x: x[-1])

## Remove sessions of length 1
df_split = df_split[df_split['itemId'].apply(lambda x: len(x) > 1)]

## Remove last item from itemId
df_split['itemId'] = df_split['itemId'].apply(lambda x: x[:-1])

# Determine the maximum sequence length
max_len = df_split['itemId'].apply(len).max()

# Define a padding function (0 int)
def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))

# Apply padding to the sequences
df_split['padded_itemId'] = df_split['itemId'].apply(lambda x: pad_sequence(x, max_len))

## train test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_split, test_size=0.2, random_state=42)

# save as vcsv
train.to_csv('data/diginetica/train.csv', index=False)
test.to_csv('data/diginetica/test.csv', index=False)

print('-'*15)
print('done!')
print('-'*15)
