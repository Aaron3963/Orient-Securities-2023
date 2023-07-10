import pandas as pd

raw_data = pd.read_csv('Crypto/raw.csv')

print(raw_data.columns)

ETH_only = raw_data[raw_data['Asset_ID'] == 6]

ETH_only = ETH_only.drop(columns='Asset_ID')

ETH_only['datetime'] = pd.to_datetime(ETH_only['timestamp'], unit='s')
ETH_only = ETH_only.set_index("datetime")

print(ETH_only)

ETH_only = ETH_only.dropna(subset=['Target'])

print(ETH_only)

ETH_only.to_csv('Crypto/ETH.csv')
