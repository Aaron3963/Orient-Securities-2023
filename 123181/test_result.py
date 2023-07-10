import pandas as pd

result_path = "results.csv"

result = pd.read_csv(result_path)

# new_row = pd.Series(['TestTest', '0.999999'], index=['Params', 'Correlation_trained'])

# Append the new row to the DataFrame
result.loc[len(result)]= ['TestTest', '0.999999','','','']

# Write the updated DataFrame back to the CSV file
result.to_csv('results.csv', index=False)