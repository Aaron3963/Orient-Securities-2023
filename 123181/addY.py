import pandas as pd


path = "C:/Users/Aaron/OneDrive/Internships and Jobs/2023 暑假/东方证券/123181/123181.csv"
data = pd.read_csv(path)

newY = []

# for i in range(data.shape[0]-3):
#     newY.append(1 - ((data.at[i+1,"ClosePrice"]+data.at[i+2,"ClosePrice"]+data.at[i+3,"ClosePrice"])/3)
#                  / data.at[i, "ClosePrice"])
# for i in range(3):
#     newY.append(0)


# for i in range(data.shape[0]-1):
#     newY.append(1 - (data.at[i+1,"ClosePrice"]) / data.at[i, "ClosePrice"])

# newY.append(0)

for i in range(data.shape[0]):
    highHalf = abs(data.at[i,"HighestPrice"] - data.at[i,"ClosePrice"])
    lowHalf = abs(data.at[i,"LowestPrice"] - data.at[i,"ClosePrice"])

    if highHalf > lowHalf:
       result = highHalf
    else:
        result = -lowHalf
    newY.append(result)
  
data["HighHalf"] = newY

data.to_csv(path, index=False)