import pandas as pd
import dimod
from dimod import quicksum, ConstrainedQuadraticModel, Real, Binary, SampleSet, Integer
from dwave.system import LeapHybridCQMSampler


bin = pd.DataFrame(data = [[5,5,5]], columns=['length', 'width', 'height'])

item_data = []
item_data.append([1,1,1])
item_data.append([2,2,2])
item_data.append([2,2,2])
item = pd.DataFrame(data = item_data, columns=['length', 'width', 'height'])

X_list = []
for x in range(0,bin.iloc[0].length+1):
  X_list.append(x)
X = pd.DataFrame(data = X_list, columns=['X'])

print(X.iloc[1].X)