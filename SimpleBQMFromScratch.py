import pandas as pd
import dimod
from dimod import quicksum, ConstrainedQuadraticModel, Real, Binary, SampleSet, Integer
from dwave.system import LeapHybridCQMSampler

bin = pd.DataFrame(data = [[2,2,2]], columns=['length', 'width', 'height'])
item_data = []
item_data.append([1,1,1])
item_data.append([1,1,1])
#item_data.append([5,5,5])
item = pd.DataFrame(data = item_data, columns=['length', 'width', 'height'])

total_item_volume = quicksum([item.iloc[i].length * item.iloc[i].width * item.iloc[i].height for i in range(len(item))])

cqm = ConstrainedQuadraticModel()

# Item position
P = {(i,x,y,z): Binary(f'P_{i,x,y,z}') for i in range(len(item)) 
                                       for x in range(1, bin.iloc[0].length+1)
                                       for y in range(1, bin.iloc[0].width+1)
                                       for z in range(1, bin.iloc[0].height+1)} 

# Bin position
T = {(x,y,z):   Binary(f'T_{x,y,z}')   for x in range(1,bin.iloc[0].length+1)
                                       for y in range(1,bin.iloc[0].width+1)
                                       for z in range(1,bin.iloc[0].height+1)}

# Rotation
# not yet for now

# Relative positions ==> cuboid vorm verzekeren
# not yet for now

# Supportive positions ==> stut-constraints in zowel z als x richting
# not yet for now   

# Constraint 1 : each case need to be packed in the bin
for i in range(len(item)): 
  cqm.add_constraint(quicksum([P[i,x,y,z] for x in range(1,bin.iloc[0].length+1) 
                                          for y in range(1,bin.iloc[0].width+1) 
                                          for z in range(1,bin.iloc[0].height+1)]) - item.iloc[i].length * item.iloc[i].width * item.iloc[i].height == 0,
                     label=f'constraint1_{i}')
  
# Constraint 2 : derive T[x,y,z] from P[i,x,y,z]
for x,y,z in [(x,y,z) for x in range(1,bin.iloc[0].length+1) for y in range(1,bin.iloc[0].width+1) for z in range(1,bin.iloc[0].height+1)]:
  cqm.add_constraint(quicksum([P[i,x,y,z] for i in range(len(item))]) - T[x,y,z] == 0,
                     label=f'constraint2_{x}_{y}_{z}')
  
# Constraint 3 : items are not overlapping (via "sum of all dots in T = known sum of dots of all items"
cqm.add_constraint(quicksum([T[x,y,z] for x,y,z in [(x,y,z) for x in range(1,bin.iloc[0].length+1) 
                                                            for y in range(1,bin.iloc[0].width+1) 
                                                            for z in range(1,bin.iloc[0].height+1)]]) 
                   - total_item_volume == 0,
                    label=f'constraint3')


sampler = LeapHybridCQMSampler()
time_limit = 15
res = sampler.sample_cqm(cqm, time_limit=time_limit, label='3d bin packing')

res.resolve()
feasible_sampleset = res.filter(lambda d: d.is_feasible)
print(feasible_sampleset)
try:
    best_feasible = feasible_sampleset.first.sample

    print(best_feasible)
    
except ValueError:
    raise RuntimeError(
        "Sampleset is empty, try increasing time limit or " +
        "adjusting problem config."
    )
