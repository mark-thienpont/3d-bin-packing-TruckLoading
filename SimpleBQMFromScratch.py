import pandas as pd
import dimod
from dimod import quicksum, ConstrainedQuadraticModel, Real, Binary, SampleSet, Integer
from dwave.system import LeapHybridCQMSampler

from utils import print_cqm_stats

bin = pd.DataFrame(data = [[3,3,3]], columns=['length', 'width', 'height'])
item_data = []
item_data.append([1,1,1])
item_data.append([2,2,2])
#item_data.append([3,3,3])
item = pd.DataFrame(data = item_data, columns=['length', 'width', 'height'])

total_item_volume = quicksum([item.iloc[i].length * item.iloc[i].width * item.iloc[i].height for i in range(len(item))])

cqm = ConstrainedQuadraticModel()

# Item position
P = {(i,x,y,z): Binary(f'P_{i,x,y,z}') for i in range(len(item)) 
                                       for x in range(0, bin.iloc[0].length+2)
                                       for y in range(0, bin.iloc[0].width+2)
                                       for z in range(0, bin.iloc[0].height+2)} 

# Bin position
T = {(x,y,z):   Binary(f'T_{x,y,z}')   for x in range(0,bin.iloc[0].length+1)
                                       for y in range(0,bin.iloc[0].width+1)
                                       for z in range(0,bin.iloc[0].height+1)}

# Rotation
# not yet for now

# Relative positions ==> ensuring cuboid form of items via constraints
# ... remark : the ranges start 1 level more 'below' and 'above' the bin-definition, to allow for 1 constraint to cover
# ...          situations in which items are touching the boundary of the bin also
NEx = {(i,x,y,z): Binary(f'NEx_{i,x,y,z}') for i in range(len(item)) 
                                         for x in range(1,bin.iloc[0].length+2)
                                         for y in range(1,bin.iloc[0].width+2)
                                         for z in range(1,bin.iloc[0].height+2)} 
NEy = {(i,x,y,z): Binary(f'NEy_{i,x,y,z}') for i in range(len(item)) 
                                         for x in range(1,bin.iloc[0].length+2)
                                         for y in range(1,bin.iloc[0].width+2)
                                         for z in range(1,bin.iloc[0].height+2)} 
NEz = {(i,x,y,z): Binary(f'NEz_{i,x,y,z}') for i in range(len(item)) 
                                         for x in range(1,bin.iloc[0].length+2)
                                         for y in range(1,bin.iloc[0].width+2)
                                         for z in range(1,bin.iloc[0].height+2)} 

## Stutted in z-direction
#Sz = {(i): Integer(f'Sz_{i}') for i in range(len(item))} 

# Constraint 1a : each case need to be packed in the bin
for i in range(len(item)): 
  cqm.add_constraint(quicksum([P[i,x,y,z] for x in range(1,bin.iloc[0].length+1) 
                                          for y in range(1,bin.iloc[0].width+1) 
                                          for z in range(1,bin.iloc[0].height+1)]) - item.iloc[i].length * item.iloc[i].width * item.iloc[i].height == 0,
                     label=f'constraint1a_{i}')

# Constraint 1b : each case need to be packed in the bin
# ... added constraint to enable outer boundaries, without possibility to have 'dots' there
for i in range(len(item)): 
  cqm.add_constraint(quicksum([P[i,x,y,z] for x in range(0, bin.iloc[0].length+2, bin.iloc[0].length+1)
                                          for y in range(bin.iloc[0].width+2) 
                                          for z in range(bin.iloc[0].height+2)]) == 0,
                     label=f'constraint1b_x_{i}')
  cqm.add_constraint(quicksum([P[i,x,y,z] for x in range(bin.iloc[0].length+2)
                                          for y in range(0, bin.iloc[0].width+2, bin.iloc[0].width+1) 
                                          for z in range(bin.iloc[0].height+2)]) == 0,
                     label=f'constraint1b_y_{i}')  
  cqm.add_constraint(quicksum([P[i,x,y,z] for x in range(bin.iloc[0].length+2)
                                          for y in range(bin.iloc[0].width+2) 
                                          for z in range(0, bin.iloc[0].height+2, bin.iloc[0].height+1)]) == 0,
                     label=f'constraint1b_z_{i}')    

# Constraint 2a : derive T[x,y,z] from P[i,x,y,z]
for x,y,z in [(x,y,z) for x in range(1,bin.iloc[0].length+1) for y in range(1,bin.iloc[0].width+1) for z in range(1,bin.iloc[0].height+1)]:
  cqm.add_constraint(quicksum([P[i,x,y,z] for i in range(len(item))]) - T[x,y,z] == 0,
                     label=f'constraint2a_{x}_{y}_{z}')

# Constraint 2b : 0-line added to enable stutting-constraints
cqm.add_constraint(quicksum([T[0,y,z] for y in range(1,bin.iloc[0].width+1) 
                                      for z in range(1,bin.iloc[0].height+1)] ) == (bin.iloc[0].width)*(bin.iloc[0].height),
                    label=f'constraint2b_x')
cqm.add_constraint(quicksum([T[x,0,z] for x in range(bin.iloc[0].length+1) 
                                      for z in range(bin.iloc[0].height+1)] ) == 0,
                    label=f'constraint2b_y')
cqm.add_constraint(quicksum([T[x,y,0] for x in range(1,bin.iloc[0].length+1) 
                                      for y in range(1,bin.iloc[0].width+1)] ) == (bin.iloc[0].width)*(bin.iloc[0].length),
                    label=f'constraint2b_z')    

# Constraint 3 : items are not overlapping (via "sum of all dots in T = known sum of dots of all items"
cqm.add_constraint(quicksum([T[x,y,z] for x,y,z in [(x,y,z) for x in range(1,bin.iloc[0].length+1) 
                                                            for y in range(1,bin.iloc[0].width+1) 
                                                            for z in range(1,bin.iloc[0].height+1)]]) 
                   - total_item_volume == 0,
                    label=f'constraint3')

# Constraint 4 : derive relative positions NEx, NEy and NEz from P[i,x,y,z]
### This assumes below equation behaves like a 'logical OR', i.e. derives 0 or 1 for NEx
for i,x,y,z in [(i,x,y,z) for i in range(len(item)) for x in range(1,bin.iloc[0].length+2) for y in range(1,bin.iloc[0].width+2) for z in range(1,bin.iloc[0].height+2)]:
  cqm.add_constraint((NEx[i,x,y,z])*(P[i,x,y,z] + P[i,x-1,y,z] - 1) == 0 ,
                     label=f'constraint4a_NEx_{i}_{x}_{y}_{z}')
  cqm.add_constraint((1-NEx[i,x,y,z])*(P[i,x,y,z] - P[i,x-1,y,z]) == 0 ,
                     label=f'constraint4b_NEx_{i}_{x}_{y}_{z}')  
for i,x,y,z in [(i,x,y,z) for i in range(len(item)) for x in range(1,bin.iloc[0].length+2) for y in range(1,bin.iloc[0].width+2) for z in range(1,bin.iloc[0].height+2)]:
  cqm.add_constraint((NEy[i,x,y,z])*(P[i,x,y,z] + P[i,x,y-1,z] - 1) == 0 ,
                     label=f'constraint4a_NEy_{i}_{x}_{y}_{z}')
  cqm.add_constraint((1-NEy[i,x,y,z])*(P[i,x,y,z] - P[i,x,y-1,z]) == 0 ,
                     label=f'constraint4b_NEy_{i}_{x}_{y}_{z}')   
for i,x,y,z in [(i,x,y,z) for i in range(len(item)) for x in range(1,bin.iloc[0].length+2) for y in range(1,bin.iloc[0].width+2) for z in range(1,bin.iloc[0].height+2)]:
  cqm.add_constraint((NEz[i,x,y,z])*(P[i,x,y,z] + P[i,x,y,z-1] - 1) == 0 ,
                     label=f'constraint4a_NEz_{i}_{x}_{y}_{z}')
  cqm.add_constraint((1-NEz[i,x,y,z])*(P[i,x,y,z] - P[i,x,y,z-1]) == 0 ,
                     label=f'constraint4b_NEz_{i}_{x}_{y}_{z}')  
  
### Constraint 5 : assure cuboid form with right dimensions for all items
for i in range(len(item)): 
  cqm.add_constraint(quicksum([NEx[i,x,y,z] for x in range(1,bin.iloc[0].length+2)
                                            for y in range(1,bin.iloc[0].width+2) 
                                            for z in range(1,bin.iloc[0].height+2)]) == 2*item.iloc[i].width*item.iloc[i].height,
                     label=f'constraint5_x_{i}')
  cqm.add_constraint(quicksum([NEy[i,x,y,z] for x in range(1,bin.iloc[0].length+2)
                                            for y in range(1,bin.iloc[0].width+2) 
                                            for z in range(1,bin.iloc[0].height+2)]) == 2*item.iloc[i].length*item.iloc[i].height,
                     label=f'constraint5_y_{i}')    
  cqm.add_constraint(quicksum([NEz[i,x,y,z] for x in range(1,bin.iloc[0].length+2)
                                            for y in range(1,bin.iloc[0].width+2) 
                                            for z in range(1,bin.iloc[0].height+2)]) == 2*item.iloc[i].length*item.iloc[i].width,
                     label=f'constraint5_z_{i}')

### Constraint 6 : each item is stutted along the z-axis for at least 98% of its bottom surface
for i in range(len(item)): 
  cqm.add_constraint(quicksum([P[i,x,y,z] * T[x,y,z-1]  
                                            for x in range(1,bin.iloc[0].length+1)
                                            for y in range(1,bin.iloc[0].width+1)
                                            for z in range(1,bin.iloc[0].height+1)]) 
                    - (item.iloc[i].height - 1) * item.iloc[i].length * item.iloc[i].width
                    - 0.98 * item.iloc[i].length * item.iloc[i].width >= 0,    
                     label=f'constraint6_{i}')
  
### Constraint 7 : each item is stutted along the x-axis for at least 40% of its front surface
for i in range(len(item)): 
  cqm.add_constraint(quicksum([P[i,x,y,z] * T[x-1,y,z]  
                                            for x in range(1,bin.iloc[0].length+1)
                                            for y in range(1,bin.iloc[0].width+1)
                                            for z in range(1,bin.iloc[0].height+1)]) 
                    - (item.iloc[i].length - 1) * item.iloc[i].width * item.iloc[i].height
                    - 0.40 * item.iloc[i].width * item.iloc[i].height >= 0,    
                     label=f'constraint7_{i}')  

sampler = LeapHybridCQMSampler()
time_limit = 15
res = sampler.sample_cqm(cqm, time_limit=time_limit, label='3d bin packing')

print_cqm_stats(cqm)

res.resolve()
feasible_sampleset = res.filter(lambda d: d.is_feasible)
#print(feasible_sampleset)
try:
    best_feasible = feasible_sampleset.first.sample
    print(best_feasible)
    print('t', bin.iloc[0].length*bin.iloc[0].width*bin.iloc[0].height, total_item_volume, bin.iloc[0].length, bin.iloc[0].width, bin.iloc[0].height )

    for i in range(len(item)):
      x_from = bin.iloc[0].length
      y_from = bin.iloc[0].width
      z_from = bin.iloc[0].height
      x_to = 0
      y_to = 0
      z_to = 0
      dots = 0
      volume = 0
      for x, y, z in [(x,y,z) for x in range(1,bin.iloc[0].length+1) for y in range(1,bin.iloc[0].width+1) for z in range(1,bin.iloc[0].height+1)]:
        if P[i,x,y,z].energy(best_feasible) == 1:
          dots += 1
          if x < x_from:
            x_from = x
          if y < y_from:
            y_from = y
          if z < z_from:
            z_from = z
          if x > x_to:
            x_to = x
          if y > y_to:
            y_to = y
          if z > z_to:
            z_to = z                      
      volume = (x_to - x_from + 1) * (y_to - y_from + 1 ) * (z_to - z_from + 1)

      print(i, dots, volume, x_from, y_from, z_from, x_to, y_to, z_to)
    
except ValueError:
    raise RuntimeError(
        "Sampleset is empty, try increasing time limit or " +
        "adjusting problem config."
    )


