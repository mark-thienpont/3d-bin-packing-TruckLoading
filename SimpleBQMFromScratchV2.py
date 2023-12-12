import pandas as pd
import dimod
from dimod import quicksum, ConstrainedQuadraticModel, Real, Binary, SampleSet, Integer
from dwave.system import LeapHybridCQMSampler

from utils import print_cqm_stats

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

Y_list = []
for y in range(0,bin.iloc[0].width+1):
  Y_list.append(y)
Y = pd.DataFrame(data = Y_list, columns=['Y'])

Z_list = []
for z in range(0,bin.iloc[0].height+1):
  Z_list.append(z)
Z = pd.DataFrame(data = Z_list, columns=['Z'])

total_item_volume = quicksum([item.iloc[i].length * item.iloc[i].width * item.iloc[i].height for i in range(len(item))])

cqm = ConstrainedQuadraticModel()

# Item position
P = {(i,x,y,z,s): Binary(f'P_{i,x,y,z,s}') for i in range(len(item)) 
                                           for x in range(1, bin.iloc[0].length+1)
                                           for y in range(1, bin.iloc[0].width+1)
                                           for z in range(1, bin.iloc[0].height+1)
                                           for s in range(2)}  # s : starting-position (1) or not (0)

# Bin position
T = {(x,y,z):   Binary(f'T_{x,y,z}')   for x in range(0,bin.iloc[0].length+1)
                                       for y in range(0,bin.iloc[0].width+1)
                                       for z in range(0,bin.iloc[0].height+1)}

# Rotation
# not yet for now

# Constraint 1 : each item has exactly 1 starting position
for i in range(len(item)):
  cqm.add_constraint(quicksum([P[i,x,y,z,1] for x in range(1,bin.iloc[0].length + 1) 
                                            for y in range(1,bin.iloc[0].width  + 1) 
                                            for z in range(1,bin.iloc[0].height + 1)]) - 1 == 0,                     
                     label=f'constraint1_{i}')

# Constraint 2 : each item is constructed from starting position and known dimensions
# .... by consequence, no more need to check if it is contained inside the truck, or volume is complete etc
for i in range(len(item)): 
  # moving 'from'-selection
  for x,y,z in [(x,y,z) for x in range(item.iloc[i].length ,bin.iloc[0].length+1) 
                        for y in range(item.iloc[i].width  ,bin.iloc[0].width+1) 
                        for z in range(item.iloc[i].height ,bin.iloc[0].height+1)]: 
    cqm.add_constraint(P[i,x,y,z,0] - quicksum([P[i,x-dx,y-dy,z-dz,1] 
                                                             for dx in range(0,item.iloc[i].length) 
                                                             for dy in range(0,item.iloc[i].width ) 
                                                             for dz in range(0,item.iloc[i].height)]) == 0,
                      label=f'constraint2a_{i}_{x}_{y}_{z}')

  # fixed 'x-from'-selection
  for x,y,z in [(x,y,z) for x in range(1, item.iloc[i].length) 
                        for y in range(item.iloc[i].width  ,bin.iloc[0].width+1) 
                        for z in range(item.iloc[i].height ,bin.iloc[0].height+1)]: 
    cqm.add_constraint(P[i,x,y,z,0] - quicksum([P[i,x-dx,y-dy,z-dz,1] 
                                                             for dx in range(0,x) 
                                                             for dy in range(0,item.iloc[i].width ) 
                                                             for dz in range(0,item.iloc[i].height)]) == 0,
                      label=f'constraint2b_{i}_{x}_{y}_{z}')
  # fixed 'y-from'-selection
  for x,y,z in [(x,y,z) for x in range(item.iloc[i].length ,bin.iloc[0].length+1) 
                        for y in range(1, item.iloc[i].width) 
                        for z in range(item.iloc[i].height ,bin.iloc[0].height+1)]: 
    cqm.add_constraint(P[i,x,y,z,0] - quicksum([P[i,x-dx,y-dy,z-dz,1] 
                                                             for dx in range(0,item.iloc[i].length) 
                                                             for dy in range(0,y) 
                                                             for dz in range(0,item.iloc[i].height)]) == 0,
                      label=f'constraint2c_{i}_{x}_{y}_{z}')    
  # fixed 'z-from'-selection
  for x,y,z in [(x,y,z) for x in range(item.iloc[i].length ,bin.iloc[0].length+1) 
                        for y in range(item.iloc[i].width  ,bin.iloc[0].width+1) 
                        for z in range(1, item.iloc[i].height)]: 
    cqm.add_constraint(P[i,x,y,z,0] - quicksum([P[i,x-dx,y-dy,z-dz,1] 
                                                             for dx in range(0,item.iloc[i].length) 
                                                             for dy in range(0,item.iloc[i].width) 
                                                             for dz in range(0,z)]) == 0,
                      label=f'constraint2d_{i}_{x}_{y}_{z}') 
  # fixed 'xy-from'-selection
  for x,y,z in [(x,y,z) for x in range(1, item.iloc[i].length) 
                        for y in range(1, item.iloc[i].width) 
                        for z in range(item.iloc[i].height ,bin.iloc[0].height+1)]: 
    cqm.add_constraint(P[i,x,y,z,0] - quicksum([P[i,x-dx,y-dy,z-dz,1] 
                                                             for dx in range(0,x) 
                                                             for dy in range(0,y) 
                                                             for dz in range(0,item.iloc[i].height)]) == 0,
                      label=f'constraint2e_{i}_{x}_{y}_{z}')
  # fixed 'xz-from'-selection
  for x,y,z in [(x,y,z) for x in range(1, item.iloc[i].length) 
                        for y in range(item.iloc[i].width  ,bin.iloc[0].width+1) 
                        for z in range(1, item.iloc[i].height)]: 
    cqm.add_constraint(P[i,x,y,z,0] - quicksum([P[i,x-dx,y-dy,z-dz,1] 
                                                             for dx in range(0,x) 
                                                             for dy in range(0,item.iloc[i].width ) 
                                                             for dz in range(0,z)]) == 0,
                      label=f'constraint2f_{i}_{x}_{y}_{z}')
  # fixed 'yz-from'-selection
  for x,y,z in [(x,y,z) for x in range(item.iloc[i].length, bin.iloc[0].length+1) 
                        for y in range(1, item.iloc[i].width) 
                        for z in range(1, item.iloc[i].height)]: 
    cqm.add_constraint(P[i,x,y,z,0] - quicksum([P[i,x-dx,y-dy,z-dz,1] 
                                                             for dx in range(0,item.iloc[i].length)  
                                                             for dy in range(0,y) 
                                                             for dz in range(0,z)]) == 0,
                      label=f'constraint2g_{i}_{x}_{y}_{z}')
  # fixed 'xyz-from'-selection
  for x,y,z in [(x,y,z) for x in range(1, item.iloc[i].length) 
                        for y in range(1, item.iloc[i].width) 
                        for z in range(1, item.iloc[i].height)]: 
    cqm.add_constraint(P[i,x,y,z,0] - quicksum([P[i,x-dx,y-dy,z-dz,1] 
                                                             for dx in range(0,x)  
                                                             for dy in range(0,y) 
                                                             for dz in range(0,z)]) == 0,
                      label=f'constraint2h_{i}_{x}_{y}_{z}')

# Constraint 2c : each items has exactly the predefined number of dots
for i in range(len(item)): 
  cqm.add_constraint(quicksum([P[i,x,y,z,0] for x,y,z in [(x,y,z) for x in range(1,bin.iloc[0].length+1) 
                                                                  for y in range(1,bin.iloc[0].width+1) 
                                                                  for z in range(1,bin.iloc[0].height+1)]]) 
                              - item.iloc[i].length*item.iloc[i].width*item.iloc[i].height == 0,
                      label=f'constraint2c_{i}')

# Constraint 3 : derive T[x,y,z] from P[i,x,y,z]
for x,y,z in [(x,y,z) for x in range(1,bin.iloc[0].length+1) for y in range(1,bin.iloc[0].width+1) for z in range(1,bin.iloc[0].height+1)]:
  cqm.add_constraint(quicksum([P[i,x,y,z,0] for i in range(len(item))]) - T[x,y,z] == 0,
                     label=f'constraint3_{x}_{y}_{z}')

# Constraint 4 : items are not overlapping (via "sum of all dots in T = known sum of dots of all items"
cqm.add_constraint(quicksum([T[x,y,z] for x,y,z in [(x,y,z) for x in range(1,bin.iloc[0].length+1) 
                                                            for y in range(1,bin.iloc[0].width+1) 
                                                            for z in range(1,bin.iloc[0].height+1)]]) 
                   - total_item_volume == 0,
                    label=f'constraint4')

# Constraint 5 : 0-line added to enable stutting-constraints
cqm.add_constraint(quicksum([T[0,y,z] for y in range(1,bin.iloc[0].width+1) 
                                      for z in range(1,bin.iloc[0].height+1)] ) == (bin.iloc[0].width)*(bin.iloc[0].height),
                    label=f'constraint5_x')
cqm.add_constraint(quicksum([T[x,0,z] for x in range(bin.iloc[0].length+1) 
                                      for z in range(bin.iloc[0].height+1)] ) == 0,
                    label=f'constraint5_y')
cqm.add_constraint(quicksum([T[x,y,0] for x in range(1,bin.iloc[0].length+1) 
                                      for y in range(1,bin.iloc[0].width+1)] ) == (bin.iloc[0].width)*(bin.iloc[0].length),
                    label=f'constraint5_z')    

### Constraint 6 : each item is stutted along the z-axis for at least 98% of its bottom surface
for i in range(len(item)): 
  cqm.add_constraint(quicksum([P[i,x,y,z,0] * T[x,y,z-1]  
                                            for x in range(1,bin.iloc[0].length+1)
                                            for y in range(1,bin.iloc[0].width+1)
                                            for z in range(1,bin.iloc[0].height+1)]) 
                    - (item.iloc[i].height - 1) * item.iloc[i].length * item.iloc[i].width
                    - 0.98 * item.iloc[i].length * item.iloc[i].width >= 0,    
                     label=f'constraint6_{i}')
  
### Constraint 7 : each item is stutted along the x-axis for at least 40% of its front surface
for i in range(len(item)): 
  cqm.add_constraint(quicksum([P[i,x,y,z,0] * T[x-1,y,z]  
                                            for x in range(1,bin.iloc[0].length+1)
                                            for y in range(1,bin.iloc[0].width+1)
                                            for z in range(1,bin.iloc[0].height+1)]) 
                    - (item.iloc[i].length - 1) * item.iloc[i].width * item.iloc[i].height
                    - 0.40 * item.iloc[i].width * item.iloc[i].height >= 0,    
                     label=f'constraint7_{i}')  


# Objective 

target_X = bin.iloc[0].length / 2
target_Y = bin.iloc[0].width  / 2
target_Z = 0

obj_COG_X = (((quicksum(T[x,y,z] * X.iloc[x].X
                                    for x in range(1,bin.iloc[0].length+1)
                                    for y in range(1,bin.iloc[0].width+1)
                                    for z in range(1,bin.iloc[0].height+1)) / total_item_volume ) - target_X) ** 2 ) 

obj_COG_Y = (((quicksum((T[x,y,z] * Y.iloc[y].Y) 
                                    for x in range(1,bin.iloc[0].length+1)
                                    for y in range(1,bin.iloc[0].width+1)
                                    for z in range(1,bin.iloc[0].height+1)) / total_item_volume ) - target_Y) ** 2 ) 
obj_COG_Z = (((quicksum((T[x,y,z] * Z.iloc[z].Z) 
                                    for x in range(1,bin.iloc[0].length+1)
                                    for y in range(1,bin.iloc[0].width+1)
                                    for z in range(1,bin.iloc[0].height+1)) / total_item_volume ) - target_Z) ** 2 ) 

X_obj_coefficient = 1
Y_obj_coefficient = 1
Z_obj_coefficient = 1

cqm.set_objective(X_obj_coefficient * obj_COG_X +
                  Y_obj_coefficient * obj_COG_Y +
                  Z_obj_coefficient * obj_COG_Z) 

sampler = LeapHybridCQMSampler()
time_limit = 5
res = sampler.sample_cqm(cqm, time_limit=time_limit, label='3d bin packing')

print_cqm_stats(cqm)

res.resolve()
feasible_sampleset = res.filter(lambda d: d.is_feasible)
#print(feasible_sampleset)
try:
    best_feasible = feasible_sampleset.first.sample
    #print(best_feasible)
    print('t', bin.iloc[0].length*bin.iloc[0].width*bin.iloc[0].height, total_item_volume, bin.iloc[0].length, bin.iloc[0].width, bin.iloc[0].height )

    COG_X_teller = 0
    COG_Y_teller = 0
    COG_Z_teller = 0
    COG_noemer = 0
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
        if P[i,x,y,z,0].energy(best_feasible) == 1:
          dots += 1
          COG_X_teller += x
          COG_Y_teller += y
          COG_Z_teller += z
          COG_noemer += 1
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


    COG_X = COG_X_teller / COG_noemer
    COG_Y = COG_Y_teller / COG_noemer
    COG_Z = COG_Z_teller / COG_noemer
    #print(obj_COG_X, obj_COG_Y, obj_COG_Z)
    print(COG_X, COG_Y, COG_Z)
    
except ValueError:
    raise RuntimeError(
        "Sampleset is empty, try increasing time limit or " +
        "adjusting problem config."
    )


