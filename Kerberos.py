import json
import dimod
bqm = dimod.BinaryQuadraticModel.from_serializable(json.load(open("simNet5.json")))
k = 5       # looking for k sensors to be placed
bqm.update(dimod.generators.combinations(bqm.variables, k, strength=1000))   # strength : balance between MI and number of sens
from hybrid.reference.kerberos import KerberosSampler
kerberos_sampler = KerberosSampler() 
solution = kerberos_sampler.sample(bqm, 
                                #qpu_sampler=qpu_sampler, 
                                #qpu_reads=10000, 
                                max_iter=10,
                                convergence=3
                                #qpu_params={'label': 'Notebook - Feature Selection'}
                                )
best = solution.first.sample
energy = solution.first.energy
# best
result = [key for (key, value) in best.items() if value == 1]
print(result)