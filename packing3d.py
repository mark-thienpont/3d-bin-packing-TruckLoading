# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
from dimod import quicksum, ConstrainedQuadraticModel, Real, Binary, SampleSet, Integer
from dwave.system import LeapHybridCQMSampler
from itertools import combinations, permutations
import numpy as np
from typing import Tuple

from utils import print_cqm_stats, plot_cuboids
from utils import read_instance, write_solution_to_file


class Cases:
    """Class for representing cuboid item data in a 3D bin packing problem.

    Args:
         data: dictionary containing raw information for both bins and cases
    
    """

    def __init__(self, data):
        self.case_ids = np.repeat(data["case_ids"], data["quantity"])
        self.num_cases = np.sum(data["quantity"], dtype=np.int32)
        self.length = np.repeat(data["case_length"], data["quantity"])
        self.width = np.repeat(data["case_width"], data["quantity"])
        self.height = np.repeat(data["case_height"], data["quantity"])
        self.weight = np.repeat(data["case_weight"], data["quantity"])
        print(f'Number of cases: {self.num_cases}')


class Bins:
    """Class for representing cuboid container data in a 3D bin packing problem.

    Args:
        data: dictionary containing raw information for both bins and cases
        cases: Instance of ``Cases``, representing cuboid items packed into containers.

    """

    def __init__(self, data, cases):
        self.length = data["bin_dimensions"][0]
        self.width = data["bin_dimensions"][1]
        self.height = data["bin_dimensions"][2]
        self.num_bins = 1
        self.target_X = data["target_X"]

class Variables:
    """Class that collects all CQM model variables for the 3D bin packing problem.

    Args:
        cases: Instance of ``Cases``, representing cuboid items packed into containers.
        bins: Instance of ``Bins``, representing containers to pack cases into.
    
    """

    def __init__(self, cases: Cases, bins: Bins):
        num_cases = cases.num_cases

        self.x = {i: Integer(f'x_{i}', lower_bound=0, upper_bound=bins.length)
                  for i in range(num_cases)}
        self.y = {i: Integer(f'y_{i}', lower_bound=0, upper_bound=bins.width)
                  for i in range(num_cases)}
        self.z = {i: Integer(f'z_{i}', lower_bound=0, upper_bound=bins.height)
                  for i in range(num_cases)}

        self.floor = {i: Binary(f'floor_{i}') for i in range(num_cases)}  

        #self.QSLOIx2 = {i: Integer(f'QSLOIx2_{i}') for i in range(num_cases)}  
        #self.QSLOIx5 = {i: Integer(f'QSLOIx5_{i}') for i in range(num_cases)}  
        #self.QSLOIy2 = {i: Integer(f'QSLOIy2_{i}') for i in range(num_cases)}  
        #self.QSLOIy5 = {i: Integer(f'QSLOIy5_{i}') for i in range(num_cases)} 
        #self.QStest2 = {i: Integer(f'QStest2_{i}') for i in range(num_cases)}  
        #self.QStest5 = {i: Integer(f'QStest5_{i}') for i in range(num_cases)}                  

        self.o = {(i, k): Binary(f'o_{i}_{k}') for i in range(num_cases)
                  for k in [0,2]}

        self.selector = {(i, j, k): Binary(f'sel_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}

        self.neighbour = {(i, j, k): Binary(f'neighbour_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}   

        self.directneighbour = {(i, j, k): Binary(f'neighbour_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}                

        self.NOBxy     = {(i, j, k): Binary(f'neighbour_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}                           

        # Variables 'Lower Than'
        self.LTxff = {(i, j): Binary(f'LTxff_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTxtt = {(i, j): Binary(f'LTxtt_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTyff = {(i, j): Binary(f'LTyff_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTytt = {(i, j): Binary(f'LTytt_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTzff = {(i, j): Binary(f'LTzff_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTztt = {(i, j): Binary(f'LTztt_{i}_{j}') for i, j in combinations(range(num_cases), r=2)} 

        self.LTxtf = {(i, j): Binary(f'LTxtf_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTxft = {(i, j): Binary(f'LTxft_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTytf = {(i, j): Binary(f'LTytf_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTyft = {(i, j): Binary(f'LTyft_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTztf = {(i, j): Binary(f'LTztf_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTzft = {(i, j): Binary(f'LTzft_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}    

        # Variables 'LTE'
        self.LTExff = {(i, j): Binary(f'LTExff_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTExtt = {(i, j): Binary(f'LTExtt_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTEyff = {(i, j): Binary(f'LTEyff_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTEytt = {(i, j): Binary(f'LTEytt_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTEzff = {(i, j): Binary(f'LTEzff_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTEztt = {(i, j): Binary(f'LTEztt_{i}_{j}') for i, j in combinations(range(num_cases), r=2)} 

        self.LTExtf = {(i, j): Binary(f'LTExtf_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTExft = {(i, j): Binary(f'LTExft_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTEytf = {(i, j): Binary(f'LTEytf_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTEyft = {(i, j): Binary(f'LTEyft_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTEztf = {(i, j): Binary(f'LTEztf_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LTEzft = {(i, j): Binary(f'LTEzft_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}    

        # Variables 'Linear Overlap Binary'
        self.LOBx = {(i, j): Binary(f'LOBx_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LOBy = {(i, j): Binary(f'LOBy_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.LOBz = {(i, j): Binary(f'LOBz_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}     

        # Variables 'Surface Overlap Binary'
        self.SOBxy = {(i, j): Binary(f'SOBxy_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.SOBxz = {(i, j): Binary(f'SOBxz_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}
        self.SOByz = {(i, j): Binary(f'SOByz_{i}_{j}') for i, j in combinations(range(num_cases), r=2)}   

        ## Variables 'test'
        self.testx     = {(i, j, k): Binary(f'neighbour_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}   
        self.testy     = {(i, j, k): Binary(f'neighbour_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}                                   

        ## Variables 'Linear Overlap Integer'
        #self.LOIx = {(i, j): Integer(f'LOIx_{i}_{j}', lower_bound = -10000) for i, j in combinations(range(num_cases), r=2)}
        #self.LOIy = {(i, j): Integer(f'LOIy_{i}_{j}', lower_bound = -10000) for i, j in combinations(range(num_cases), r=2)}
        #self.LOIz = {(i, j): Integer(f'LOIz_{i}_{j}', lower_bound = -10000) for i, j in combinations(range(num_cases), r=2)}                 

        ## helper variable to determine real position "left", "right", "before", "after", "above", "below"
        #self.position = {(i, j, k): Binary(f'sel_{i}_{j}_{k}')
        #                 for i, j in combinations(range(num_cases), r=2)
        #                 for k in range(6)}                 


def _initialisation_constraints(cqm: ConstrainedQuadraticModel,
                                 vars: Variables, cases: Cases) -> list:

    cqm.add_constraint( vars.x[0] == 0, label=f'x_{0}') 
    cqm.add_constraint( vars.y[0] == 0, label=f'y_{0}') 
    cqm.add_constraint( vars.z[0] == 0, label=f'z_{0}') 


def _add_orientation_constraints(cqm: ConstrainedQuadraticModel,
                                 vars: Variables, cases: Cases) -> list:
    num_cases = cases.num_cases
    dx = {}
    dy = {}
    dz = {}
    x2 = {}
    y2 = {}
    z2 = {}
    for i in range(num_cases):
        p1 = list(
            permutations([cases.length[i], cases.width[i], cases.height[i]]))
        dx[i] = 0
        dy[i] = 0
        dz[i] = 0
        for j, (a, b, c) in enumerate(p1):
            if (j == 0 or j == 2):      # only rotations around z-axis is potentially feasible in all our cases
                dx[i] += a * vars.o[i, j]
                dy[i] += b * vars.o[i, j]
                dz[i] += c * vars.o[i, j]
        x2[i] = vars.x[i] + dx[i]
        y2[i] = vars.y[i] + dy[i]
        z2[i] = vars.z[i] + dz[i]

    for i in range(num_cases):
        cqm.add_discrete(quicksum([vars.o[i, k] for k in [0,2]]),
                         label=f'orientation_{i}')
    return [dx, dy, dz, x2, y2, z2]


def _add_geometric_constraints(cqm: ConstrainedQuadraticModel, vars: Variables,
                               bins: Bins, cases: Cases,
                               effective_dimensions: list):
    num_cases = cases.num_cases
    dx, dy, dz, x2, y2, z2 = effective_dimensions

    common_X = {}
    common_Y = {}
    common_Z = {}

    max_xff = {}
    min_xtt = {}
    max_yff = {}
    min_ytt = {}
    max_zff = {}
    min_ztt = {}    

    for i in range(num_cases):
        ## floor determination : eased toward 5 cm floating above bottom surface...
        cqm.add_constraint( ( 1 - vars.floor[i] ) * (5 - vars.z[i]) - vars.floor[i] * (6 - vars.z[i]) + 1 <= 0, label=f'floor_{i}')  

    for i, j in combinations(range(num_cases), r=2):
        cqm.add_constraint(sum([vars.selector[i,j,s] for s in range(6)]) >= 1,
                         label=f'selector_{i}_{j}')

        ## 2 cases can only be direct neighbours in maximally one direction only
        #cqm.add_constraint(sum([vars.neighbour[i,j,s] for s in range(6)]) <= 1,
        #                 label=f'neighbour_{i}_{j}')

        cases_on_same_bin = 1


        ##### determination LT x/y/z tt/ff/tf/ft #####
        cqm.add_constraint(vars.LTxtt[i,j] * (vars.x[i] + dx[i] - vars.x[j] - dx[j]) - (1 - vars.LTxtt[i,j]) * (vars.x[i] + dx[i] - vars.x[j] - dx[j] + 1) + 1 <= 0,
                            label=f'LT_xtt_{i}_{j}')  
        cqm.add_constraint(vars.LTytt[i,j] * (vars.y[i] + dy[i] - vars.y[j] - dy[j]) - (1 - vars.LTytt[i,j]) * (vars.y[i] + dy[i] - vars.y[j] - dy[j] + 1) + 1 <= 0,
                            label=f'LT_ytt_{i}_{j}')  
        cqm.add_constraint(vars.LTztt[i,j] * (vars.z[i] + dz[i] - vars.z[j] - dz[j]) - (1 - vars.LTztt[i,j]) * (vars.z[i] + dz[i] - vars.z[j] - dz[j] + 1) + 1 <= 0,
                            label=f'LT_ztt_{i}_{j}')  
        cqm.add_constraint(vars.LTxff[i,j] * (vars.x[i] - vars.x[j]) - (1 - vars.LTxff[i,j]) * (vars.x[i] - vars.x[j] + 1) + 1 <= 0,
                            label=f'LT_xff_{i}_{j}')
        cqm.add_constraint(vars.LTyff[i,j] * (vars.y[i] - vars.y[j]) - (1 - vars.LTyff[i,j]) * (vars.y[i] - vars.y[j] + 1) + 1 <= 0,
                            label=f'LT_yff_{i}_{j}')
        cqm.add_constraint(vars.LTzff[i,j] * (vars.z[i] - vars.z[j]) - (1 - vars.LTzff[i,j]) * (vars.z[i] - vars.z[j] + 1) + 1 <= 0,
                            label=f'LT_zff_{i}_{j}')
        cqm.add_constraint(vars.LTxtf[i,j] * (vars.x[i] + dx[i] - vars.x[j]) - (1 - vars.LTxtf[i,j]) * (vars.x[i] + dx[i] - vars.x[j] + 1) + 1 <= 0,
                            label=f'LT_xtf_{i}_{j}')  
        cqm.add_constraint(vars.LTytf[i,j] * (vars.y[i] + dy[i] - vars.y[j]) - (1 - vars.LTytf[i,j]) * (vars.y[i] + dy[i] - vars.y[j] + 1) + 1 <= 0,
                            label=f'LT_ytf_{i}_{j}')  
        cqm.add_constraint(vars.LTztf[i,j] * (vars.z[i] + dz[i] - vars.z[j]) - (1 - vars.LTztf[i,j]) * (vars.z[i] + dz[i] - vars.z[j] + 1) + 1 <= 0,
                            label=f'LT_ztf_{i}_{j}')  
        cqm.add_constraint(vars.LTxft[i,j] * (vars.x[i] - vars.x[j] - dx[j]) - (1 - vars.LTxft[i,j]) * (vars.x[i] - vars.x[j] - dx[j] + 1) + 1 <= 0,
                            label=f'LT_xft_{i}_{j}')
        cqm.add_constraint(vars.LTyft[i,j] * (vars.y[i] - vars.y[j] - dy[j]) - (1 - vars.LTyft[i,j]) * (vars.y[i] - vars.y[j] - dy[j] + 1) + 1 <= 0,
                            label=f'LT_yft_{i}_{j}')
        cqm.add_constraint(vars.LTzft[i,j] * (vars.z[i] - vars.z[j] - dz[j]) - (1 - vars.LTzft[i,j]) * (vars.z[i] - vars.z[j] - dz[j] + 1) + 1 <= 0,
                            label=f'LT_zft_{i}_{j}')
 
        ##### determination LTE x/y/z tt/ff/ft/tf #####
        cqm.add_constraint(vars.LTExtt[i,j] * (vars.x[i] + dx[i] - vars.x[j] - dx[j] - 1) - (1 - vars.LTExtt[i,j]) * (vars.x[i] + dx[i] - vars.x[j] - dx[j]) + 1 <= 0,
                            label=f'LTE_xtt_{i}_{j}')  
        cqm.add_constraint(vars.LTEytt[i,j] * (vars.y[i] + dy[i] - vars.y[j] - dy[j] - 1) - (1 - vars.LTEytt[i,j]) * (vars.y[i] + dy[i] - vars.y[j] - dy[j]) + 1 <= 0,
                            label=f'LTE_ytt_{i}_{j}')  
        cqm.add_constraint(vars.LTEztt[i,j] * (vars.z[i] + dz[i] - vars.z[j] - dz[j] - 1) - (1 - vars.LTEztt[i,j]) * (vars.z[i] + dz[i] - vars.z[j] - dz[j]) + 1 <= 0,
                            label=f'LTE_ztt_{i}_{j}')  
        cqm.add_constraint(vars.LTExff[i,j] * (vars.x[i] - vars.x[j] - 1) - (1 - vars.LTExff[i,j]) * (vars.x[i] - vars.x[j]) + 1 <= 0,
                            label=f'LTE_xff_{i}_{j}')
        cqm.add_constraint(vars.LTEyff[i,j] * (vars.y[i] - vars.y[j] - 1) - (1 - vars.LTEyff[i,j]) * (vars.y[i] - vars.y[j]) + 1 <= 0,
                            label=f'LTE_yff_{i}_{j}')
        cqm.add_constraint(vars.LTEzff[i,j] * (vars.z[i] - vars.z[j] - 1) - (1 - vars.LTEzff[i,j]) * (vars.z[i] - vars.z[j]) + 1 <= 0,
                            label=f'LTE_zff_{i}_{j}')
        cqm.add_constraint(vars.LTExtf[i,j] * (vars.x[i] + dx[i] - vars.x[j] - 1) - (1 - vars.LTExtf[i,j]) * (vars.x[i] + dx[i] - vars.x[j]) + 1 <= 0,
                            label=f'LTE_xtf_{i}_{j}')  
        cqm.add_constraint(vars.LTEytf[i,j] * (vars.y[i] + dy[i] - vars.y[j] - 1) - (1 - vars.LTEytf[i,j]) * (vars.y[i] + dy[i] - vars.y[j]) + 1 <= 0,
                            label=f'LTE_ytf_{i}_{j}')  
        cqm.add_constraint(vars.LTEztf[i,j] * (vars.z[i] + dz[i] - vars.z[j] - 1) - (1 - vars.LTEztf[i,j]) * (vars.z[i] + dz[i] - vars.z[j]) + 1 <= 0,
                            label=f'LTE_ztf_{i}_{j}')  
        cqm.add_constraint(vars.LTExft[i,j] * (vars.x[i] - vars.x[j] - dx[j] - 1) - (1 - vars.LTExft[i,j]) * (vars.x[i] - vars.x[j] - dx[j]) + 1 <= 0,
                            label=f'LTE_xft_{i}_{j}')
        cqm.add_constraint(vars.LTEyft[i,j] * (vars.y[i] - vars.y[j] - dy[j] - 1) - (1 - vars.LTEyft[i,j]) * (vars.y[i] - vars.y[j] - dy[j]) + 1 <= 0,
                            label=f'LTE_yft_{i}_{j}')
        cqm.add_constraint(vars.LTEzft[i,j] * (vars.z[i] - vars.z[j] - dz[j] - 1) - (1 - vars.LTEzft[i,j]) * (vars.z[i] - vars.z[j] - dz[j]) + 1 <= 0,
                            label=f'LTE_zft_{i}_{j}')


        ##### determination of 'Linear Overlap Binary' LOB x/y/z
        cqm.add_constraint(vars.LTxft[i,j] * (1 - vars.LTxtf[i,j]) + (1 - vars.LTxft[i,j]) * vars.LTxtf[i,j] - vars.LOBx[i,j] == 0, label=f'LOBx_{i}_{j}')        
        cqm.add_constraint(vars.LTyft[i,j] * (1 - vars.LTytf[i,j]) + (1 - vars.LTyft[i,j]) * vars.LTytf[i,j] - vars.LOBy[i,j] == 0, label=f'LOBy_{i}_{j}')        
        cqm.add_constraint(vars.LTzft[i,j] * (1 - vars.LTztf[i,j]) + (1 - vars.LTzft[i,j]) * vars.LTztf[i,j] - vars.LOBz[i,j] == 0, label=f'LOBz_{i}_{j}')        
 

        ###### determination of 'Surface Overlap Binary' SOB xy/xz/yz
        cqm.add_constraint(vars.LOBx[i,j] * vars.LOBy[i,j] - vars.SOBxy[i,j] == 0, label=f'SOBxy_{i}_{j}')   
        cqm.add_constraint(vars.LOBx[i,j] * vars.LOBz[i,j] - vars.SOBxz[i,j] == 0, label=f'SOBxz_{i}_{j}')   
        cqm.add_constraint(vars.LOBy[i,j] * vars.LOBz[i,j] - vars.SOByz[i,j] == 0, label=f'SOByz_{i}_{j}')                       
           
        # case i is behind of case j
        cqm.add_constraint(
            - (2 - cases_on_same_bin - vars.selector[i, j, 0]) * bins.length +
            (vars.x[i] + dx[i] - vars.x[j]) + 0 <= 0,
            label=f'overlap_{i}_{j}_0')                           

        # case i is left of case j
        cqm.add_constraint(
            -(2 - cases_on_same_bin - vars.selector[i, j, 1]) * bins.width +
            (vars.y[i] + dy[i] - vars.y[j]) + 0 <= 0,
            label=f'neighbour_{i}_{j}_1')                     

        # case i is below case j 
        cqm.add_constraint(
            -(2 - cases_on_same_bin - vars.selector[i, j, 2]) * bins.height +
            (vars.z[i] + dz[i] - vars.z[j] ) + 0 <= 0,    
            label=f'overlap_{i}_{j}_2')    

        ## case i is below case j, and is touching (i.e. neighbour)          
        cqm.add_constraint(
            vars.neighbour[i,j,2]*(vars.z[j]-vars.z[i]-dz[i])-(1-vars.neighbour[i,j,2])*(vars.z[j]-vars.z[i]-dz[i]+1) >= 0,
            label=f'neighbour_{i}_{j}_2')

        cqm.add_constraint(
            vars.neighbour[i,j,2]-vars.selector[i,j,2]<=0,
            label=f'neighbour2_{i}_{j}_4')      
        cqm.add_constraint(vars.neighbour[i,j,2] * vars.SOBxy[i,j] - vars.NOBxy[i,j,2] == 0, label=f'NOBxy_{i}_{j}_2')     

        cqm.add_constraint(
            ##direct neighbour : distance = 0
            vars.NOBxy[i,j,2] * (vars.LTEztf[i,j]-vars.LTztf[i,j]) - vars.directneighbour[i,j,2] == 0,            
            label=f'directneighbour_{i}_{j}_2')                                                  

        # case i is in front of of case j
        cqm.add_constraint(
            -(2 - cases_on_same_bin - vars.selector[i, j, 3]) * bins.length +
            (vars.x[j] + dx[j] - vars.x[i]) + 0 <= 0,
            label=f'overlap_{i}_{j}_3')                   

        # case i is right of case k
        cqm.add_constraint(
            -(2 - cases_on_same_bin - vars.selector[i, j, 4]) * bins.width +
            (vars.y[j] + dy[j] - vars.y[i]) + 0 <= 0,
            label=f'overlap_{i}_{j}_4')                       

        # case i is above case j
        cqm.add_constraint(
            -(2 - cases_on_same_bin - vars.selector[i, j, 5]) * bins.height +
            (vars.z[j] + dz[j] - vars.z[i] ) + 0 <= 0,    
            label=f'overlap_{i}_{j}_5')   
        # case i is above case j, and is touching (i.e. neighbour)           
        cqm.add_constraint(
            vars.neighbour[i,j,5]*(vars.z[i]-vars.z[j]-dz[j])-(1-vars.neighbour[i,j,5])*(vars.z[i]-vars.z[j]-dz[j]+1) >= 0,
            ##revised towards maximum 5 cm distance
            ##(1-vars.neighbour[i,j,5])*(5-(vars.z[i]-vars.z[j]-dz[j])) - vars.neighbour[i,j,5]*(6 - (vars.z[i]-vars.z[j]-dz[j])) + 1 <= 0,
            label=f'neighbour_{i}_{j}_5')
        cqm.add_constraint(
            vars.neighbour[i,j,5]-vars.selector[i,j,5]<=0,
            label=f'neighbour2_{i}_{j}_5')
        cqm.add_constraint(vars.neighbour[i,j,5] * vars.SOBxy[i,j] - vars.NOBxy[i,j,5] == 0, label=f'NOBxy_{i}_{j}_5')   
        cqm.add_constraint(
            ##direct neighbour : distance = 0
            vars.NOBxy[i,j,5] * (vars.LTEzft[i,j]-vars.LTzft[i,j]) - vars.directneighbour[i,j,5] == 0,            
            label=f'directneighbour_{i}_{j}_5')                   


def _add_stacking_constraints(cqm: ConstrainedQuadraticModel, vars: Variables,
                               bins: Bins, cases: Cases,
                               effective_dimensions: list):
    num_cases = cases.num_cases

    for i, j in  combinations(range(num_cases), r=2):
        ### IF directneighbour == 1 (i.e. z-direction touching items), then the 'above' item should fit the below item 
        cqm.add_constraint((vars.directneighbour[i,j,2])*(2 - vars.LTExff[i,j] - (1 - vars.LTxtt[i,j])) - vars.testx[i,j,2] == 0, label=f'pyramid_x_{i}_{j}_{2}')
        cqm.add_constraint((vars.directneighbour[i,j,5])*(2 - (1 - vars.LTxff[i,j]) - vars.LTExtt[i,j]) - vars.testx[i,j,5] == 0, label=f'pyramid_x_{i}_{j}_{5}')      
        cqm.add_constraint((vars.directneighbour[i,j,2])*(2 - vars.LTEyff[i,j] - (1 - vars.LTytt[i,j])) - vars.testy[i,j,2] == 0, label=f'pyramid_y_{i}_{j}_{2}')
        cqm.add_constraint((vars.directneighbour[i,j,5])*(2 - (1 - vars.LTyff[i,j]) - vars.LTEytt[i,j]) - vars.testy[i,j,5] == 0, label=f'pyramid_y_{i}_{j}_{5}')

    #for k in range(num_cases):
    #    ## ofwel staat item[k] op de bodem, ofwel op een andere pallet
    #    cqm.add_constraint((quicksum([vars.NOBxy[i,k,2] for i in [i for (i, j) in combinations(range(num_cases), r=2) if j == k]]) 
    #                                     + quicksum([vars.NOBxy[k,j,5] for j in [j for (i, j) in combinations(range(num_cases), r=2) if i == k]])) == 1, label=f'test_{k}')


def _add_boundary_constraints(cqm: ConstrainedQuadraticModel, vars: Variables,
                              bins: Bins, cases: Cases,
                              effective_dimensions: list):
    num_cases = cases.num_cases
    dx, dy, dz, x2, y2, z2 = effective_dimensions
    for i in range(num_cases):

        cqm.add_constraint(vars.x[i] + dx[i] - bins.length <= 0,
                            label=f'maxx_{i}_less')

        cqm.add_constraint(vars.y[i] + dy[i] - bins.width <= 0,
                            label=f'maxy_{i}_less')

        cqm.add_constraint(vars.z[i] + dz[i] - bins.height <= 0,
                            label=f'maxx_height_{i}')                            

def _define_objective(cqm: ConstrainedQuadraticModel, vars: Variables,
                      bins: Bins, cases: Cases, effective_dimensions: list):
    num_cases = cases.num_cases
    dx, dy, dz, x2, y2, z2 = effective_dimensions
    
    target_X = bins.target_X
    target_Y = bins.width / 2
    target_Z = quicksum(cases.weight[i] * cases.height[i] for i in range(num_cases)) / quicksum(cases.weight[i] for i in range(num_cases)) / 2

    obj_COG_X = ((quicksum((vars.x[i] + dx[i]/2 ) * cases.weight[i] for i in range(num_cases)) / quicksum(cases.weight[i] for i in range(num_cases)) - target_X) ** 2 ) 
    obj_COG_Y = ((quicksum((vars.y[i] + dy[i]/2 ) * cases.weight[i] for i in range(num_cases)) / quicksum(cases.weight[i] for i in range(num_cases)) - target_Y) ** 2 ) 
    obj_COG_Z = ((quicksum((vars.z[i] + dz[i]/2 ) * cases.weight[i] for i in range(num_cases)) / quicksum(cases.weight[i] for i in range(num_cases)) - 0) ** 2 )
    #obj_test = quicksum((1-vars.test[i]) for i in range(num_cases))

    first_obj_coefficient = 1
    second_obj_coefficient = 1
    third_obj_coefficient = 100
    cqm.set_objective(first_obj_coefficient  * obj_COG_X +
                      second_obj_coefficient * obj_COG_Y +
                      third_obj_coefficient  * obj_COG_Z) 
    cqm.set_objective(obj_COG_Z)                       


def build_cqm(vars: Variables, bins: Bins,
              cases: Cases) -> Tuple[ConstrainedQuadraticModel, list]:
    """Builds the CQM model from the problem variables and data.

    Args:
        vars: Instance of ``Variables`` that defines the complete set of variables
            for the 3D bin packing problem.
        bins: Instance of ``Bins``, representing containers to pack cases into.
        cases: Instance of ``Cases``, representing cuboid items packed into containers.

    Returns:
        A ``dimod.CQM`` object that defines the 3D bin packing problem.
        effective_dimensions: List of case dimensions based on orientations of cases.
    
    """
    cqm = ConstrainedQuadraticModel()
    _initialisation_constraints(cqm, vars, cases)
    effective_dimensions = _add_orientation_constraints(cqm, vars, cases)
    _add_geometric_constraints(cqm, vars, bins, cases, effective_dimensions)
    _add_boundary_constraints(cqm, vars, bins, cases, effective_dimensions)
    _add_stacking_constraints(cqm, vars, bins, cases, effective_dimensions)
    _define_objective(cqm, vars, bins, cases, effective_dimensions)

    return cqm, effective_dimensions


def call_solver(cqm: ConstrainedQuadraticModel,
                time_limit: float,
                use_cqm_solver: bool = True) -> SampleSet:
    """Helper function to call the CQM Solver.

    Args:
        cqm: A ``CQM`` object that defines the 3D bin packing problem.
        time_limit: Time limit parameter to pass on to the CQM sampler.

    Returns:
        A ``dimod.SampleSet`` that represents the best feasible solution found.
    
    """
    if use_cqm_solver:
        sampler = LeapHybridCQMSampler()
        res = sampler.sample_cqm(cqm, time_limit=time_limit, label='3d bin packing')
    else:
        sampler = MIPCQMSolver()
        res = sampler.sample_cqm(cqm, time_limit=time_limit)

    res.resolve()
    feasible_sampleset = res.filter(lambda d: d.is_feasible)
    print(feasible_sampleset)
    try:
        best_feasible = feasible_sampleset.first.sample

        return best_feasible
        
    except ValueError:
        raise RuntimeError(
            "Sampleset is empty, try increasing time limit or " +
            "adjusting problem config."
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str, nargs="?",
                        help="Filename with path to bin-packing data file.",
                        default="input/LoadList_4.csv")
    
    parser.add_argument("--output_filepath", type=str,  nargs="?",
                        help="Path for the output solution file.",
                        default=None)

    parser.add_argument("--time_limit", type=float, nargs="?",
                        help="Time limit for the hybrid CQM Solver to run in"
                             " seconds.",
                        default=5)
    
    parser.add_argument("--use_cqm_solver", type=bool, nargs="?",
                        help="Flag to either use CQM or MIP solver",
                        default=True)
    
    parser.add_argument("--html_filepath", type=str, nargs="?",
                        help="Filename with path to plot html file.",
                        default=None)

    parser.add_argument("--color_coded", type=bool, nargs="?",
                        help="View plot with coded or randomly colored cases.",
                        default=False)

    args = parser.parse_args()
    output_filepath = args.output_filepath
    time_limit = args.time_limit
    use_cqm_solver = args.use_cqm_solver
    html_filepath = args.html_filepath
    color_coded = args.color_coded

    data = read_instance(args.data_filepath)
    cases = Cases(data)
    bins = Bins(data, cases)

    vars = Variables(cases, bins)

    cqm, effective_dimensions = build_cqm(vars, bins, cases)

    print_cqm_stats(cqm)

    best_feasible = call_solver(cqm, time_limit, use_cqm_solver)

    if output_filepath is not None:
        write_solution_to_file(output_filepath, cqm, vars, best_feasible, 
                               cases, bins, effective_dimensions)

    fig = plot_cuboids(best_feasible, vars, cases,
                       bins, effective_dimensions, color_coded)

    if html_filepath is not None:
        fig.write_html(html_filepath)

    fig.show()
