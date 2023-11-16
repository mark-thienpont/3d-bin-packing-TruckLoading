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
        self.o = {(i, k): Binary(f'o_{i}_{k}') for i in range(num_cases)
                  for k in [0,2]}
        self.selector = {(i, j, k): Binary(f'sel_{i}_{j}_{k}')
                         for i, j in combinations(range(num_cases), r=2)
                         for k in range(6)}                                                        

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

    max_xff = {}
    min_xtt = {}
    max_yff = {}
    min_ytt = {}
    max_zff = {}
    min_ztt = {}    

    for i, j in combinations(range(num_cases), r=2):
        cqm.add_constraint(sum([vars.selector[i,j,s] for s in range(6)]) >= 1,
                         label=f'selector_{i}_{j}')

        cases_on_same_bin = 1                   
           
        # case i is behind of case j
        cqm.add_constraint(
            - (2 - cases_on_same_bin - vars.selector[i, j, 0]) * bins.length +
            (vars.x[i] + dx[i] - vars.x[j]) + 0 <= 0,
            label=f'overlap_{i}_{j}_0')                           

        # case i is left of case j
        cqm.add_constraint(
            -(2 - cases_on_same_bin - vars.selector[i, j, 1]) * bins.width +
            (vars.y[i] + dy[i] - vars.y[j]) + 0 <= 0,
            label=f'overlap_{i}_{j}_1')                     

        # case i is below case j 
        cqm.add_constraint(
            -(2 - cases_on_same_bin - vars.selector[i, j, 2]) * bins.height +
            (vars.z[i] + dz[i] - vars.z[j] ) + 0 <= 0,    
            label=f'overlap_{i}_{j}_2')                                                

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
