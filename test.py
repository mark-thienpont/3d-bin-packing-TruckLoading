from io import StringIO
import numpy as np
import sys
import streamlit as st
from typing import Optional
from packing3d import (Cases,
                       Bins,
                       Variables,
                       build_cqm,
                       call_solver)
from utils import (print_cqm_stats,
                   plot_cuboids,
                   read_instance,
                   write_solution_to_file,
                   write_input_data)
#from bin_packing_app import (_get_cqm_stats,
#                             _solve_bin_packing_instance,
#                             )


problem_filepath = "input/LoadList_835982.csv"
time_limit = 5
color_coded = True
display_input = False
write_to_file = "Write solution to file"
solution_filename = "output/LoadList_835982.csv"
#run_button = st.sidebar.button("Run")

data = read_instance(problem_filepath)

cases = Cases(data)
bins = Bins(data, cases=cases)

model_variables = Variables(cases, bins)




cqm, effective_dimensions = build_cqm(model_variables, bins, cases)

test = 4

#best_feasible = call_solver(cqm, time_limit, use_cqm_solver)                             