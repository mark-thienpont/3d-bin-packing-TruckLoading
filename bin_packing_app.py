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


def _get_cqm_stats(cqm) -> str:
    cqm_info_stream = StringIO()
    sys.stdout = cqm_info_stream
    print_cqm_stats(cqm)
    sys.stdout = sys.__stdout__

    return cqm_info_stream.getvalue()


def _solve_bin_packing_instance(data: dict,
                                write_to_file: bool,
                                solution_filename: Optional[str],
                                use_cqm_solver: bool = True,
                                **st_plotly_kwargs):
    cases = Cases(data)
    bins = Bins(data, cases=cases)

    model_variables = Variables(cases, bins)

    ##cqm, effective_dimensions, effective_overlap = build_cqm(model_variables, bins, cases)
    cqm, effective_dimensions = build_cqm(model_variables, bins, cases)

    best_feasible = call_solver(cqm, time_limit, use_cqm_solver)

    plotly_fig = plot_cuboids(best_feasible, model_variables, cases,
                              bins, effective_dimensions, color_coded)

    st.plotly_chart(plotly_fig, **st_plotly_kwargs)

    st.code(_get_cqm_stats(cqm))

    if write_to_file:
        write_solution_to_file(solution_filename, cqm, 
                               model_variables, best_feasible,
                               cases, bins, effective_dimensions) ##, effective_overlap)


st.set_page_config(layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>Truck Loading Configuration</h1>",
    unsafe_allow_html=True
)

run_type = "File upload"
solver_type = "Constrained Quadratic Model"
use_cqm_solver = True

#solver_type = st.sidebar.radio(label="Choose solver to run problems on:",
#                               options=["Constrained Quadratic Model",
#                                        "CBC (Python-MIP)",
#                                        ])
#if solver_type == "Constrained Quadratic Model":
#    use_cqm_solver = True
#else:
#    use_cqm_solver = False

if run_type == "File upload":
    problem_filepath = st.sidebar.text_input(label="Problem instance file",
                                             value="input/LoadList_4.csv")
    time_limit = st.sidebar.number_input(label="Hybrid solver time limit (S)",
                                         value=5)

    color_coded = True
    display_input = False
    write_to_file = True
    solution_filename = st.sidebar.text_input(label="Solution filename", 
                                              value = "output/LoadList_4.csv")

    run_button = st.sidebar.button("Run")

    if run_button:
        data = read_instance(problem_filepath)
        _solve_bin_packing_instance(data,
                                    write_to_file,
                                    solution_filename,
                                    use_cqm_solver,
                                    **{"use_container_width": True})