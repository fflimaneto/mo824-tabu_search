from scqbf.scqbf_evaluator import *
from scqbf.scqbf_solution import *
from scqbf.scqbf_ts import *
from scqbf.scqbf_instance import *

import pandas as pd
import numpy as np
import pickle
import os
import sys

def run_experiment(config: dict, tenure_func) -> pd.DataFrame:
    results = []
    exp_num = 1

    instance_paths = [(f"instances/gen{i}/instance{j}.txt", i, j) for i in range(1, 4) for j in range(1, 6)]

    for instance_path, gen, inst in instance_paths:
        instance = read_max_sc_qbf_instance(instance_path)
        print(f"{exp_num}: {inst}th instance of generation strategy {gen}. Path: {instance_path}")
        exp_num += 1
        
        time_limit = 60*30
        tenure = tenure_func(instance.n)
        tabu_search = ScQbfTabuSearch(instance, config=config, max_iterations=None, time_limit_secs=time_limit, tenure=tenure, patience=1000)
        best_solution = tabu_search.solve()

        if tabu_search.solve_time >= time_limit:
            stop_reason = "time_limit"
        else:
            stop_reason = "patience_exceeded" 
        
        evaluator = ScQbfEvaluator(instance)
        
        results.append({
            'gen': gen,
            'inst': inst,
            'n': instance.n,
            'stop_reason': stop_reason,
            'best_objective': evaluator.evaluate_objfun(best_solution),
            'coverage': evaluator.evaluate_coverage(best_solution),
            'time_taken': tabu_search.solve_time
        })
        
        last_result = results[-1]
        print(f"\tBest objective value: {last_result['best_objective']:.4f}")
        print(f"Selected elements: {best_solution.elements}")
        print(f"\tCoverage: {last_result['coverage']:.2%}")
        print(f"\tTime taken (secs): {last_result['time_taken']:.4f} s")
        print(f"\tStop reason: {last_result['stop_reason']}")
        print()
    
    df = pd.DataFrame(results)
    return df


def run_tests(tenure_config: int, local_search_method: str, diversification_method: str, probabilistic_weighted: bool):

    if tenure_config == 0:
        tenure_func = lambda x: math.ceil(math.sqrt(x))
    elif tenure_config == 1:
        tenure_func = lambda x: min(math.ceil(x/4), 20)
    else:
        tenure_func = lambda x: tenure_config

    config_1 = {
        "local_search_method": local_search_method,
        "diversification_method": diversification_method,
        "probabilistic_weighted": probabilistic_weighted
    }

    results_df = run_experiment(config_1, tenure_func=tenure_func)

    with open(f"pickles/results_config_{tenure_config}_{local_search_method}_{diversification_method}.pkl", "wb") as f:
        pickle.dump(results_df, f)
    results_df.to_csv(f"csv/results_config_{tenure_config}_{local_search_method}_{diversification_method}.csv", index=False)



if __name__ == "__main__":
    try:
        config = int(sys.argv[1])
        valid_config = True if (config >= 0 
                                and sys.argv[2] in ['best_improve', 'first_improve', 'probabilistic'] 
                                and sys.argv[3] in ['infrequent_elements', 'none']
                                and (len(sys.argv) < 5 or sys.argv[4] in ['True', 'False'])) \
                        else False
    except:
        valid_config = False

    if valid_config:
        run_tests(int(sys.argv[1]), sys.argv[2], sys.argv[3], (True if len(sys.argv) < 5 else sys.argv[4] == 'True'))
    else:
        print("Arguments needed. Expected: python main.py <tenure> <local_search_method> <diversification_method> [<probabilistic_weighted>]")
        print("If <tenure> = 0: tenure = sqrt(n); if <tenure> = 1: tenure = min(n/4,20) and if <tenure> > 1: tenure = <config>")
        print("<local_search_method> must be either 'best_improve', 'first_improve' or 'probabilistic'")
        print("<diversification_method> must be either 'infrequent_elements' or 'none'")
        print("<probabilistic_weighted> is optional and must be either 'True' or 'False'. Default is 'False'. Only used if <local_search_method> = 'probabilistic'")