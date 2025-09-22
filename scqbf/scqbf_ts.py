# scqbf/scqbf_Tabu Search.py

import math
from scqbf.scqbf_instance import *
from scqbf.scqbf_evaluator import *
import random
import time

class ScQbfTabuSearch:

    FAKE = -1
    
    def __init__(self, instance: ScQbfInstance,
                 max_iterations,
                 config: dict = {
                    "construction_method": "traditional",  # traditional | random_plus_greedy | sampled_greedy
                    "construction_args": (),
                    "local_search_method": "best_improve"  # best_improve | first_improve
                 },
                 time_limit_secs: float = None,
                 tenure: int = None,
                 patience: int = 20,
                 debug: bool = False):
        
        
        self.instance = instance
        self.max_iterations = max_iterations
        self.config = config
        self.time_limit_secs = time_limit_secs
        self.patience = patience
        self.debug = debug
        self.solve_time = 0
        self.tenure = tenure
        self.TL = [self.FAKE] * (2 * tenure)
        self.iterations = 0
        self.evaluator = ScQbfEvaluator(instance)
        self.bestValue = float("-inf")
        self.currentValue = 0


    def solve(self) -> ScQbfSolution:
        if self.instance is None:
            raise ValueError("Problem instance is not initialized")
        
        best_sol = ScQbfSolution([])
        start_time = time.perf_counter()
        self.iterations = 0
        current_patience = self.patience

        sol = self._constructive_heuristic()
        if self.debug:
            print(f"Constructed solution (iteration {self.iterations}): {sol.elements}")

        if not self.evaluator.is_solution_valid(sol):
            if self.debug:
                print("Constructed solution is not feasible, fixing...")
            sol = self._fix_solution(sol)

        while ((self.iterations < self.max_iterations) if self.max_iterations is not None else True):
            self.iterations += 1
            
            sol = self._local_search(sol)
            
            if (self.evaluator.evaluate_objfun(sol) > self.evaluator.evaluate_objfun(best_sol)):
                best_sol = sol
                current_patience = self.patience
            else:
                if current_patience is not None:
                    current_patience -= 1
                    if current_patience <= 0:
                        print(f"Patience exhausted, no improvement to the objective solution in {self.patience} iterations, stopping Tabu Search.")
                        break
            
            self.solve_time = time.perf_counter() - start_time
            if self.time_limit_secs is not None and self.solve_time >= self.time_limit_secs:
                print(f"Time limit of {self.time_limit_secs} seconds reached, stopping Tabu Search.")
                break
            
        return best_sol
    
    def _fix_solution(self, sol: ScQbfSolution) -> ScQbfSolution:
        """
        This function is called when the constructed solution is not feasible.
        It'll add the most covering elements until the solution is feasible.
        """
        while not self.evaluator.is_solution_valid(sol):
            cl = [i for i in range(self.instance.n) if i not in sol.elements]
            best_cand = None
            best_coverage = -1
            
            for cand in cl:
                coverage = self.evaluator.evaluate_insertion_delta_coverage(cand, sol)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_cand = cand
            
            if best_cand is not None:
                sol.elements.append(best_cand)
            else:
                break
        
        if not self.evaluator.is_solution_valid(sol):
            raise ValueError("Could not fix the solution to be feasible")
        
        return sol

    def _constructive_heuristic(self) -> ScQbfSolution:
        return self._constructive_greedy_heuristic()
    
    def _constructive_greedy_heuristic(self) -> ScQbfSolution:
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)] # makeCl

        while not self.evaluator.is_solution_valid(constructed_sol): # Constructive Stop Criteria
            # traditional constructive heuristic
            max_delta = -math.inf
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            best_cand = None
            
            for candidate_element in cl:
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun > max_delta and self.evaluator.evaluate_insertion_delta_coverage(candidate_element, constructed_sol) > 0:
                    max_delta = delta_objfun
                    best_cand = candidate_element

            # Adds best candidate found to the solution
            constructed_sol.elements.append(best_cand)

        self.currentValue = self.evaluator.evaluate_objfun(constructed_sol)

        return constructed_sol

    def _constructive_heuristic_traditional(self, alpha: float) -> ScQbfSolution:
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)] # makeCl

        while not self.evaluator.is_solution_valid(constructed_sol): # Constructive Stop Criteria
            # traditional constructive heuristic
            rcl = []
            min_delta = math.inf
            max_delta = -math.inf
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            
            for candidate_element in cl:
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun < min_delta:
                    min_delta = delta_objfun
                if delta_objfun > max_delta:
                    max_delta = delta_objfun
            
            # This is where we define the RCL.
            for candidate_element in cl:
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun >= (min_delta + alpha * (max_delta - min_delta)):

                    ## ONLY add to rcl if coverage increases
                    if self.evaluator.evaluate_insertion_delta_coverage(candidate_element, constructed_sol) > 0:
                        rcl.append(candidate_element)

            # Randomly select an element from the RCL to add to the solution
            if rcl:
                chosen_element = random.choice(rcl)
                constructed_sol.elements.append(chosen_element)
            else:
                break

        self.currentValue = self.evaluator.evaluate_objfun(constructed_sol)

        return constructed_sol

    def _constructive_heuristic_random_plus_greedy(self, alpha: float, p: float):
        constructed_sol = ScQbfSolution([])
        cl = [i for i in range(self.instance.n)] # make_cl

        # Select first p elements at random
        for _ in range(int(p * self.instance.n)):
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            constructed_sol.elements.append(random.choice(cl))
        
        # Continue with a purely greedy approach
        while not self.evaluator.is_solution_valid(constructed_sol): # Constructive Stop Criteria
            cl = [i for i in cl if i not in constructed_sol.elements] # update_cl
            
            best_delta = float("-inf")
            best_cand_in = -1
            
            for candidate_element in cl:
                # Only consider candidates that improve coverage and objective function
                delta_objfun = self.evaluator.evaluate_insertion_delta(candidate_element, constructed_sol)
                if delta_objfun > best_delta and self.evaluator.evaluate_insertion_delta_coverage(candidate_element, constructed_sol) > 0:
                    best_cand_in = candidate_element
                    best_delta = delta_objfun
            
            if best_delta > 0:
                constructed_sol.elements.append(best_cand_in)
            else:
                break

        return constructed_sol

    ####################

    def _local_search(self, starting_point: ScQbfSolution) -> ScQbfSolution:
        if self.config.get("local_search_method", False) == "best_improve":
            return self._local_search_best_improve(starting_point)
        elif self.config.get("local_search_method", False) == "first_improve":
            return self._local_search_first_improve(starting_point)


    def _local_search_best_improve(self, starting_point: ScQbfSolution) -> ScQbfSolution:
        sol = ScQbfSolution(starting_point.elements.copy())
        
        best_delta = float("-inf")
        best_cand_in = None
        best_cand_out = None

        cl = [i for i in range(self.instance.n) if i not in sol.elements]

        # Evaluate insertions
        for cand_in in cl:
            delta = self.evaluator.evaluate_insertion_delta(cand_in, sol)
            if delta > best_delta and (cand_in not in self.TL or self.currentValue + delta > self.bestValue):
                best_delta = delta
                best_cand_in = cand_in
                best_cand_out = None

        # Evaluate removals
        for cand_out in sol.elements:
            delta = self.evaluator.evaluate_removal_delta(cand_out, sol)
            if delta > best_delta and (cand_out not in self.TL or self.currentValue + delta > self.bestValue):
                # Check if removing this element would break feasibility
                temp_sol = ScQbfSolution(sol.elements.copy())
                temp_sol.elements.remove(cand_out)
                if self.evaluator.is_solution_valid(temp_sol):
                    best_delta = delta
                    best_cand_in = None
                    best_cand_out = cand_out

        # Evaluate exchanges
        for cand_in in cl:
            for cand_out in sol.elements:
                delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, sol)
                if delta > best_delta and ((cand_in not in self.TL and cand_out not in self.TL) or self.currentValue + delta > self.bestValue):
                    # Check if this exchange would break feasibility
                    temp_sol = ScQbfSolution(sol.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    temp_sol.elements.append(cand_in)
                    if self.evaluator.is_solution_valid(temp_sol):
                        best_delta = delta
                        best_cand_in = cand_in
                        best_cand_out = cand_out


        # Apply the best move found and update the Tabu List
        if best_cand_in is not None:
            sol.elements.append(best_cand_in)
            self.TL.append(best_cand_in)
        else:
            self.TL.append(self.FAKE)

        if best_cand_out is not None:
            sol.elements.remove(best_cand_out)
            self.TL.append(best_cand_out)
        else:
            self.TL.append(self.FAKE)
        
        self.TL = self.TL[2:]

        self.currentValue = self.evaluator.evaluate_objfun(sol)
        
        return sol

    def _local_search_first_improve(self, starting_point: ScQbfSolution) -> ScQbfSolution:
        sol = ScQbfSolution(starting_point.elements.copy())
        
        cl = [i for i in range(self.instance.n) if i not in sol.elements]
        current_elements = sol.elements.copy()

        random.shuffle(cl)
        random.shuffle(current_elements)

        neighborhoods = ['insertion', 'removal', 'exchange']
        random.shuffle(neighborhoods)

        best_delta = float("-inf")
        best_cand_in = None
        best_cand_out = None

        move_found = False

        for neighborhood in neighborhoods:

            if neighborhood == 'insertion':
                # Evaluate insertions
                for cand_in in cl:
                    delta = self.evaluator.evaluate_insertion_delta(cand_in, sol)
                    valid_move = True if (cand_in not in self.TL or self.currentValue + delta > self.bestValue) else False
                    if delta > best_delta and valid_move:
                        best_delta = delta
                        best_cand_in = cand_in
                        best_cand_out = None
                    if delta > 0 and valid_move:
                        move_found = True
                        if self.debug:
                            print(f"[local_search]: First improvement found (insertion)! Delta: {delta}, in {cand_in}")
                        break
                        
            if neighborhood == 'removal':
                # Evaluate removals
                for cand_out in sol.elements:
                    delta = self.evaluator.evaluate_removal_delta(cand_out, sol)
                    valid_move = True if (cand_out not in self.TL or self.currentValue + delta > self.bestValue) else False
                    if delta > best_delta and valid_move:
                        # Check if removing this element would break feasibility
                        temp_sol = ScQbfSolution(sol.elements.copy())
                        temp_sol.elements.remove(cand_out)
                        if self.evaluator.is_solution_valid(temp_sol):
                            best_delta = delta
                            best_cand_in = None
                            best_cand_out = cand_out
                    if delta > 0 and valid_move:
                        move_found = True
                        if self.debug:
                            print(f"[local_search]: First improvement found (removal)! Delta: {delta}, out {cand_out}")
                        break
            
            if neighborhood == 'exchange':
                # Evaluate exchanges
                for cand_in in cl:
                    for cand_out in sol.elements:
                        delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, sol)
                        valid_move = True if ((cand_in not in self.TL and cand_out not in self.TL) or self.currentValue + delta > self.bestValue) else False
                        if delta > best_delta and valid_move:
                            # Check if this exchange would break feasibility
                            temp_sol = ScQbfSolution(sol.elements.copy())
                            temp_sol.elements.remove(cand_out)
                            temp_sol.elements.append(cand_in)
                            if self.evaluator.is_solution_valid(temp_sol):
                                best_delta = delta
                                best_cand_in = cand_in
                                best_cand_out = cand_out
                        if delta > 0 and valid_move:
                            move_found = True
                            if self.debug:
                                print(f"[local_search]: First improvement found (exchange)! Delta: {delta}, in {cand_in}, out {cand_out}")
                            break

            if move_found:
                break

        # Apply the best move found and update the Tabu List
        if best_cand_in is not None:
            sol.elements.append(best_cand_in)
            self.TL.append(best_cand_in)
        else:
            self.TL.append(self.FAKE)

        if best_cand_out is not None:
            sol.elements.remove(best_cand_out)
            self.TL.append(best_cand_out)
        else:
            self.TL.append(self.FAKE)
        
        self.TL = self.TL[2:]

        self.currentValue = self.evaluator.evaluate_objfun(sol)
        
        return sol
