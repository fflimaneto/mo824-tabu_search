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
                    "local_search_method": "best_improve",  # best_improve | first_improve | probabilistic
                    "diversification_method": "none",  # none | infrequent_elements,
                    "probabilistic_weighted": False  # True | False
                 },
                 time_limit_secs: float = None,
                 tenure: int = None,
                 patience: int = 5000,
                 freq_threshold: int = None,
                 debug: bool = False):
        
        
        self.instance = instance
        self.max_iterations = max_iterations
        self.config = config
        self.time_limit_secs = time_limit_secs
        self.patience = patience
        self.debug = debug
        self.solve_time = 0
        self.tenure = tenure
        self.TL = [self.FAKE for i in range(2 * tenure)]
        self.iterations = 0
        self.evaluator = ScQbfEvaluator(instance)
        self.bestValue = float("-inf")
        self.currentValue = 0
        self.frequecy = [0 for i in range(self.instance.n)]
        self.freq_threshold = freq_threshold if freq_threshold is not None else self.patience


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

            # if self.debug:
            #     print(f"Iteration {self.iterations}: Current value {self.evaluator.evaluate_objfun(sol)} and best value {self.evaluator.evaluate_objfun(best_sol)}")
            
            sol = self._local_search(sol)
            
            if (self.evaluator.evaluate_objfun(sol) > self.evaluator.evaluate_objfun(best_sol)):
                best_sol = sol
                self.bestValue = self.evaluator.evaluate_objfun(best_sol)
                current_patience = self.patience
                if self.debug:
                    print(f"New best value {self.bestValue:.3f} with solution {best_sol.elements} (iteration {self.iterations})...")
            else:
                if current_patience is not None and current_patience > 0:
                    current_patience -= 1
            
            self.solve_time = time.perf_counter() - start_time
            if self.time_limit_secs is not None and self.solve_time >= self.time_limit_secs:
                print(f"Time limit of {self.time_limit_secs} seconds reached, stopping Tabu Search.")
                break

            self._update_frequency_counter(sol)
            if current_patience == 0 and self.config["diversification_method"] == "infrequent_elements":
                if self.debug:
                    print(f"Patience exhausted. Best value {self.evaluator.evaluate_objfun(best_sol)}. Current best solution {best_sol.elements} (iteration {self.iterations})...")

                filtered_counter = [self.frequecy[i] for i in range(self.instance.n) if self.frequecy[i] < self.freq_threshold]

                if len(filtered_counter) > 0:
                    filtered_counter.sort()
                    freq_cut = filtered_counter[min(self.tenure, len(filtered_counter)-1)]
                    infrequent_elements = [i for i in range(self.instance.n) if self.frequecy[i] <= freq_cut]
                    random.shuffle(infrequent_elements)
                    infrequent_elements = infrequent_elements[:min(self.tenure, len(infrequent_elements))]
                    if self.debug:
                        print(f"Diversifying by adding {infrequent_elements} (freq <= {freq_cut})...")

                    # Adds the infrequent elements to the best solution and the Tabu List in a random order
                    sol.elements = best_sol.elements.copy()
                    for e in infrequent_elements:
                        if e not in sol.elements:
                            sol.elements.append(e)
                        self.TL.append(e)
                        self.TL.append(self.FAKE)
                    self.TL = self.TL[-2*self.tenure:]
                    self.currentValue = self.evaluator.evaluate_objfun(sol)

                current_patience = self.patience
                for i in range(self.instance.n):
                    self.frequecy[i] = self.frequecy[i] // 2
            
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


    def _local_search(self, starting_point: ScQbfSolution) -> ScQbfSolution:
        if self.config.get("local_search_method", False) == "best_improve":
            return self._local_search_best_improve(starting_point)
        elif self.config.get("local_search_method", False) == "first_improve":
            return self._local_search_first_improve(starting_point)
        elif self.config.get("local_search_method", False) == "probabilistic":
            return self._local_search_probabilistic(starting_point)
        
    def _local_search_probabilistic(self, starting_point: ScQbfSolution) -> ScQbfSolution:
        sol = ScQbfSolution(starting_point.elements.copy())
        cl = [i for i in range(self.instance.n) if i not in sol.elements]
        current_elements = sol.elements.copy()
        neighborhoods = ['insertion', 'removal', 'exchange']

        # Shuffle neighborhoods, candidates and current_elements
        random.shuffle(neighborhoods)
        random.shuffle(cl)
        random.shuffle(current_elements)

        # Set number of samples per neighborhood to be selected
        sample_size = min(25, len(cl))

        # Generate sampled candidates and save valid moves
        moves = []
        for neighborhood in neighborhoods:
            if neighborhood == 'insertion':
                # Get sample of candidates to insert
                sampled_in = random.sample(cl, sample_size) if len(cl) > sample_size else cl
                for cand_in in sampled_in:
                    # Evaluate insertions
                    delta = self.evaluator.evaluate_insertion_delta(cand_in, sol)
                    valid_move = (cand_in not in self.TL or self.currentValue + delta > self.bestValue)
                    if valid_move:
                        moves.append({
                            'neighborhood': neighborhood,
                            'elem_in': cand_in,
                            'elem_out': None,
                            'delta': delta
                        })
            if neighborhood == 'removal':
                # Get sample of candidates to remove
                sampled_out = random.sample(current_elements, sample_size) if len(current_elements) > sample_size else current_elements
                for cand_out in sampled_out:
                    # Evaluate removals
                    delta = self.evaluator.evaluate_removal_delta(cand_out, sol)
                    valid_move = (cand_out not in self.TL or self.currentValue + delta > self.bestValue)
                    temp_sol = ScQbfSolution(sol.elements.copy())
                    temp_sol.elements.remove(cand_out)
                    # Check if this removal would break feasibility
                    if valid_move and self.evaluator.is_solution_valid(temp_sol):
                        moves.append({
                            'neighborhood': neighborhood,
                            'elem_in': None,
                            'elem_out': cand_out,
                            'delta': delta
                        })
            if neighborhood == 'exchange':
                # Get sample of candidates to exchange
                sampled_in = random.sample(cl, sample_size) if len(cl) > sample_size else cl
                sampled_out = random.sample(current_elements, sample_size) if len(current_elements) > sample_size else current_elements
                for cand_in in sampled_in:
                    for cand_out in sampled_out:
                        # Evaluate exchanges
                        delta = self.evaluator.evaluate_exchange_delta(cand_in, cand_out, sol)
                        valid_move = ((cand_in not in self.TL and cand_out not in self.TL) or self.currentValue + delta > self.bestValue)
                        temp_sol = ScQbfSolution(sol.elements.copy())
                        temp_sol.elements.remove(cand_out)
                        temp_sol.elements.append(cand_in)
                        # Check if this exchange would break feasibility
                        if valid_move and self.evaluator.is_solution_valid(temp_sol):
                            moves.append({
                                'neighborhood': neighborhood,
                                'elem_in': cand_in,
                                'elem_out': cand_out,
                                'delta': delta
                            })

        if not moves:
            return sol

        # Choose a random move (can be probabilistic or totally random)
        if self.config.get("probabilistic_weighted", False):
            # Weighted sampling based on positive deltas
            deltas = [max(0, m['delta']) for m in moves]
            total = sum(deltas)
            if total > 0:
                # Calc probabilities
                probs = [d / total for d in deltas]
                idx = random.choices(range(len(moves)), weights=probs, k=1)[0]
                move = moves[idx]
            else:
                # Choose random move if all deltas are non-positive
                move = random.choice(moves)
        else:
            # Random choice among all valid moves
            move = random.choice(moves)

        # Apply the chosen move and update the Tabu List
        neighborhood = move['neighborhood']
        cand_in = move['elem_in']
        cand_out = move['elem_out']
        delta = move['delta']

        if cand_in is not None:
            sol.elements.append(cand_in)
            self.TL.append(cand_in)
        else:
            self.TL.append(self.FAKE)

        if cand_out is not None:
            sol.elements.remove(cand_out)
            self.TL.append(cand_out)
        else:
            self.TL.append(self.FAKE)

        self.TL = self.TL[2:]
        self.currentValue = self.evaluator.evaluate_objfun(sol)

        if self.debug:
            print(f"[local_search_probabilistic]: Move: {neighborhood}, in: {cand_in}, out: {cand_out}, delta: {delta:.3f}")

        return sol

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

        if self.debug:
            print(f"[local_search]: Best improvement found! Delta: {best_delta}, in {best_cand_in}, out {best_cand_out}")

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
        neighborhoods = ['insertion', 'removal', 'exchange']

        random.shuffle(cl)
        random.shuffle(current_elements)
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

        # Apply the first improving move (or best move in case all deltas <= 0) found and update the Tabu List
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


    def _update_frequency_counter(self, sol: ScQbfSolution):
        if self.config["diversification_method"] == "infrequent_elements":
            for elem in sol.elements:
                self.frequecy[elem] += 1 if self.frequecy[elem] < self.freq_threshold else 0
