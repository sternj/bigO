import ast
import atexit
import builtins
import hashlib
import inspect
import json
import time
import marshal
import random
import tracemalloc
import numpy as np

from collections import defaultdict
from functools import wraps
from scipy.stats import linregress

# Global dictionary to store performance data
performance_data = {}

performance_data_filename = "timespace_data.json"
performance_analysis_filename = "timespace_analysis.json"

wrapped_functions = set()

delay_factor = defaultdict(int)

def monkey_patch_function(obj, func_name):
    """
    Monkey patch the specified function in the given object.
    If factor > 0, the patched version will time the function and then sleep 
    factor times as long as it took to run.
    """
    original = obj[func_name]
    def wrapper(*args, **kwargs):
        global delay_factor
        delay = delay_factor[func_name]
        print(f"Found {delay=}")
        if delay > 0.0:
            start = time.time()
            ret = original(*args, **kwargs)
            elapsed = time.time() - start
            time.sleep(elapsed * delay)
        else:
            ret = original(*args, **kwargs)
        return ret

    # Set the wrapped function in place of the original
    print(f"MONKEY PATCH {obj=}")
    obj[func_name] = wrapper
    wrapped_functions.add(func_name)

    
class FunctionCallFinder(ast.NodeVisitor):
    def __init__(self, function_name):
        self.function_name = function_name
        self.in_target_function = False
        self.called_functions = []
        self.function_stack = []
        self.builtin_names = set(dir(builtins))  # All builtin identifiers
        self.builtin_names.add('track')

    def visit_FunctionDef(self, node):
        # Push current function name
        self.function_stack.append(node.name)

        if node.name == self.function_name:
            self.in_target_function = True

        self.generic_visit(node)

        # Pop and reset states if leaving target function
        popped = self.function_stack.pop()
        if popped == self.function_name:
            self.in_target_function = False

    def visit_Call(self, node):
        if self.in_target_function:
            func_node = node.func
            func_name = None
            if isinstance(func_node, ast.Name):
                # Direct function call: foo()
                func_name = func_node.id
            elif isinstance(func_node, ast.Attribute):
                # Method or attribute call: obj.method()
                func_name = func_node.attr

            # Check if the function name is not a builtin before adding
            if func_name is not None and func_name not in self.builtin_names:
                self.called_functions.append(func_name)
        self.generic_visit(node)

        
def set_performance_data_filename(fname):
    global performance_data_filename
    global performance_data
    performance_data_filename = fname
    try:
        with open(performance_data_filename, 'r') as infile:
            performance_data = json.load(infile)
    except FileNotFoundError:
        performance_data = {}
        pass

def track(length_computation):
    """
    A decorator to measure and store performance metrics of a function.

    Args:
        length_computation (callable): A function that calculates the "length"
                                       of one or more arguments.

    Returns:
        callable: The decorated function.
    """
    def decorator(func):
        # print(f"Tracking {func.__name__}")
        
        # Grab the source code and identify any functions invoked by this function.
        source = inspect.getsource(inspect.getmodule(func))
        tree = ast.parse(source)
        finder = FunctionCallFinder(func.__name__)
        finder.visit(tree)
        print(f"Functions called by {func.__name__}: {finder.called_functions}")
        for fn in finder.called_functions:
            monkey_patch_function(func.__globals__, fn)

        code = marshal.dumps(func.__code__)
        hash_value = hashlib.sha256(code).hexdigest()
        func_name = func.__name__
        file_name = inspect.getmodule(func).__file__
        full_name = str((func_name, file_name))
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Calculate the length based on the provided computation
            length = length_computation(*args, **kwargs)

            # Delay all the called functions.
            print(f"TIME TO DELAY {finder.called_functions}")
            global delay_factor
            delay = random.uniform(1.0, 2.0)
            for fn in finder.called_functions:
                delay_factor[fn] = delay
            
            # Start measuring time and memory
            start_time = time.perf_counter()
            tracemalloc.start()
            try:
                result = func(*args, **kwargs)
            finally:
                # Stop measuring memory
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Stop measuring time
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # Store the performance data
                if full_name not in performance_data:
                    performance_data[full_name] = []
                new_entry = {
                    "hash" : hash_value,
                    "length": length * delay,
                    "time": elapsed_time,
                    "memory": peak,  # Peak memory usage in bytes
                }
                print(new_entry)
                performance_data[full_name].append(new_entry)
                for fn in finder.called_functions:
                    delay_factor[fn] = 0

            return result
        return wrapper
    return decorator


def fit_complexity_curves(input_file, output_file):
    """
    Reads a JSON file, fits time and space complexity curves for each function,
    and writes the results to another JSON file.

    :param input_file: Path to the input JSON file
    :param output_file: Path to the output JSON file
    """
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    results = {}

    for func_name, measurements in data.items():
        if len(measurements) <= 1:
            continue
        lengths = [entry['length'] for entry in measurements]
        times = [entry['time'] for entry in measurements]
        memory = [entry['memory'] for entry in measurements]

        # Log-transform data for linear regression (assumes log-log relationship)
        log_lengths = np.log(lengths)
        log_times = np.log(times)
        log_memory = np.log(memory)

        # Fit time complexity
        slope_time, intercept_time, r_value_time, p_value_time, _ = linregress(log_lengths, log_times)
        
        # Fit memory complexity
        slope_memory, intercept_memory, r_value_memory, p_value_memory, _ = linregress(log_lengths, log_memory)

        print(func_name, slope_time, slope_memory)
        
        # Interpret slopes as complexity classes
        def interpret_complexity(slope):
            if np.isclose(slope, 0):
                return "O(1)"
            elif np.isclose(slope, 1):
                return "O(n)"
            elif np.isclose(slope, 2):
                return "O(n^2)"
            elif np.isclose(slope, np.log2(np.e)):
                return "O(log n)"
            elif np.isclose(slope, 1 + np.log2(np.e)):
                return "O(n log n)"
            else:
                return f"O(n^{slope:.2f})"

        time_complexity = interpret_complexity(slope_time)
        memory_complexity = interpret_complexity(slope_memory)

        # Store results
        results[func_name] = {
            "time_complexity": time_complexity,
            "time_complexity_r": r_value_time,
            "time_complexity_p": p_value_time,
            "memory_complexity": memory_complexity,
            "memory_complexity_r": r_value_memory,
            "memory_complexity_p": p_value_memory
        }

    # Write results to output file
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

# Example usage:
# fit_complexity_curves("input.json", "output.json")


@atexit.register
def save_performance_data():
    """
    Saves the collected performance data to a JSON file at program exit.
    """

    # Load any saved data into a dictionary.
    global performance_data_filename
    try:
        with open(performance_data_filename, 'r') as infile:
            old_data = json.load(infile)
    except FileNotFoundError:
        old_data = {}
        pass

    # Merge the old with the new dictionary
    for key, value_list in old_data.items():
        if key in performance_data:
            # Key exists in both dicts; extend the list from performance_data with the new entries
            performance_data[key].extend(value_list)
        else:
            # Key only exists in old_data; add it to performance_data
            performance_data[key] = value_list
    
    with open(performance_data_filename, "w") as f:
        json.dump(performance_data, f, indent=4)
    fit_complexity_curves(performance_data_filename, performance_analysis_filename)
