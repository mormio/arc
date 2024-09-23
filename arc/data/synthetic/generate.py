import random 
import inspect 
import copy 
import json 
import signal
import time
from functools import partial
from typing import List

from arc.data import load_data 
from arc.functions_library.functions import (
    flip_horizontal, 
    flip_vertical, 
    duplicate_cols,
    duplicate_rows, 
    rotate_90,
    shift_cols,
    shift_rows,
    transpose,
)
functions_library = [
    flip_vertical,
    flip_horizontal,
    duplicate_cols,
    duplicate_rows, 
    rotate_90,
    shift_cols,
    shift_rows, 
    transpose
]

synthetic_dataset_size = 800

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")

def apply_functions_with_timeout(n_functions_to_apply: int, 
                                 inputs_list: List[List[List[int]]], 
                                 timeout: int = 10) -> tuple[List[List[List[int]]], List[str]]:
    start_time = time.time()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set the alarm for 10 seconds

    used_functions = []
    
    try:
        chosen_fns = random.choices(functions_library, k=n_functions_to_apply)
        duplicated = False 
        
        for chosen_fn in chosen_fns:
            # only duplicate once - otherwise inputs can get too big 
            if 'duplicate' in chosen_fn.__name__:
                if duplicated:
                    continue 
                else:
                    duplicated = True 

            if time.time() - start_time > timeout:
                raise TimeoutException("Total execution time exceeded")

            try:
                sig = inspect.signature(chosen_fn)
                if len(sig.parameters) == 2:
                    param = 2 if 'duplicate' in chosen_fn.__name__ else random.randint(1, 15)
                    inputs_list = [chosen_fn(input_grid, param) for input_grid in inputs_list]
                else:
                    inputs_list = [chosen_fn(input_grid) for input_grid in inputs_list]
                
                if chosen_fn.__name__ not in used_functions:
                    used_functions.append(chosen_fn.__name__)
            
            except TimeoutException:
                raise
            except Exception as e:
                print(f"Error applying function {chosen_fn.__name__}: {str(e)}")
                continue 

    except TimeoutException:
        print(f"apply_functions() timed out after {timeout} seconds")
    finally:
        signal.alarm(0)  # Cancel the alarm

    return inputs_list, used_functions


def apply_functions(n_functions_to_apply, inputs_list):

    chosen_fns = random.choices(functions_library, k=n_functions_to_apply)
    used_functions = []
    
    for chosen_fn in chosen_fns:

        try:
            # check how many args it takes.
            sig = inspect.signature(chosen_fn)
            if len(sig.parameters) == 2:
                param = random.randint(1, 15) # scale transform up to the size of m 
                for i, m in enumerate(inputs_list):
                    inputs_list[i] = chosen_fn(m, param)

            else:
                for i, m in enumerate(inputs_list):
                    inputs_list[i] = chosen_fn(m)
            
            if chosen_fn.__name__ not in used_functions:
                used_functions.append(chosen_fn.__name__)
        
        except:
            continue 

    return inputs_list, used_functions

## get all matrices from the arc dataset 

train_x, train_y = load_data('train')
val_x, val_y = load_data('val')

all_matrices = []

train_problem_ids = list(train_x.keys())

for pid in train_problem_ids:
    for ex in train_x[pid]['train']:
        all_matrices.append(ex['input'])
    for ex in train_x[pid]['test']:
        all_matrices.append(ex['input'])
    for ex in train_y[pid]:
        all_matrices.append(ex)


val_problem_ids = list(val_x.keys())

for pid in val_problem_ids:
    for ex in val_x[pid]['train']:
        all_matrices.append(ex['input'])
    for ex in val_x[pid]['test']:
        all_matrices.append(ex['input'])
    for ex in val_y[pid]:
        all_matrices.append(ex)

synthetic_dataset = {}
i_prob = 0 

while i_prob < synthetic_dataset_size:
    print(f"i_prob: {i_prob}")

    # choose how many examples to show 
    k = random.randint(3, 4)

    # get input matrices 
    inputs_ = random.choices(all_matrices, k=k)
    inputs = copy.deepcopy(inputs_)
    
    # choose how many functions to apply (this can include the same one >1 time)
    n_functions_to_apply = random.randint(1, 5)

    # apply them 
    outputs, used_functions = apply_functions_with_timeout(n_functions_to_apply, inputs)
    problem_data = []

    # add to dataset 
    if len(used_functions) > 0:
        for input, output in zip(inputs, outputs):
            problem_data.append(
                {'input': input, 
                'output': output}
            )
        synthetic_dataset[i_prob] = {
            'samples': problem_data, 
            'functions': used_functions,
        }
        i_prob += 1 

        if i_prob % 100 == 0:
            with open('synthetic_data.json', 'w') as f:
                json.dump(synthetic_dataset, f)