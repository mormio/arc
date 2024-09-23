import os 
import json 
from arc.functions_library import FUNCTION_CLASSES

def load_synthetic_data():

    home_dir = os.environ['HOME']
    data_dir = os.path.join(home_dir, 'arc', 'arc', 'data', 'synthetic')
    with open(os.path.join(data_dir, 'synthetic_data.json'), 'r') as f:
        synthetic_data = json.load(f)
    
    return synthetic_data
