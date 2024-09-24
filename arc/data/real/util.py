import json 
import os 

TEST_CHALLENGES_PATH = 'arc-agi_test_challenges.json'
VAL_CHALLENGES_PATH = 'arc-agi_evaluation_challenges.json'
VAL_SOLUTIONS_PATH = 'arc-agi_evaluation_solutions.json'
TRAIN_CHALLENGES_PATH = 'arc-agi_training_challenges.json'
TRAIN_SOLUTIONS_PATH = 'arc-agi_training_solutions.json'


def load_data(split):

    # assumes data is in the /data folder of this repo 
    data_dir = os.path.dirname(os.path.abspath(__file__))

    if split == 'test': # challenges only 
        with open(os.path.join(data_dir, TEST_CHALLENGES_PATH), 'r') as f:
            challenges = json.load(f)
        solutions = None 

    else: # challenges and solutions 
        if split == 'val':
            challenges_path, solutions_path = VAL_CHALLENGES_PATH, VAL_SOLUTIONS_PATH
        elif split == 'train':
            challenges_path, solutions_path = TRAIN_CHALLENGES_PATH, TRAIN_SOLUTIONS_PATH
        else:
            print("Split must be one of train, val, test.")
            return None 
        
        # Load 
        with open(os.path.join(data_dir, challenges_path), 'r') as f:
            challenges = json.load(f)
        with open(os.path.join(data_dir, solutions_path), 'r') as f:
            solutions = json.load(f)
    
    return challenges, solutions

def extract_train_samples(dataset, problem_id):
    return dataset[problem_id]['train']

def create_train_string(dataset, problem_id):
    data = extract_train_samples(dataset, problem_id)
    string = "" 

    for i, example in enumerate(data):
        string += f"Input {i+1}: {example['input']}\nOutput {i+1}: {example['output']}\n\n"

    return string

