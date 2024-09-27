import json
import os
from typing import List

from arc import REPO_ROOT


def load_data(problem_ids: List[str] = None, n_problems: int = None):
    """Returns a dictionary like {problem_id: [{input: [...], output: [...]}, {input: [...], output: [...]}]}"""
    task_dir = os.path.join(REPO_ROOT, "data", "re_arc", "tasks")

    data_dict = {}

    if not problem_ids:
        problem_ids = [x.strip(".json") for x in os.listdir(task_dir)]
        if n_problems:
            problem_ids = problem_ids[:n_problems]
            print(f"Loading a subset of {n_problems}.")
        else:
            print("No list of problem_ids passed, loading all 400...")

    for problem in problem_ids:
        suffix = ".json" if not problem.endswith(".json") else ""
        data_path = os.path.join(task_dir, problem + suffix)

        with open(data_path, "r") as f:
            problem_data = json.load(f)

        data_dict[problem.strip(".json")] = problem_data

    return data_dict
