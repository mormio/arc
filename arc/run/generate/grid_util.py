import os
import random

from arc import REPO_ROOT
from arc.data.re_arc.util import load_data as load_rearc_data

REARC_PATH = os.path.join(REPO_ROOT, "data", "re_arc", "tasks")
REARC_PROBLEMS = [x.strip(".json") for x in os.listdir(REARC_PATH)]


def randomly_select_problem() -> str:
    return random.choice(REARC_PROBLEMS)


def load_problem_data(problem_id: str) -> dict:
    problem_data = load_rearc_data(problem_ids=[problem_id])
    return problem_data


def random_rearc_grid():
    problem_id = randomly_select_problem()
    problem_data = load_problem_data(problem_id=problem_id).get(problem_id)
    sample = random.choice(problem_data)
    sample = sample[random.choice(("input", "output"))]
    return sample
