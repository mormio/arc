import inspect
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from arc import REPO_ROOT
from arc.arcdsl import PRIMITIVES
from arc.arcdsl import solvers as solvers_mod

from .easy_subset import EASY_SUBSET
from .real.util import format_arc_challenges_for_dataset
from .util import get_solver, load_data


class REARCDataset(Dataset):
    def __init__(self, data_dict=None, task_dir=None, debug=False):
        self.data = []

        if task_dir is None:
            task_dir = os.path.join(REPO_ROOT, "data", "re_arc", "tasks")

        # if data dict is already loaded and passed, those keys are our problems
        if data_dict:
            problem_names = list(data_dict.keys())
        # otherwise, look in the re_arc/tasks folder for problems
        else:
            problem_names = [x.strip(".json") for x in os.listdir(task_dir)]
            if debug:
                problem_names = problem_names[:25]

        for problem in problem_names:
            if not data_dict:
                # load synthetic data if it's not provided
                data_path = os.path.join(task_dir, problem + ".json")
                with open(data_path, "r") as f:
                    problem_data = json.load(f)
            else:
                # we already have it, get it
                problem_data = data_dict[problem]

            # load the solver
            solver = get_solver(problem, solvers_mod)

            # if theres no solver then get_solver() returns None
            if solver:
                solver_source = inspect.getsource(solver)

                # check which primitives are in the solver
                # make it into the label vector
                label = [0] * len(PRIMITIVES)
                for i, prim in enumerate(PRIMITIVES):
                    if prim in solver_source:
                        label[i] = 1
            else:
                label = None

            # append all the examples to the training data
            for sample in problem_data:
                self.data.append(
                    (sample["input"], sample["output"], label, problem)
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_matrix, output_matrix, label_vector, problem_id = self.data[idx]
        return {
            "input": input_matrix,
            "output": output_matrix,
            "label": np.array(label_vector, dtype=np.float32),
            "problem_id": problem_id,
        }


class ARCDataset(Dataset):
    def __init__(
        self,
        challenges_data=None,
        task_dir=None,
        debug=False,
        split="train",
        easy=False,
    ):
        self.data = []

        if task_dir is None:
            task_dir = os.path.join(REPO_ROOT, "data", "re_arc", "tasks")

        # if data dict is already loaded and passed, those keys are our problems
        if not challenges_data:
            challenges_data = self.load_challenges_data(
                split=split, easy=easy, debug=debug
            )

        problem_names = list(challenges_data.keys())

        for problem in problem_names:
            problem_data = challenges_data[problem]

            # load the solver
            solver = get_solver(problem, solvers_mod)

            # if theres no solver then get_solver() returns None
            if solver:
                solver_source = inspect.getsource(solver)

                # check which primitives are in the solver
                # make it into the label vector
                label = [0] * len(PRIMITIVES)
                for i, prim in enumerate(PRIMITIVES):
                    if prim in solver_source:
                        label[i] = 1
            else:
                label = None

            # append all the examples to the training data
            for sample in problem_data:
                self.data.append(
                    (sample["input"], sample["output"], label, problem)
                )

    def load_challenges_data(self, split, easy, debug):
        challenges, _ = load_data(dataset="ARC", **{"split": split})
        if easy:
            challenges = {
                k: v for k, v in challenges.items() if k in EASY_SUBSET
            }
        if debug:
            challenges = challenges[:5]
        return format_arc_challenges_for_dataset(challenges)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_matrix, output_matrix, label_vector, problem_id = self.data[idx]
        return {
            "input": input_matrix,
            "output": output_matrix,
            "label": np.array(label_vector, dtype=np.float32),
            "problem_id": problem_id,
        }


class ARCDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        normalize=True,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.arc_collate_fn,
        )
        self.normalize = normalize

    @staticmethod
    def pad_matrix(matrix, max_height, max_width):
        h, w = matrix.shape
        return np.pad(
            matrix,
            ((0, max_height - h), (0, max_width - w)),
            mode="constant",
            constant_values=-1,
        )

    def arc_collate_fn(self, batch):
        inputs = [np.array(item["input"]) for item in batch]
        outputs = [np.array(item["output"]) for item in batch]

        # find max dims
        max_input_height = max(m.shape[0] for m in inputs)
        max_input_width = max(m.shape[1] for m in inputs)
        max_output_height = max(m.shape[0] for m in outputs)
        max_output_width = max(m.shape[1] for m in outputs)

        max_height = max(max_input_height, max_output_height)
        max_width = max(max_input_width, max_output_width)

        # pad + concatenate each input-output pair
        combined_matrices = []
        for input_matrix, output_matrix in zip(inputs, outputs):
            padded_input = self.pad_matrix(input_matrix, max_height, max_width)
            padded_output = self.pad_matrix(
                output_matrix, max_height, max_width
            )

            # normalize
            if self.normalize:
                padded_input = (padded_input + 1) / 10
                padded_output = (padded_output + 1) / 10

            # concatenate along the width dimension
            combined = np.concatenate([padded_input, padded_output], axis=1)
            combined = np.expand_dims(
                combined, axis=0
            )  # Add channel dimension

            combined_matrices.append(combined)

        # stack them into a batch
        combined_input = np.stack(combined_matrices)

        # Convert to torch tensor
        combined_input = torch.from_numpy(combined_input).float()
        labels = torch.tensor(
            np.array([item["label"] for item in batch]), dtype=torch.float
        )
        problem_ids = [item["problem_id"] for item in batch]

        return {
            "combined_input": combined_input,
            "labels": labels,
            "problem_ids": problem_ids,
        }
