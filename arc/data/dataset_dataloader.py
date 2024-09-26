import inspect
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from arc.arcdsl import PRIMITIVES
from arc.arcdsl import solvers as solvers_mod
from arc.data import get_solver


class REARCDataset(Dataset):
    def __init__(self, task_dir, debug=False):
        self.data = []
        problem_names = [x.strip(".json") for x in os.listdir(task_dir)]
        if debug:
            problem_names = problem_names[:25]
        for problem in problem_names:
            # load synthetic data
            data_path = os.path.join(task_dir, problem + ".json")
            with open(data_path, "r") as f:
                problem_data = json.load(f)

            # load the solver
            solver = get_solver(problem, solvers_mod)
            solver_source = inspect.getsource(solver)

            # check which primitives are in the solver
            # make it into the label vector
            label = [0] * len(PRIMITIVES)
            for i, prim in enumerate(PRIMITIVES):
                if prim in solver_source:
                    label[i] = 1

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


class ARCDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        normalize=True,
        channels=1,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.arc_collate_fn,
        )
        self.normalize = normalize
        self.channels = channels

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
