from arc.functions_library import functions_to_vector
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class ARCSyntheticDataset(Dataset):
    def __init__(self, synthetic_data):
        self.data = []

        for problem_id, problem_data in synthetic_data.items():
            for sample in problem_data["samples"]:
                input_matrix = sample["input"]
                output_matrix = sample["output"]
                label_vector = functions_to_vector(problem_data["functions"])
                self.data.append(
                    (input_matrix, output_matrix, label_vector, problem_id)
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


class ArcSyntheticDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.arc_collate_fn,
        )

    @staticmethod
    def pad_matrix(matrix, max_height, max_width):
        h, w = matrix.shape
        return np.pad(
            matrix,
            ((0, max_height - h), (0, max_width - w)),
            mode="constant",
            constant_values=-1,
        )

    @classmethod
    def arc_collate_fn(cls, batch):
        inputs = [np.array(item["input"]) for item in batch]
        outputs = [np.array(item["output"]) for item in batch]

        # find max dims
        max_input_height = max(m.shape[0] for m in inputs)
        max_input_width = max(m.shape[1] for m in inputs)
        max_output_height = max(m.shape[0] for m in outputs)
        max_output_width = max(m.shape[1] for m in outputs)

        max_height = max(max_input_height, max_output_height)
        max_width = max(
            max_input_width, max_output_width
        )  # supports concatenation

        # pad + concatenate each input-output pair
        combined_matrices = []
        for input_matrix, output_matrix in zip(inputs, outputs):
            padded_input = cls.pad_matrix(input_matrix, max_height, max_width)
            padded_output = cls.pad_matrix(
                output_matrix, max_height, max_width
            )

            # concatenate along the width dimension
            combined = np.concatenate([padded_input, padded_output], axis=1)
            combined_matrices.append(combined)

        # stack them into a batch
        combined_input = np.stack(combined_matrices)

        # Convert to torch tensor
        combined_input = torch.from_numpy(combined_input).float()

        labels = torch.tensor(
            [item["label"] for item in batch], dtype=torch.float
        )
        problem_ids = [item["problem_id"] for item in batch]

        return {
            "combined_input": combined_input,
            "labels": labels,
            "problem_ids": problem_ids,
        }
