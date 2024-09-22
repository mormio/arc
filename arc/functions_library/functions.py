from typing import List

def flip_vertical(input: List[List[int]]) -> List[List[int]]:
    """
    Flips the input grid vertically.
    """
    return input[::-1]

def duplicate_cols(input: List[List[int]], num_duplicates: int) -> List[List[int]]:
    """
    Duplicates each column in the input grid the specified number of times.
    """
    return transpose(duplicate_rows(transpose(input), num_duplicates))

def flip_horizontal(input: List[List[int]]) -> List[List[int]]:
    """
    Flips the input grid horizontally.
    """
    return [row[::-1] for row in input]

def transpose(input: List[List[int]]) -> List[List[int]]:
    """
    Transposes the input grid.
    """
    return list(map(list, zip(*input)))

def rotate_90(input: List[List[int]]) -> List[List[int]]:
    """
    Rotates the input grid 90 degrees clockwise.
    """
    return list(zip(*input[::-1]))

def shift_cols(input: List[List[int]], shift: int) -> List[List[int]]:
    """
    Shifts each column in the input grid by the specified number of positions.
    """
    return transpose(shift_rows(transpose(input), shift))

def shift_rows(input: List[List[int]], shift: int) -> List[List[int]]:
    """
    Shifts each row in the input grid by the specified number of positions.
    """
    return [row[-shift:] + row[:-shift] for row in input]

def duplicate_rows(input: List[List[int]], num_duplicates: int) -> List[List[int]]:
    """
    Duplicates each row in the input grid the specified number of times.
    """
    return [row for row in input for _ in range(num_duplicates)]

