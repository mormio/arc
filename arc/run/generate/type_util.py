import random

from grid_util import random_rearc_grid as random_grid

from arc.arcdsl.arc_types import *
from arc.arcdsl.constants import *


def random_boolean() -> Boolean:
    return random.choice([True, False])


def random_integer(min_val: int = -2, max_val: int = 10) -> Integer:
    return random.randint(min_val, max_val)


def random_integer_tuple() -> IntegerTuple:
    return (random_integer(), random_integer())


def random_numerical() -> Numerical:
    return random.choice([random_integer(), random_integer_tuple()])


def random_integer_set(size: int = 5) -> IntegerSet:
    return frozenset(random_integer() for _ in range(size))


def random_cell() -> Cell:
    return (random_integer(), random_integer_tuple())


def random_object(size: int = 3) -> Object:
    return frozenset(random_cell() for _ in range(size))


def random_objects(size: int = 2) -> Objects:
    return frozenset(random_object() for _ in range(size))


def random_indices(size: int = 3) -> Indices:
    return frozenset(random_integer_tuple() for _ in range(size))


def random_indices_set(size: int = 2) -> IndicesSet:
    return frozenset(random_indices() for _ in range(size))


def random_patch() -> Patch:
    return random.choice([random_object(), random_indices()])


def random_element() -> Element:
    return random.choice([random_object(), random_grid()])


def random_piece() -> Piece:
    return random.choice([random_grid(), random_patch()])


def random_tuple_tuple(outer: int = 2, inner: int = 2) -> TupleTuple:
    return tuple(
        tuple(random_integer() for _ in range(inner)) for _ in range(outer)
    )


def random_container_container() -> ContainerContainer:
    container_types = [list, tuple, set, frozenset]
    outer_type = random.choice(container_types)
    inner_type = random.choice(container_types)
    return outer_type(
        inner_type(random_integer() for _ in range(3)) for _ in range(2)
    )


RANDOM_CREATORS = {
    Boolean: random_boolean,
    Integer: random_integer,
    IntegerTuple: random_integer_tuple,
    Numerical: random_numerical,
    IntegerSet: random_integer_set,
    Cell: random_cell,
    Object: random_object,
    Objects: random_objects,
    Indices: random_indices,
    IndicesSet: random_indices_set,
    Patch: random_patch,
    Element: random_element,
    Piece: random_piece,
    TupleTuple: random_tuple_tuple,
    ContainerContainer: random_container_container,
}
