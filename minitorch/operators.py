"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable, Union, List, TypeVar


# Task 0.1

Number = Union[float, int]

def mul(a: Number, b: Number) -> Number:
    return a * b

def add(a: Number, b: Number) -> Number:
    return a + b

def neg(a: Number) -> Number:
    return -a

def id(a: Number) -> Number:
    return a

def lt(a: Number, b: Number) -> bool:
    return a < b

def eq(a: Number, b: Number) -> bool:
    return a == b

def max(a: Number, b: Number) -> Number:
    return a if a > b else b

def is_close(a: Number, b: Number, tolerance: float = 1e-2) -> bool:
    return abs(a - b) < tolerance

def sigmoid(a: Number) -> float:
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        exp_a = math.exp(a)
        return exp_a / (1.0 + exp_a)

def relu(a: Number) -> Number:
    return max(a, 0)

def log(a: Number) -> float:
    return math.log(a)

def exp(a: Number) -> float:
    return math.exp(a)

def inv(a: Number) -> float:
    return 1 / a

def log_back(a: Number, d: Number) -> float:
    return d * (1 / a)

def inv_back(a: Number, d: Number) -> float:
    return d * (-1 / (a ** 2))

def relu_back(a: Number, d: Number) -> Number:
    return d if a > 0 else 0

# Task 0.2

def sigmoid_back(a: Number, d: Number) -> float:
    s = sigmoid(a)
    return d * s * (1 - s)


def transitive_lt(a: Number, b: Number, c: Number) -> bool:
    return not (lt(a, b) and lt(b, c)) or lt(a, c)


def symmetric_eq(a: Number, b: Number) -> bool:
    return eq(a, b) == eq(b, a)


def distribute_mul_over_add(a: Number, b: Number, c: Number) -> bool:
    return is_close(mul(a, add(b, c)), add(mul(a, b), mul(a, c)))


def other_properties(a: Number, b: Number) -> bool:
    return is_close(add(a, b), add(b, a))

# Task 0.3

def map(f: Callable, lst: Iterable) -> List:
    return [f(x) for x in lst]

def zipWith(f: Callable, lst1: Iterable, lst2: Iterable) -> List:
    return [f(a, b) for a, b in zip(lst1, lst2)]

def reduce(f: Callable, lst: Iterable, initial=None):
    it = iter(lst)
    if initial is None:
        initial = next(it)
    result = initial
    for x in it:
        result = f(result, x)
    return result


def addLists(lst1: Iterable[Number], lst2: Iterable[Number]) -> List[Number]:
    return zipWith(add, lst1, lst2)

def negList(lst: Iterable[Number]) -> List[Number]:
    return map(neg, lst)

def sum(lst: Iterable[Number]) -> Number:
    return reduce(add, lst, 0.0)

def prod(lst: Iterable[Number]) -> Number:
    result = 1
    for x in lst:
        result *= x
    return result

def sum_distribute(lst1: Iterable[Number], lst2: Iterable[Number]) -> bool:
    left = sum(lst1) + sum(lst2)
    right = sum(addLists(lst1, lst2))
    return is_close(left, right)
