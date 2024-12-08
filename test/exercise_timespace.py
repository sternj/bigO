import random
import timespace
import numpy as np

# timespace.set_performance_data_filename("perf.json")

def multiply(x, n):
    z = 1
    for i in range(500):
        z += 1
    return x * n

@timespace.track(lambda n, **kwargs: n)
def linear_function(n, multiplier=2):
    return [multiply(x, multiplier) for x in range(n)]

@timespace.track(lambda x, y: len(x) + len(y))
def linear_function_2(x, y):
    for i in range(999):
        z = x + y
    return x + y

@timespace.track(lambda x: len(x))
def nlogn_function(x):
    return sorted(x)

@timespace.track(lambda n: n)
def quadratic_function(n):
    print(f"quadratic function {n=}")
    x = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(100):
                q = multiply(x, 1.0)
                # x += 1
for i in range(10):
    print("do something.")
    quadratic_function(10) #  * random.randint(1, 10))
import sys
sys.exit(0)

# Example function calls
for i in range(10):
    print("example functions.")
    linear_function(1_000)
    # linear_function(random.randint(1,1_000_000))
for i in range(10):
    print("do something.")
    quadratic_function(10 * random.randint(1, 10))
for i in range(10):
    print("combine lists")
    linear_function_2(list(range(random.randint(100_000, 200_000))), list(range(random.randint(100_000, 200_000))))
for i in range(10):
    print("sort me")
    nlogn_function(list(np.random.rand(random.randint(100_000,1_000_000))))
