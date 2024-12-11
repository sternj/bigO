## bigO

### `test/facts.py`

Here is a demonstration program.

```python
import bigO

def fact(x: int) -> int:
  v = 1
  for i in range(x):
    v *= i
  return v

@bigO.track(lambda xs: len(xs))
def factorialize(xs: list[int]) -> list[int]:
  new_list = [fact(x) for x in xs]
  return new_list

# Exercise the same input
for i in range(30):
    factorialize([i for i in range(1000)])
```

### Run the program

```bash
python3 test/factorialize.py
```

Running this code produces two files in the current directory:
1. `bigO_data.json`
2. `bigO_analysis.json`

### Get a graph

```bash
python3 -m bigO.graph
```

Creates the file `bigO.pdf`.

