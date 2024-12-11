## bigO

`bigO` automatically measures empirical computational complexity (in both time and space) of functions.
To use `bigO`, you just need to decorate your function with a "length" function (the "n"),
and then run your function as usual, ideally along a wide range of inputs.
As long as your function calls another function, `bigO` will automatically generate diversity,
so you can get computational complexity information even when you aren't easily able to generate a wide range of inputs to the function.
`bigO` accumulates in a file `bigO_data.json` in the local directory; you can then generate a graph of time and space complexity for each tracked function.

### Demonstration

The file `test/facts.py` is a small program that demonstrates `bigO`.

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

# Exercise the function - more inputs are better!
for i in range(30):
    factorialize([i for i in range(i * 100)])
```

Now run the program as usual:

```bash
python3 test/facts.py
```

Now you can easily generate a graph of all tracked functions. Just run the following command in the same directory.

```bash
python3 -m bigO.graph
```

This command creates the file `bigO.pdf` that contains graphs like this:

![bigO](https://github.com/user-attachments/assets/8428180b-a454-4fc7-822c-7a130f9ba54e)
