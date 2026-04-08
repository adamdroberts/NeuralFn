# neuralfn.port

Declares the `Port` dataclass, which specifies an input or output slot on a neuron.

## Class: Port

```python
@dataclass(frozen=True)
class Port:
    name: str
    range: tuple[float, float] = (-1.0, 1.0)
    precision: float = 0.001
    dtype: str = "float"
```

A `Port` is immutable (frozen). It defines the name, value range, quantization precision, and semantic type of a single connection point on a neuron.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable identifier, unique within a neuron |
| `range` | `tuple[float, float]` | `(-1.0, 1.0)` | `(low, high)` bounds for values on this port |
| `precision` | `float` | `0.001` | Smallest meaningful step; values are quantized to this resolution |
| `dtype` | `str` | `"float"` | Semantic type hint: `"float"`, `"int"`, `"bool"`, `"tensor"`, `"tokens"`, or `"loss"` |

### Validation

`__post_init__` raises `ValueError` if:

- `range[0] >= range[1]` (low must be strictly less than high)
- `precision <= 0`

### Methods

#### `clamp(value: float) -> float`

Clamp `value` to the port's `[low, high]` range.

```python
port = Port("x", range=(0.0, 1.0))
port.clamp(1.5)   # 1.0
port.clamp(-0.5)  # 0.0
```

#### `quantize(value: float) -> float`

Round `value` to the nearest multiple of `precision`.

```python
port = Port("x", precision=0.01)
port.quantize(0.123456)  # 0.12
```

#### `condition(value: float) -> float`

Clamp then quantize. This is the standard conditioning applied when a value flows through an edge into a port.

```python
port = Port("x", range=(0.0, 1.0), precision=0.1)
port.condition(1.37)  # 1.0 (clamped to 1.0, then quantized)
```

#### `to_dict() -> dict`

Serialize the port to a JSON-compatible dictionary.

```python
port.to_dict()
# {"name": "x", "range": [0.0, 1.0], "precision": 0.001, "dtype": "float"}
```

#### `Port.from_dict(d: dict) -> Port` *(classmethod)*

Reconstruct a `Port` from a serialized dictionary. Missing keys use defaults (`precision=0.001`, `dtype="float"`).

```python
port = Port.from_dict({"name": "x", "range": [-1.0, 1.0]})
```
