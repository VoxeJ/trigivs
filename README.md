# Trigivs

A Rust library for solving **tridiagonal systems of linear equations** using **Givens rotations**. This library is designed to work in both `std` and `no_std` environments, with optional heap allocation.

---

## Features

The library supports three modes of operation, controlled by Cargo features:

| Feature | Description                                                                 |
|---------|-----------------------------------------------------------------------------|
| `std`   | Default mode. Uses standard heap allocation (`Vec<T>`).                   |
| `alloc` | Works in `no_std` environments but requires an allocator for heap usage.   |
| None    | Fully `no_std` mode. No heap allocation; uses fixed-size arrays only.     |

---

## Usage

Add `trigivs` to your `Cargo.toml`:

```toml
[dependencies]
trigivs = "0.1"
```

To use the library in `no_std` mode (no heap allocation):

```toml
[dependencies]
trigivs = { version = "0.1", default-features = false }
```

To use the library with heap allocation but without `std`:

```toml
[dependencies]
trigivs = { version = "0.1", default-features = false, features = ["alloc"] }
```

---

## Examples

### Compute the Determinant

```rust
use trigivs::prelude::compute_tridiag_determinant;

fn main() {
    let sup = [-4.0];
    let diag = [3.0, 2.0];
    let sub = [5.0];
    let determinant = compute_tridiag_determinant(&sup, &diag, &sub);
    println!("Determinant: {}", determinant);
}
```

---

### Solve a Tridiagonal System

```rust
use trigivs::prelude::solve_givens;

fn main() {
    let sup = [-4.0];
    let diag = [3.0, 2.0];
    let sub = [5.0];
    let rhs = [-3.0, 21.0];
    let x = solve_givens(&sup, &diag, &sub, &rhs).unwrap();
    println!("Solution: {:?}", x);
}
```

---

### Precompute a System for Multiple Right-Hand Sides

```rust
use trigivs::prelude::{precompute_givens, TridiagonalSystemPrecomputed};

fn main() {
    let sup = [-4.0];
    let diag = [3.0, 2.0];
    let sub = [5.0];
    let precomputed = precompute_givens(&sup, &diag, &sub).unwrap();

    let rhs1 = [-3.0, 21.0];
    let rhs2 = [-23.0, 5.0];

    let x1 = precomputed.solve_givens_rhs(&rhs1).unwrap();
    let x2 = precomputed.solve_givens_rhs(&rhs2).unwrap();

    println!("Solution 1: {:?}", x1);
    println!("Solution 2: {:?}", x2);
}
```

### Error Handling

`SolverErrors`:
* `InvalidDiagonals` - The superdiagonal and subdiagonal must be equal in length and exactly one element shorter than the main diagonal.
* `InvalidRhsSizing` - The right-hand side (RHS) must match the length of the main diagonal.
* `DivisionByZero`   - Division by zero was attempted during computation. 

---

## License

This project is licensed under the MIT License.
