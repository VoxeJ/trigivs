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

```rust
use trigivs::prelude::*;

fn main() {
    // Define a 3x3 tridiagonal system
    let sub = [2.0, 1.0];          // subdiagonal (length n-1)
    let diag = [3.0, 4.0, 5.0];    // main diagonal (length n)
    let sup = [1.0, 2.0];          // superdiagonal (length n-1)
    
    // 1. Compute determinant
    let det = compute_tridiag_determinant(&sup, &diag, &sub);
    println!("Determinant: {}", det);

    // 2. Solve directly
    let rhs = [1.0, 2.0, 3.0];
    let solution = solve_givens(&sup, &diag, &sub, &rhs).unwrap();
    println!("Direct solution: {:?}", solution);
    
    // 3. Precompute for multiple RHS
    let precomputed = precompute_givens(&sup, &diag, &sub).unwrap();
    let rhs2 = [4.0, 5.0, 6.0];
    let solution2 = precomputed.solve_givens_rhs(&rhs2).unwrap();
    println!("Solution with precomputed: {:?}", solution2);
    
    // 4. Refine solution iteratively
    // NOTE: Meant for large ill-defined systems or as rare fallback
    let refined = refine_tridiag_solution_iter_kaczmarz(
        &sub, &diag, &sup, &rhs, &solution, 50, 1e-10
    ).unwrap();
    println!("Refined solution: {:?}", refined);
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
