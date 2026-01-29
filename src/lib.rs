#![cfg_attr(not(feature = "std"), no_std)]

//! # trigivs
//! 
//! This library provides a solver for tridiagonal systems of linear equations that works through Givens rotations.
//! It's behavious is governed by several features
//!
//! # Modes of operation
//! 
//! - `std`: Default mode. Uses standard heap allocation.
//! - `alloc`: The same as `std`, but works in `no_std` environments and requires a supplied allocator.
//! - `no_std`: Achieved through disabling all default features. Does not use heap allocation.

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::solver_error::SolverErrors;
    pub use crate::compute_tridiag_determinant;
    pub use crate::compute_solution_norm;

    #[cfg(feature = "alloc")]
    pub use crate::alloc::{TridiagonalSystemPrecomputed, precompute_givens, solve_givens};

    #[cfg(not(feature = "alloc"))]
    pub use crate::no_alloc::{TridiagonalSystemPrecomputed, precompute_givens, solve_givens};
}

use num_traits::Float;

/// Provides possible errors
/// 
pub mod solver_error;
mod solver_parts;

/// Provides functionality with heap allocation
/// 
#[cfg(any(doc, feature = "alloc", feature = "std"))]
pub mod alloc;

/// Provides functionality without heap-allocation
/// 
#[cfg(any(doc, not(any(feature = "alloc", feature = "std"))))]
pub mod no_alloc;

/// Computes a determinant of a tridiagonal matrix.
///
/// # Argumens
/// 
/// * `sup` - superdiagonal slice
/// * 'diag` - main diagonal slice
/// * `sub` - subdiagonal slice
/// 
/// # Example
///
/// ```
/// let sup = [-4.];
/// let diag = [3., 2.];
/// let sub = [5.];
/// let determinant = trigivs::prelude::compute_tridiag_determinant(&sup, &diag, &sub);
/// ```
pub fn compute_tridiag_determinant<T: Float>(sup: &[T], diag: &[T], sub: &[T]) -> T{
    let mut dpp;
    let mut dp = T::one();
    let mut d = diag[0];
    for ((&a, &b), &c) in diag.iter().skip(1).zip(sup).zip(sub){
        dpp = dp;
        dp = d;
        d = a * dp - b * c * dpp;
    }
    d
}

/// Computes solution inf norm
/// 
/// # Argumens
/// 
/// * `sup` - superdiagonal slice
/// * 'diag` - main diagonal slice
/// * `sub` - subdiagonal slice
/// * `rhs` - right-hand part slice
/// * `x` - solution slice
/// 
/// # Exmple
/// 
/// ```
/// let sup = [-4.];
/// let diag = [3., 2.];
/// let sub = [5.];
/// let rhs = [-3., 21.];
/// let rhs_wrong = [-1., 21.];
/// let result = trigivs::prelude::solve_givens(&sup, &diag, &sub, &rhs).unwrap();
/// let norm = trigivs::prelude::compute_solution_norm(&sup, &diag, &sub, &rhs_wrong, &result).unwrap();
/// ```
/// 
pub fn compute_solution_norm<T: Float>(sup: &[T], diag: &[T], sub: &[T], rhs: &[T], x: &[T]) -> Result<T, solver_error::SolverErrors>{
    if sup.len() != sub.len() || sup.len() + 1 != diag.len(){
        return Err(solver_error::SolverErrors::InvalidDiagonals);
    } else if diag.len() != rhs.len() {
        return Err(solver_error::SolverErrors::InvalidRhsSizing);
    }
    let n = diag.len();
    if diag.len() == 1{
        return Ok((rhs[0] - diag[0] * x[0]).abs())
    }
    let mut nmax;
    let mut max = (rhs[0] - diag[0] * x[0] - sup[0] * x[1]).abs();
    for i in 1..n-1{
        nmax = (rhs[i] - sub[i - 1] * x[i - 1] - diag[i] * x[i] - sup[i] * x[i + 1]).abs();
        if nmax > max{
            max = nmax;
        }
    }
    nmax = (rhs[n-1] - diag[n-1]*x[n-1] - sub[n-2] * x[n-2]).abs();
    if nmax > max{
        max = nmax;
    }
    Ok(max)
}