#![cfg_attr(not(feature = "std"), no_std)]

//! # trigivs
//! 
//! This library provides a solver for tridiagonal systems of linear equations.
//! It's behavious is governed by several features
//!
//! # Features
//! 
//! - `std`: Default mode of operation. Uses heap allocation and `Vec<T>` outputs
//! - `alloc`: Works the same as `std`, but works in `no_std` environments and returns `Box<T>`

pub mod prelude {
    pub use crate::solver_error::SolverErrors;
    pub use crate::compute_tridiag_determinant;

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

#[cfg(test)]
mod tests;

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