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
    pub use crate::compute_solution_residual_norm;

    #[cfg(feature = "alloc")]
    pub use crate::alloc::{
        TridiagSysPrecomp, 
        precompute_givens, 
        precompute_givens_ruiz,
        solve_givens, 
        tridiag_iter_kaczmarz,
        solve_givens_ruiz_precond,
        compute_ruiz_scaling
    };

    #[cfg(not(feature = "alloc"))]
    pub use crate::no_alloc::{
        TridiagSysPrecomp, 
        precompute_givens,
        precompute_givens_ruiz,
        solve_givens,
        tridiag_iter_kaczmarz,
        solve_givens_ruiz_precond,
        compute_ruiz_scaling
    };
}

use num_traits::Float;

use crate::prelude::SolverErrors;

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
/// * `diag` - main diagonal slice
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
pub fn compute_tridiag_determinant<T: Float>(sup: &[T], diag: &[T], sub: &[T]) -> Result<T, SolverErrors>{
    if sup.len() != sub.len() || sup.len() + 1 != diag.len(){
        return Err(SolverErrors::InvalidDiagonals)
    }
    let mut dpp;
    let mut dp = T::one();
    let mut d = diag[0];
    for ((&a, &b), &c) in diag.iter().skip(1).zip(sup).zip(sub){
        dpp = dp;
        dp = d;
        d = a * dp - b * c * dpp;
    }
    Ok(d)
}

/// Computes solution euclidean norm
/// 
/// # Argumens
/// 
/// * `sup` - superdiagonal slice
/// * `diag` - main diagonal slice
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
/// 
/// let result = trigivs::prelude::solve_givens(&sup, &diag, &sub, &rhs).unwrap();
/// let norm = trigivs::prelude::compute_solution_residual_norm(
///     &sup, 
///     &diag, 
///     &sub, 
///     &rhs, 
///     &result
/// ).unwrap();
/// ```
/// 
pub fn compute_solution_residual_norm<T: Float>(sup: &[T], diag: &[T], sub: &[T], rhs: &[T], x: &[T]) -> Result<T, solver_error::SolverErrors>{
    if sup.len() != sub.len() || sup.len() + 1 != diag.len() || x.len() != diag.len(){
        return Err(solver_error::SolverErrors::InvalidDiagonals);
    } else if diag.len() != rhs.len() {
        return Err(solver_error::SolverErrors::InvalidRhsSizing);
    }
    let n = diag.len();
    if diag.len() == 1{
        return Ok((rhs[0] - diag[0] * x[0]).abs())
    }
    let mut sum = 
        (rhs[0] - diag[0] * x[0] - sup[0] * x[1]).powi(2) +
        (rhs[n-1] - diag[n-1]*x[n-1] - sub[n-2] * x[n-2]).powi(2);
    for i in 1..n-1{
        sum = sum + (rhs[i] - sub[i - 1] * x[i - 1] - diag[i] * x[i] - sup[i] * x[i + 1]).powi(2);
    }
    Ok(sum.sqrt())
}