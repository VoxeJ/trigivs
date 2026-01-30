use crate::solver_error::SolverErrors;
use crate::solver_parts::*;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use num_traits::Float;

/// Holds precomputed tridiagonal system for working with multiple righ hand sides with heap allocation
/// 
/// # Example
/// 
/// ```
/// let sup = [-4.];
/// let diag = [3., 2.];
/// let sub = [5.];
/// 
/// let rhs1 = [-3., 21.];
/// let rhs2 = [-23., 5.];
/// 
/// let precomputed = trigivs::prelude::precompute_givens(&sup, &diag, &sub).unwrap();
/// 
/// let x1 = precomputed.solve_givens_rhs(&rhs1).unwrap();
/// let x2 = precomputed.solve_givens_rhs(&rhs2).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct TridiagonalSystemPrecomputed<T: Float> {
    diag: Vec<T>,
    sup1: Option<Vec<T>>,
    sup2: Option<Vec<T>>,

    sins_cosins: Option<Vec<(T, T)>>
}

/// Solves a trigiagonal system of linear equations with heap allocation
/// 
/// # Arguments 
/// 
/// * `sub` - subdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sup` - superdiagonal elements, length n-1
/// * `rhs` - right-hand side vector, length n
/// 
/// # Example
/// 
/// ```
/// let sup = [-4.];
/// let diag = [3., 2.];
/// let sub = [5.];
/// let rhs = [-3., 21.];
/// 
/// let x = trigivs::prelude::solve_givens(&sup, &diag, &sub, &rhs).unwrap();
/// ```
pub fn solve_givens<T: Float>(sup: &[T], diag: &[T], sub: &[T], rhs: &[T]) -> Result<Vec<T>, SolverErrors> {
    if sup.len() != sub.len() || sup.len() + 1 != diag.len() {
        return Err(SolverErrors::InvalidDiagonals);
    } else if diag.len() != rhs.len() {
        return Err(SolverErrors::InvalidRhsSizing);
    }

    let n = diag.len();

    let mut a = sup.to_vec();
    let mut d = diag.to_vec();
    let mut u = if n > 1 {vec![T::zero(); n-2]} else {vec![]};
    let mut rhs = rhs.to_vec();
    let mut x = vec![T::zero(); n];

    solve_givens_body(sub, &mut d, &mut a, &mut u, &mut rhs, &mut x)?;

    Ok(x)
}

/// Precomputes a system for multiple differenr RHS with heap allocation
/// 
/// # Arguments
/// 
/// * `sub` - subdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sup` - superdiagonal elements, length n-1
/// 
/// # Example
/// 
/// ```
/// let sup = [-4.];
/// let diag = [3., 2.];
/// let sub = [5.];
/// 
/// let rhs = [-3., 21.];
/// 
/// let precomputed = trigivs::prelude::precompute_givens(&sup, &diag, &sub).unwrap();
/// 
/// ```
pub fn precompute_givens<T: Float>(sup: &[T], diag: &[T], sub: &[T]) -> Result<TridiagonalSystemPrecomputed<T>, SolverErrors> {
    let n = diag.len();

    if sup.len() != sub.len() || sup.len() + 1 != diag.len() {
        return Err(SolverErrors::InvalidDiagonals);
    }
    let mut a = sup.to_vec();
    let mut d = diag.to_vec();
    let mut u  = if n > 1 {vec![T::zero(); n-2]} else {vec![]};

    let mut sins_cosins = vec![(T::zero(), T::zero()); n-1];

    precompute_givens_body(sub, &mut d, &mut a, &mut u, &mut sins_cosins)?;
    Ok(TridiagonalSystemPrecomputed {
        diag: d,
        sup1: if a.len() > 0 {Some(a)} else {None},
        sup2: if u.len() > 0 {Some(u)} else {None},
        sins_cosins: if sins_cosins.len() > 0 {Some(sins_cosins)} else {None},
    })
}

impl<T: Float> TridiagonalSystemPrecomputed<T> {

    /// Solves precomputed system with a provided right hand side with heap allocation
    /// 
    /// # Arguments
    /// 
    /// * `rhs` - right-hand side vector, length n
    /// 
    /// # Example
    /// 
    /// ```
    /// let sup = [-4.];
    /// let diag = [3., 2.];
    /// let sub = [5.];
    /// 
    /// let rhs = [-3., 21.];
    /// 
    /// let precomputed = trigivs::prelude::precompute_givens(&sup, &diag, &sub).unwrap();
    /// 
    /// let x = precomputed.solve_givens_rhs(&rhs).unwrap();
    /// ```
    pub fn solve_givens_rhs(&self, rhs: &[T]) -> Result<Vec<T>, SolverErrors> {
        let mut rhsl = rhs.to_vec();

        if rhsl.len() != self.diag.len() {
            return Err(SolverErrors::InvalidRhsSizing);
        }

        let mut x = vec![T::zero(); self.diag.len()];

        if let Some(sins_cosins) = &self.sins_cosins {
            solve_givens_sc_rhs_body(sins_cosins, &mut rhsl);
        }
        
        compute_x(
            &mut x,
            &rhsl,
            &self.diag,
            &self.sup1.clone().unwrap_or(vec![]),
            &self.sup2.clone().unwrap_or(vec![])
        )?;
        Ok(x)
    }
}

/// Refines a tridiagonal system solution using the Kaczmarz iterative method with heap allocation
/// 
/// # Arguments
///
/// * `sub` - subdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sup` - superdiagonal elements, length n-1
/// * `rhs` - right-hand side vector, length n
/// * `x_init` - initial solution approximation, length n
/// * `iter` - maximum number of Kaczmarz iterations to perform
/// * `eps` - convergence tolerance
/// 
/// # Example
///
/// ```
/// let sub = [5.];
/// let diag = [3., 2.];
/// let sup = [-4.];
/// let rhs = [-3., 21.];
/// 
/// let x_init = [1., 1.];
/// let x_refined = trigivs::prelude::refine_tridiag_solution_iter_kaczmarz(
///     &sub, 
///     &diag, 
///     &sup, 
///     &rhs, 
///     &x_init, 
///     1000, 
///     1e-6 
/// ).unwrap();
/// ```
/// 
pub fn refine_tridiag_solution_iter_kaczmarz<T: Float>(sub: &[T], diag: &[T], sup: &[T], rhs: &[T], x_init: &[T], iter: usize, eps: T) -> Result<Vec<T>, SolverErrors>{
    let mut x = x_init.to_vec();
    let n = x.len();
    let ai_ai_dotproducts = if n > 1 {
        let mut prod = vec![diag[0].powi(2) + sup[0].powi(2)];
        prod.extend((1..n-1).map(|i| diag[i].powi(2) + sup[i].powi(2) + sub[i - 1].powi(2)));
        prod.push(sub[n-2].powi(2) + diag[n-1].powi(2));
        prod
    } else {
        vec![diag[0].powi(2)]
    };
    kaczmarz_body(sub, diag, sup, rhs, &ai_ai_dotproducts, &mut x, n, iter, eps)?;
    Ok(x)
}