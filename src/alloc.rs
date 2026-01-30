use crate::solver_error::SolverErrors;
use crate::solver_parts::*;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
use std::iter;

#[cfg(not(feature = "std"))]
use core::iter;

#[cfg(feature = "std")]
use std::mem::swap;

#[cfg(not(feature = "std"))]
use core::mem::swap;

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

#[inline]
fn compute_x<T: Float>(rhs: &[T], d: &[T], a: &[T], u: &[T]) -> Result<Vec<T>, SolverErrors> {
    let n = d.len();
    let mut x: Vec<T> = iter::repeat(T::zero()).take(n).collect();
    for i in (0..n).rev() {
        let mut sum = T::zero();
        if i < n - 1 {
            sum = a[i] * x[i + 1];
        }
        if n > 1 && i < n - 2 {
            sum = sum + u[i] * x[i + 2];
        }
        let rhs_sum = rhs[i] - sum;
        let di = d[i];
        if is_zero_eps_mag(di, rhs_sum) {
            return Err(SolverErrors::DivisionByZero);
        }
        x[i] = (rhs_sum) / di;
    }
    Ok(x)
}

/// Solves a trigiagonal system of linear equations with heap allocation
/// 
/// # Arguments 
/// 
/// * `sup` - superdiagonal slice
/// * `diag` - main diagonal slice
/// * `sub` - subdiagonal slice
/// * `rhs` - right hand side slice
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

    for i in 0..(n - 1) {
        let bi = sub[i];
        let di = d[i];

        if is_zero_eps_mag(bi, di) {
            continue;
        } else if is_zero_eps_mag(di, bi) {
            d[i] = sub[i];
            swap(&mut d[i + 1], &mut a[i]);
            rhs.swap(i, i + 1);
            if i < u.len() {
                u[i] = a[i + 1];
                a[i + 1] = T::zero();
            }
            continue;
        }

        let ai = a[i];
        let di1 = d[i + 1];
        let rhsi = rhs[i];
        let rhsi1 = rhs[i + 1];

        let c;
        let s;

        (d[i], a[i], d[i + 1], s, c) = rotate_primary(ai, di, bi, di1)?;
        (rhs[i], rhs[i + 1]) = rotate_rhs(rhsi, rhsi1, s, c);

        if i < u.len() {
            (u[i], a[i + 1]) = rotate_secondary(a[i + 1], s, c);
        }
    }
    compute_x(&rhs, &d, &a, &u)
}

/// Precomputes a system for multiple differenr RHS with heap allocation
/// 
/// # Arguments
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

    for i in 0..(n - 1) {
        let bi = sub[i];
        let di = d[i];

        if is_zero_eps_mag(bi, di) {
            sins_cosins[i] = (T::zero(), T::one().copysign(di));
            continue;
        } else if is_zero_eps_mag(di, bi) {
            sins_cosins[i] = (-T::one().copysign(bi), T::zero());
            continue;
        }

        let ai = a[i];
        let di1 = d[i + 1];

        let c;
        let s;

        (d[i], a[i], d[i + 1], s, c) = rotate_primary(ai, di, bi, di1)?;
        sins_cosins[i] = (s, c);
        
        if i < u.len() {
            (u[i], a[i + 1]) = rotate_secondary(a[i + 1], s, c);
        }
    }
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
    /// * `rhs` - right hand side slice
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

        if let Some(sins_cosins) = &self.sins_cosins{
            for (i, &(s, c)) in sins_cosins.iter().enumerate() {
                if s.abs() < T::epsilon() {
                    continue;
                } else if c.abs() < T::epsilon() {
                    rhsl.swap(i, i + 1);
                    continue;
                }
                (rhsl[i], rhsl[i + 1]) = rotate_rhs(rhsl[i], rhsl[i + 1], s, c);
            }
        }
        compute_x(
            &rhsl,
            &self.diag,
            &self.sup1.clone().unwrap_or(vec![]),
            &self.sup2.clone().unwrap_or(vec![])
        )
    }
}

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