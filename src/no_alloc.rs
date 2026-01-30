use crate::solver_error::SolverErrors;
use crate::solver_parts::*;

use num_traits::Float;

#[cfg(feature = "std")]
use std::mem::swap;

#[cfg(not(feature = "std"))]
use core::mem::swap;

/// A struct for holding tridiagonal system precomputed for working with multiple righ hand sides without heap allocation
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
pub struct TridiagonalSystemPrecomputed<T: Float, const D: usize, const S: usize> {
    diag: [T; D],
    sup1: Option<[T; S]>,
    sup2: Option<[T; S]>,

    sins_cosins: Option<[(T, T); S]>
}

#[inline]
fn compute_x<T: Float, const D: usize>(rhs: &[T], d: &[T], a: &[T], u: &[T]) -> Result<[T; D], SolverErrors> {
    let mut x = [T::zero(); D];
    for i in (0..D).rev() {
        let mut sum = T::zero();
        if i < D - 1 {
            sum = a[i] * x[i + 1];
        }
        if D > 1 && i < D - 2 {
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

/// Solves a trigiagonal system of linear equations without heap allocation
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
pub fn solve_givens<T: Float, const D: usize, const S: usize>(sup: &[T; S], diag: &[T; D], sub: &[T; S], rhs: &[T; D]) -> Result<[T; D], SolverErrors> 
{
    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut a = sup.clone();
    let mut d = diag.clone();
    let mut rhs = rhs.clone();

    let mut ur = [T::zero(); S];
    let u = if S > 1 {&mut ur[..S-1]} else {&mut []};

    for i in 0..(D - 1) {
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

/// Precomputes a system for multiple differenr RHS without heap allocation
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
pub fn precompute_givens<T: Float, const D: usize, const S: usize>(sup: &[T; S], diag: &[T; D], sub: &[T; S]) -> Result<TridiagonalSystemPrecomputed<T, D, S>, SolverErrors> {

    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut a = sup.clone();
    let mut d = diag.clone();

    let mut ur = [T::zero(); S];
    let u = if S > 1 {&mut ur[..S-1]} else {&mut []};

    let mut sins_cosins = [(T::zero(), T::zero()); S];

    for i in 0..(D - 1) {
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
        sup2: if u.len() > 0 {Some(ur)} else {None},
        sins_cosins: if sins_cosins.len() > 0 {Some(sins_cosins)} else {None},
    })
}

impl<T: Float, const D: usize, const S: usize> TridiagonalSystemPrecomputed<T, D, S> {

    /// Solves precomputed system with a provided right hand side without heap allocation
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
    pub fn solve_givens_rhs(&self, rhs: &[T; D]) -> Result<[T; D], SolverErrors> {
        let mut rhsl = rhs.clone();

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
            self.sup1.as_ref().map_or(&[], |v| v.as_slice()),
            self.sup2.as_ref().map_or(&[], |v| v.as_slice()),
        )
    }
}

pub fn refine_tridiag_solution_iter_kaczmarz<T: Float, const D: usize, const S: usize>(sub: &[T; S], diag: &[T; D], sup: &[T; S], rhs: &[T; D], x_init: &[T; D], iter: usize, eps: T) -> Result<[T; D], SolverErrors>{
    
    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut x = x_init.clone();
    let mut ai_ai_dotproducts = [T::zero(); D];
    if D == 1 {
        ai_ai_dotproducts[0] = diag[0].powi(2);
    } else {
        ai_ai_dotproducts[0] = diag[0].powi(2) + sup[0].powi(2);
        ai_ai_dotproducts
            .iter_mut().enumerate()
            .skip(1).take(D - 2)
            .for_each(|(i, elem)| *elem = diag[i].powi(2) + sup[i].powi(2) + sub[i - 1].powi(2));
        ai_ai_dotproducts[D-1] = sub[D - 2].powi(2) + diag[D - 1].powi(2)
    }
    kaczmarz_body(sub, diag, sup, rhs, &ai_ai_dotproducts, &mut x, D, iter, eps)?;
    Ok(x)
}