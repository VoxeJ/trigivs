use crate::solver_error::SolverErrors;
use crate::solver_parts::*;

use num_traits::Float;

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
pub fn solve_givens<T: Float, const D: usize, const S: usize>(sup: &[T; S], diag: &[T; D], sub: &[T; S], rhs: &[T; D]) -> Result<[T; D], SolverErrors> 
{
    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut a = sup.clone();
    let mut d = diag.clone();
    let mut rhs = rhs.clone();

    let mut ur = [T::zero(); S];
    let u = if S > 1 {&mut ur[..S-1]} else {&mut []};

    let mut x = [T::zero(); D];

    solve_givens_body(sub, &mut d, &mut a, u, &mut rhs, &mut x)?;

    Ok(x)
}

/// Precomputes a system for multiple differenr RHS without heap allocation
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
pub fn precompute_givens<T: Float, const D: usize, const S: usize>(sup: &[T; S], diag: &[T; D], sub: &[T; S]) -> Result<TridiagonalSystemPrecomputed<T, D, S>, SolverErrors> {

    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut a = sup.clone();
    let mut d = diag.clone();

    let mut ur = [T::zero(); S];
    let u = if S > 1 {&mut ur[..S-1]} else {&mut []};

    let mut sins_cosins = [(T::zero(), T::zero()); S];

    precompute_givens_body(sub, &mut d, &mut a, u, &mut sins_cosins)?;

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
    pub fn solve_givens_rhs(&self, rhs: &[T; D]) -> Result<[T; D], SolverErrors> {
        let mut rhsl = rhs.clone();

        if let Some(sins_cosins) = &self.sins_cosins{
            solve_givens_sc_rhs_body(sins_cosins, &mut rhsl);
        }

        compute_x(
            &rhsl,
            &self.diag,
            self.sup1.as_ref().map_or(&[], |v| v.as_slice()),
            self.sup2.as_ref().map_or(&[], |v| v.as_slice()),
        )
    }
}

/// Refines a tridiagonal system solution using the Kaczmarz iterative method without heap allocation
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