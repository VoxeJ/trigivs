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
pub struct TridiagSysPrecomp<T: Float> {
    diag: Vec<T>,
    sup1: Option<Vec<T>>,
    sup2: Option<Vec<T>>,

    sins_cosins: Option<Vec<(T, T)>>
}

/// Holds precomputed tridiagonal system with Ruiz preconditioning for working with multiple righ hand sides with heap allocation
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
/// let precomputed = trigivs::prelude::precompute_givens_ruiz(
///     &sup, 
///     &diag, 
///     &sub, 
///     5, 
///     0.001
/// ).unwrap();
/// 
/// let x1 = precomputed.solve_givens_rhs(&rhs1).unwrap();
/// let x2 = precomputed.solve_givens_rhs(&rhs2).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct TridiagSysRuizPrecomp<T: Float> {
    sys: TridiagSysPrecomp<T>,

    row_mul: Vec<T>,
    col_mul: Vec<T>
}

/// Solves a trigiagonal system of linear equations with heap allocation
/// 
/// # Arguments 
/// 
/// * `sup` - superdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sub` - subdiagonal elements, length n-1
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
/// * `sup` - superdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sub` - subdiagonal elements, length n-1
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
pub fn precompute_givens<T: Float>(sup: &[T], diag: &[T], sub: &[T]) -> Result<TridiagSysPrecomp<T>, SolverErrors> {
    let n = diag.len();

    if sup.len() != sub.len() || sup.len() + 1 != diag.len() {
        return Err(SolverErrors::InvalidDiagonals);
    }
    let mut a = sup.to_vec();
    let mut d = diag.to_vec();
    let mut u  = if n > 1 {vec![T::zero(); n-2]} else {vec![]};

    let mut sins_cosins = vec![(T::zero(), T::zero()); n-1];

    precompute_givens_body(sub, &mut d, &mut a, &mut u, &mut sins_cosins)?;
    Ok(TridiagSysPrecomp {
        diag: d,
        sup1: if a.len() > 0 {Some(a)} else {None},
        sup2: if u.len() > 0 {Some(u)} else {None},
        sins_cosins: if sins_cosins.len() > 0 {Some(sins_cosins)} else {None},
    })
}

/// Precomputes a system for multiple differenr RHS with Ruiz preconditioning with heap allocation
/// 
/// # Arguments
/// 
/// * `sup` - superdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sub` - subdiagonal elements, length n-1
/// * `r_iter` - maximum amount of iterations for Ruiz preconditioning (impractical over 10)
/// * `r_eps` - finishing criteria for Ruiz preconditioning through maximum column/row inf-norm deviation from 1
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
/// let precomputed = trigivs::prelude::precompute_givens_ruiz(&sup, &diag, &sub, 5, 0.01).unwrap();
/// ```
pub fn precompute_givens_ruiz<T: Float>(
    sup: &[T], 
    diag: &[T], 
    sub: &[T], 
    r_iter: usize, 
    r_eps: T
) -> Result<TridiagSysRuizPrecomp<T>, SolverErrors> {
    let n = diag.len();

    if sup.len() != sub.len() || sup.len() + 1 != diag.len() {
        return Err(SolverErrors::InvalidDiagonals);
    }

    let mut sub_b = sub.to_vec();
    let mut sup_b = sup.to_vec();
    let mut diag_b = diag.to_vec();

    let mut col_b = vec![T::one(); n];
    let mut row_b = vec![T::one(); n];

    get_ruiz_equilibrium_mul(
        &diag_b,
        &sup_b, 
        &sub_b, 
        &mut row_b, 
        &mut col_b, 
        r_iter, 
        r_eps
    )?;
    apply_ruiz(
        &mut diag_b, 
        &mut sup_b, 
        &mut sub_b, 
        &row_b, 
        &col_b
    );

    let mut u  = if n > 1 {vec![T::zero(); n-2]} else {vec![]};
    let mut sins_cosins = vec![(T::zero(), T::zero()); n-1];

    precompute_givens_body(&sub_b, &mut diag_b, &mut sup_b, &mut u, &mut sins_cosins)?;
    Ok(
        TridiagSysRuizPrecomp { 
            sys: TridiagSysPrecomp { 
                diag: diag_b, 
                sup1: if sup_b.len() > 0 {Some(sup_b)} else {None},
                sup2: if u.len() > 0 {Some(u)} else {None}, 
                sins_cosins: if sins_cosins.len() > 0 {Some(sins_cosins)} else {None} 
            }, 
            row_mul: row_b, 
            col_mul: col_b 
        }
    )
}

impl<T: Float> TridiagSysPrecomp<T> {

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

impl<T: Float> TridiagSysRuizPrecomp<T>{

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
    /// let precomputed = trigivs::prelude::precompute_givens_ruiz(
    ///     &sup,
    ///     &diag,
    ///     &sub,
    ///     5,
    ///     0.001
    /// ).unwrap();
    /// 
    /// let x = precomputed.solve_givens_rhs(&rhs).unwrap();
    /// ```
    pub fn solve_givens_rhs(&self, rhs: &[T]) -> Result<Vec<T>, SolverErrors> {
        if rhs.len() != self.sys.diag.len() {
            return Err(SolverErrors::InvalidRhsSizing);
        }
        let rhsl = rhs
            .into_iter()
            .zip(&self.row_mul)
            .map(|(&r, &row)| r * row)
            .collect::<Vec<_>>();
        let x = 
            self.sys.solve_givens_rhs(&rhsl)?
            .into_iter()
            .zip(&self.col_mul)
            .map(|(x, &col)| x * col)
            .collect::<Vec<_>>();
        Ok(x)
    }
}

/// Solves or refines a tridiagonal system using the Kaczmarz iterative method with heap allocation
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
/// let x_refined = trigivs::prelude::tridiag_iter_kaczmarz(
///     &sup, 
///     &diag, 
///     &sub, 
///     &rhs, 
///     &x_init, 
///     1000, 
///     1e-6 
/// ).unwrap();
/// ```
/// 
pub fn tridiag_iter_kaczmarz<T: Float>(
    sup: &[T], 
    diag: &[T], 
    sub: &[T], 
    rhs: &[T], 
    x_init: &[T], 
    iter: usize, 
    eps: T
) -> Result<Vec<T>, SolverErrors>{
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

/// Compute row and column scaling parameters using Ruiz preconditioning without heap allocation
/// 
/// # Arguments
/// 
/// * `sup` - superdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sub` - subdiagonal elements, length n-1
/// * `r_iter` - maximum amount of iterations for Ruiz preconditioning (impractical over 10)
/// * `r_eps` - finishing criteria for Ruiz preconditioning through maximum column/row inf-norm deviation from 1
/// 
/// # Output
/// 
/// (R, C)
/// * `R` - row scaling parmeters
/// * `C` - column scaling parmeters
/// 
/// # Example
/// 
/// ```
/// let sup = [-4.];
/// let diag = [3., 2.];
/// let sub = [5.];
/// 
/// let (row_s, column_s) = trigivs::prelude::compute_ruiz_scaling(&sup, &diag, &sub, 5, 0.01).unwrap();
/// ```
pub fn compute_ruiz_scaling<T: Float>(
    sup: &[T], 
    diag: &[T], 
    sub: &[T], 
    iter: usize, 
    eps: T
) -> Result<(Vec<T>, Vec<T>), SolverErrors>
{
    if sup.len() != sub.len() || sup.len() + 1 != diag.len() {
        return Err(SolverErrors::InvalidDiagonals);
    }
    let n = diag.len();
    let mut row_b = vec![T::one(); n];
    let mut col_b = vec![T::one(); n]; 
    get_ruiz_equilibrium_mul(
        diag,
        sup, 
        sub, 
        &mut row_b, 
        &mut col_b, 
        iter, 
        eps
    )?;
    Ok((row_b, col_b))
}

/// Solves a tridiagonal system using Givens rotations with Ruiz preconditioning
/// 
/// # Arguments
/// 
/// * `sup` - superdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sub` - subdiagonal elements, length n-1
/// * `r_iter` - maximum amount of iterations for Ruiz preconditioning (impractical over 10)
/// * `r_eps` - finishing criteria for Ruiz preconditioning through maximum column/row inf-norm deviation from 1
/// 
/// # Example
///
/// ```
/// let sub = [5.];
/// let diag = [3., 2.];
/// let sup = [-4.];
/// let rhs = [-3., 21.];
/// 
/// let x = trigivs::prelude::solve_givens_ruiz_precond(
///     &sub,
///     &diag,
///     &sup,
///     &rhs,
///     5,
///     0.001
/// ).unwrap();
/// ```
/// 
pub fn solve_givens_ruiz_precond<T: Float>(sup: &[T], diag: &[T], sub: &[T], rhs: &[T], r_iter: usize, r_eps: T) -> Result<Vec<T>, SolverErrors> {
    if sup.len() != sub.len() || sup.len() + 1 != diag.len() {
        return Err(SolverErrors::InvalidDiagonals);
    } else if diag.len() != rhs.len() {
        return Err(SolverErrors::InvalidRhsSizing);
    }

    let n = diag.len();

    let mut sub_b = sub.to_vec();
    let mut sup_b = sup.to_vec();
    let mut diag_b = diag.to_vec();
    let mut rhs = rhs.to_vec();

    let mut col_b = vec![T::one(); n];

    {
        let mut row_b = vec![T::one(); n];
        get_ruiz_equilibrium_mul(
            &diag_b,
            &sup_b, 
            &sub_b, 
            &mut row_b, 
            &mut col_b, 
            r_iter, 
            r_eps
        )?;
        apply_ruiz(
            &mut diag_b, 
            &mut sup_b, 
            &mut sub_b, 
            &row_b, 
            &col_b
        );
        rhs.iter_mut().zip(row_b).for_each(|(rhs, row)| *rhs = *rhs * row);
    }
    
    let mut u = if n > 1 {vec![T::zero(); n-2]} else {vec![]};
    let mut x = vec![T::zero(); n];

    solve_givens_body(&sub_b, &mut diag_b, &mut sup_b, &mut u, &mut rhs, &mut x)?;

    x.iter_mut().zip(col_b).for_each(|(x, col)| *x = *x * col);

    Ok(x)
}