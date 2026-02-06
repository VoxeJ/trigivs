use crate::solver_error::SolverErrors;
use crate::solver_parts::*;

use num_traits::Float;

/// Holds tridiagonal system precomputed for working with multiple righ hand sides without heap allocation
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
pub struct TridiagSysPrecomp<T: Float, const D: usize, const S: usize> {
    diag: [T; D],
    sup1: Option<[T; S]>,
    sup2: Option<[T; S]>,

    sins_cosins: Option<[(T, T); S]>
}

/// Holds precomputed tridiagonal system with Ruiz preconditioning for working with multiple righ hand sides without heap allocation
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
pub struct TridiagSysRuizPrecomp<T: Float, const D: usize, const S: usize>{
    sys: TridiagSysPrecomp<T, D, S>,

    row_mul: [T; D],
    col_mul: [T; D]
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
pub fn precompute_givens<T: Float, const D: usize, const S: usize>(sup: &[T; S], diag: &[T; D], sub: &[T; S]) -> Result<TridiagSysPrecomp<T, D, S>, SolverErrors> {

    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut a = sup.clone();
    let mut d = diag.clone();

    let mut ur = [T::zero(); S];
    let u = if S > 1 {&mut ur[..S-1]} else {&mut []};

    let mut sins_cosins = [(T::zero(), T::zero()); S];

    precompute_givens_body(sub, &mut d, &mut a, u, &mut sins_cosins)?;

    Ok(TridiagSysPrecomp {
        diag: d,
        sup1: if a.len() > 0 {Some(a)} else {None},
        sup2: if u.len() > 0 {Some(ur)} else {None},
        sins_cosins: if sins_cosins.len() > 0 {Some(sins_cosins)} else {None},
    })
}

/// Precomputes a system for multiple differenr RHS with Ruiz preconditioning without heap allocation
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
pub fn precompute_givens_ruiz<T: Float, const D: usize, const S: usize>(
    sup: &[T; S],
    diag: &[T; D],
    sub: &[T; S],
    r_iter: usize, 
    r_eps: T
) -> Result<TridiagSysRuizPrecomp<T, D, S>, SolverErrors> {

    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut sub_b = sub.clone();
    let mut sup_b = sup.clone();
    let mut diag_b = diag.clone();

    let mut col_b = [T::one(); D];
    let mut row_b = [T::one(); D];

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

    let mut sins_cosins = [(T::zero(), T::zero()); S];

    let mut ur = [T::zero(); S];
    let u = if S > 1 {&mut ur[..S-1]} else {&mut []};

    precompute_givens_body(&sub_b, &mut diag_b, &mut sup_b, u, &mut sins_cosins)?;

    Ok(
        TridiagSysRuizPrecomp { 
            sys: TridiagSysPrecomp { 
                diag: diag_b, 
                sup1: if sup_b.len() > 0 {Some(sup_b)} else {None},
                sup2: if u.len() > 0 {Some(ur)} else {None}, 
                sins_cosins: if sins_cosins.len() > 0 {Some(sins_cosins)} else {None} 
            }, 
            row_mul: row_b, 
            col_mul: col_b 
        }
    )
}

impl<T: Float, const D: usize, const S: usize> TridiagSysPrecomp<T, D, S> {

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

        let mut x_buffer = [T::zero(); D];

        compute_x(
            &mut x_buffer,
            &rhsl,
            &self.diag,
            self.sup1.as_ref().map_or(&[], |v| v.as_slice()),
            self.sup2.as_ref().map_or(&[], |v| v.as_slice()),
        )?;

        Ok(x_buffer)
    }
}


impl<T: Float, const D: usize, const S: usize> TridiagSysRuizPrecomp<T, D, S> {

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
    /// let precomputed = trigivs::prelude::precompute_givens_ruiz(&sup, &diag, &sub, 5, 0.001).unwrap();
    /// 
    /// let x = precomputed.solve_givens_rhs(&rhs).unwrap();
    /// ```
    pub fn solve_givens_rhs(&self, rhs: &[T; D]) -> Result<[T; D], SolverErrors> {
        let mut rhsl = rhs.clone();
        rhsl.iter_mut().zip(self.row_mul).for_each(|(r, row)| *r = *r * row);

        let mut x = self.sys.solve_givens_rhs(&rhsl)?;
        x.iter_mut().zip(self.col_mul).for_each(|(x, col)| *x = *x * col);

        Ok(x)
    }
}

/// Refines a tridiagonal system solution using the Kaczmarz iterative method without heap allocation
/// 
/// # Arguments
///
/// * `sup` - superdiagonal elements, length n-1
/// * `diag` - main diagonal elements, length n
/// * `sub` - subdiagonal elements, length n-1
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
pub fn tridiag_iter_kaczmarz<T: Float, const D: usize, const S: usize>(sup: &[T; S], diag: &[T; D], sub: &[T; S], rhs: &[T; D], x_init: &[T; D], iter: usize, eps: T) -> Result<[T; D], SolverErrors>{
    
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
pub fn compute_ruiz_scaling<T: Float, const D: usize, const S: usize>(
    sup: &[T; S], 
    diag: &[T; D], 
    sub: &[T; S], 
    iter: usize, 
    eps: T
) -> Result<([T; D], [T; D]), SolverErrors>
{
    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut row_b = [T::one(); D];
    let mut col_b = [T::one(); D];
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

/// Solves a tridiagonal system using Givens rotations with Ruiz preconditioning without heap allocation
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
pub fn solve_givens_ruiz_precond<T: Float, const D: usize, const S: usize>(sup: &[T; S], diag: &[T; D], sub: &[T; S], rhs: &[T; D], r_iter: usize, r_eps: T) -> Result<[T; D], SolverErrors> {
    
    const { assert!(D == S + 1, "Sub and sup diagonals must be exctly 1 element smaller than main diagonal") };

    let mut sub_b = sub.clone();
    let mut sup_b = sup.clone();
    let mut diag_b = diag.clone();
    let mut rhs = rhs.clone();

    let mut col_b = [T::one(); D];

    {
        let mut row_b = [T::one(); D];
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
    
    let mut ur = [T::zero(); S];
    let u = if S > 1 {&mut ur[..S-1]} else {&mut []};

    let mut x = [T::zero(); D];

    solve_givens_body(&sub_b, &mut diag_b, &mut sup_b, u, &mut rhs, &mut x)?;

    x.iter_mut().zip(col_b).for_each(|(x, col)| *x = *x * col);

    Ok(x)
}