use crate::prelude::*;
use crate::solver_error::SolverErrors;
use itertools::izip;
use num_traits::float::Float;

use std::iter;
#[cfg(feature = "std")]
use std::mem::swap;

#[cfg(not(feature = "std"))]
use core::mem::swap;

#[cfg(feature = "std")]
#[cfg(not(feature = "std"))]
use rand::rngs::SmallRng;

#[inline]
pub fn rotate_primary<T: Float>(
    ai: T,
    di: T,
    bi: T,
    di1: T,
) -> Result<(T, T, T, T, T), SolverErrors> {
    let r = (di * di + bi * bi).sqrt();
    if r < T::epsilon() {
        return Err(SolverErrors::DivisionByZero);
    }

    let c = di / r;
    let s = -bi / r;

    let new_a = c * ai - s * di1;
    let new_di1 = s * ai + c * di1;

    Ok((r, new_a, new_di1, s, c))
}

#[inline]
pub fn rotate_secondary<T: Float>(ai1: T, s: T, c: T) -> (T, T) {
    let u = -s * ai1;
    let new_ai1 = c * ai1;
    (u, new_ai1)
}

#[inline]
fn rotate_rhs<T: Float>(rhsi: T, rhsi1: T, s: T, c: T) -> (T, T) {
    let new_rhsi = c * rhsi - s * rhsi1;
    let new_rhsi1 = s * rhsi + c * rhsi1;
    (new_rhsi, new_rhsi1)
}

#[inline]
pub fn is_zero_eps_mag<T: Float>(val: T, mag: T) -> bool {
    let magabs = mag.abs();
    val.abs()
        < (if magabs < T::one() {
            T::epsilon()
        } else {
            magabs * T::epsilon()
        })
}

#[inline]
pub fn compute_x<T: Float>(
    x_buffer: &mut [T],
    rhs: &[T],
    d: &[T],
    a: &[T],
    u: &[T],
) -> Result<(), SolverErrors> {
    let z = T::zero();
    let a_iter = a.iter().chain(iter::once(&z)).rev();
    let u_iter = u.iter().chain(iter::repeat(&z).take(2)).rev();
    let mut x_pp = z;
    let mut x_p = z;
    for (&rhsi, &di, &ai, &ui, x) in izip!(rhs.iter().rev(), d.iter().rev(), a_iter, u_iter, x_buffer.iter_mut().rev()) {
        let rhs_sum = rhsi - ai * x_p - ui * x_pp;
        if is_zero_eps_mag(di, rhs_sum) {
            return Err(SolverErrors::DivisionByZero);
        }
        *x = rhs_sum / di;
        x_pp = x_p;
        x_p = *x;
    }
    Ok(())
}

pub fn solve_givens_body<T: Float>(
    sub: &[T],
    d_buffer: &mut [T],
    a_buffer: &mut [T],
    u_buffer: &mut [T],
    r_buffer: &mut [T],
    x_buffer: &mut [T],
) -> Result<(), SolverErrors> {
    let n = d_buffer.len();
    for i in 0..(n - 1) {
        let bi = sub[i];
        let di = d_buffer[i];

        if is_zero_eps_mag(bi, di) {
            continue;
        } else if is_zero_eps_mag(di, bi) {
            d_buffer[i] = sub[i];
            swap(&mut d_buffer[i + 1], &mut a_buffer[i]);
            r_buffer.swap(i, i + 1);
            if i < u_buffer.len() {
                u_buffer[i] = a_buffer[i + 1];
                a_buffer[i + 1] = T::zero();
            }
            continue;
        }

        let ai = a_buffer[i];
        let di1 = d_buffer[i + 1];
        let rhsi = r_buffer[i];
        let rhsi1 = r_buffer[i + 1];

        let c;
        let s;

        (d_buffer[i], a_buffer[i], d_buffer[i + 1], s, c) = rotate_primary(ai, di, bi, di1)?;
        (r_buffer[i], r_buffer[i + 1]) = rotate_rhs(rhsi, rhsi1, s, c);

        if i < u_buffer.len() {
            (u_buffer[i], a_buffer[i + 1]) = rotate_secondary(a_buffer[i + 1], s, c);
        }
    }
    compute_x(x_buffer, r_buffer, d_buffer, a_buffer, u_buffer)?;
    Ok(())
}

pub fn precompute_givens_body<T: Float>(
    sub: &[T],
    d_buffer: &mut [T],
    a_buffer: &mut [T],
    u_buffer: &mut [T],
    sc_buffer: &mut [(T, T)],
) -> Result<(), SolverErrors> {
    let n = d_buffer.len();
    for i in 0..(n - 1) {
        let bi = sub[i];
        let di = d_buffer[i];

        if is_zero_eps_mag(bi, di) {
            sc_buffer[i] = (T::zero(), T::one().copysign(di));
            if di.is_sign_negative() {
                d_buffer[i] = -d_buffer[i];
                a_buffer[i] = -a_buffer[i];
                d_buffer[i + 1] = -d_buffer[i + 1];
                if i < u_buffer.len() {
                    a_buffer[i + 1] = -a_buffer[i + 1];
                }
            }
            continue;
        } else if is_zero_eps_mag(di, bi) {
            sc_buffer[i] = (-T::one().copysign(bi), T::zero());
            d_buffer[i] = sub[i].abs();
            swap(&mut d_buffer[i + 1], &mut a_buffer[i]);
            if bi.is_sign_negative() {
                a_buffer[i] = -a_buffer[i];
                d_buffer[i] = -sub[i];
            } else {
                d_buffer[i + 1] = -d_buffer[i + 1];
                d_buffer[i] = sub[i];
            }
            if i < u_buffer.len() {
                u_buffer[i] = if bi.is_sign_negative() {
                    -a_buffer[i + 1]
                } else {
                    a_buffer[i + 1]
                };
                a_buffer[i + 1] = T::zero();
            }
            continue;
        }

        let ai = a_buffer[i];
        let di1 = d_buffer[i + 1];

        let c;
        let s;

        (d_buffer[i], a_buffer[i], d_buffer[i + 1], s, c) = rotate_primary(ai, di, bi, di1)?;
        sc_buffer[i] = (s, c);

        if i < u_buffer.len() {
            (u_buffer[i], a_buffer[i + 1]) = rotate_secondary(a_buffer[i + 1], s, c);
        }
    }
    Ok(())
}

pub fn solve_givens_sc_rhs_body<T: Float>(sins_cosins: &[(T, T)], r_buffer: &mut [T]) {
    for (i, &(s, c)) in sins_cosins.iter().enumerate() {
        if s.abs() < T::epsilon() {
            if c.is_sign_negative() {
                r_buffer[i] = -r_buffer[i];
                r_buffer[i + 1] = -r_buffer[i + 1];
            }
            continue;
        } else if c.abs() < T::epsilon() {
            r_buffer.swap(i, i + 1);
            if s.is_sign_negative() {
                r_buffer[i + 1] = -r_buffer[i + 1];
            } else {
                r_buffer[i] = -r_buffer[i];
            }
            continue;
        }
        (r_buffer[i], r_buffer[i + 1]) = rotate_rhs(r_buffer[i], r_buffer[i + 1], s, c);
    }
}

pub fn kaczmarz_body<T: Float>(
    sub: &[T],
    diag: &[T],
    sup: &[T],
    rhs: &[T],
    ai_ai_prod: &[T],
    x_buffer: &mut [T],
    n: usize,
    iter: usize,
    eps: T,
) -> Result<(), SolverErrors> {
    let fade = T::from(0.75).unwrap();
    let min_to_fade = T::from(1.2).unwrap();
    let mut overshoot_counter = T::one();
    let mut r = compute_solution_residual_norm(sup, diag, sub, rhs, x_buffer)?;
    let mut w = T::one();
    let mut prev_r;
    if r < eps {
        return Ok(());
    }
    if n == 1 {
        if is_zero_eps_mag(diag[0], rhs[0]) {
            return Err(SolverErrors::DivisionByZero);
        }
        x_buffer[0] = rhs[0] / diag[0];
        return Ok(());
    }
    for _ in 0..iter {
        for i in 0..n {
            let mut xi_ai_dotproduct = diag[i] * x_buffer[i];
            if i < n - 1 {
                xi_ai_dotproduct = xi_ai_dotproduct + sup[i] * x_buffer[i + 1];
            }
            if i > 0 {
                xi_ai_dotproduct = xi_ai_dotproduct + sub[i - 1] * x_buffer[i - 1];
            }
            let numerator = (rhs[i] - xi_ai_dotproduct) * w;
            if is_zero_eps_mag(ai_ai_prod[i], numerator) {
                return Err(SolverErrors::DivisionByZero);
            }
            let c = numerator / ai_ai_prod[i];
            x_buffer[i] = x_buffer[i] + c * diag[i];
            if i < n - 1 {
                x_buffer[i + 1] = x_buffer[i + 1] + c * sup[i];
            }
            if i > 0 {
                x_buffer[i - 1] = x_buffer[i - 1] + c * sub[i - 1];
            }
        }
        prev_r = r;
        r = compute_solution_residual_norm(sup, diag, sub, rhs, x_buffer)?;
        w = if r < prev_r {
            T::one() / (overshoot_counter)
        } else {
            T::one() / (overshoot_counter + r / prev_r)
        };
        if overshoot_counter > min_to_fade {
            overshoot_counter = overshoot_counter * fade;
        }
        if prev_r < r {
            overshoot_counter = overshoot_counter + T::one();
        }
        if r < eps {
            break;
        }
    }
    Ok(())
}

pub fn get_ruiz_equilibrium_mul<T: Float>(
    d_buffer: &[T],
    sup_buffer: &[T],
    sub_buffer: &[T],
    row_buffer: &mut [T],
    col_buffer: &mut [T],
    iter: usize,
    eps: T,
) -> Result<(), SolverErrors> {
    let n = d_buffer.len();
    if n == 1 {
        if d_buffer[0].abs() < T::epsilon() {
            return Err(SolverErrors::DivisionByZero);
        }
        row_buffer[0] = T::one() / d_buffer[0].abs();
        //d_buffer[0] = T::one();
        return Ok(());
    }
    for _ in 0..iter {
        let mut max_norm_diff = T::zero();
        for i in 0..n {
            let row_s = row_buffer[i];
            let mut div = (d_buffer[i]).abs() * col_buffer[i] * row_s;
            if i < n - 1 {
                div = div.max(sup_buffer[i].abs() * col_buffer[i + 1] * row_s);
            }
            if i > 0 {
                div = div.max(sub_buffer[i - 1].abs() * col_buffer[i - 1] * row_s);
            }
            div = div.sqrt();

            if div < T::epsilon() {
                return Err(SolverErrors::DivisionByZero);
            }

            //let r = T::one() / div;

            row_buffer[i] = row_buffer[i] / div;

            let curr_norm_diff = (div - T::one()).abs();
            if max_norm_diff < curr_norm_diff {
                max_norm_diff = curr_norm_diff;
            }
        }

        for i in 0..n {
            let col_s = col_buffer[i];
            let mut div = d_buffer[i].abs() * row_buffer[i] * col_s;
            if i < n - 1 {
                div = div.max(sub_buffer[i].abs() * row_buffer[i + 1] * col_s);
            }
            if i > 0 {
                div = div.max(sup_buffer[i - 1].abs() * row_buffer[i - 1] * col_s);
            }
            div = div.sqrt();
            if div < T::epsilon() {
                return Err(SolverErrors::DivisionByZero);
            }
            col_buffer[i] = col_buffer[i] / div;

            let curr_norm_diff = (div - T::one()).abs();
            if max_norm_diff < curr_norm_diff {
                max_norm_diff = curr_norm_diff;
            }
        }
        if max_norm_diff <= eps {
            break;
        }
    }
    Ok(())
}

pub fn apply_ruiz<T: Float>(
    d_buffer: &mut [T],
    sup_buffer: &mut [T],
    sub_buffer: &mut [T],
    row_buffer: &[T],
    col_buffer: &[T],
) {
    let n = d_buffer.len();
    if n == 1 {
        d_buffer[0] = T::one();
    }
    for i in 0..n {
        d_buffer[i] = d_buffer[i] * row_buffer[i] * col_buffer[i];
        if i < n - 1 {
            sup_buffer[i] = sup_buffer[i] * row_buffer[i] * col_buffer[i + 1];
        }
        if i > 0 {
            sub_buffer[i - 1] = sub_buffer[i - 1] * row_buffer[i] * col_buffer[i - 1];
        }
    }
}
