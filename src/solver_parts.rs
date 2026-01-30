use crate::solver_error::SolverErrors;
use num_traits::float::Float;
use crate::prelude::*;

#[cfg(feature = "std")]
use std::mem::swap;

#[cfg(not(feature = "std"))]
use core::mem::swap;

#[inline]
pub fn rotate_primary<T: Float>(ai: T, di: T, bi: T, di1: T) -> Result<(T, T, T, T, T), SolverErrors> {
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
pub fn compute_x<T: Float>(x_buffer: &mut [T], rhs: &[T], d: &[T], a: &[T], u: &[T]) -> Result<(), SolverErrors> {
    let n = x_buffer.len();
    for i in (0..n).rev() {
        let mut sum = T::zero();
        if i < n - 1 {
            sum = a[i] * x_buffer[i + 1];
        }
        if n > 1 && i < n - 2 {
            sum = sum + u[i] * x_buffer[i + 2];
        }
        let rhs_sum = rhs[i] - sum;
        let di = d[i];
        if is_zero_eps_mag(di, rhs_sum) {
            return Err(SolverErrors::DivisionByZero);
        }
        x_buffer[i] = (rhs_sum) / di;
    }
    Ok(())
}

pub fn solve_givens_body<T: Float>(
    sub: &[T], 
    d_buffer: &mut [T], 
    a_buffer: &mut [T], 
    u_buffer: &mut [T], 
    r_buffer: &mut [T],
    x_buffer: &mut [T]
) -> Result<(), SolverErrors>{
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
) -> Result<(), SolverErrors>{
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
                u_buffer[i] = if bi.is_sign_negative() {-a_buffer[i + 1]} else {a_buffer[i + 1]};
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

pub fn solve_givens_sc_rhs_body<T: Float>(sins_cosins: &[(T, T)], r_buffer: &mut [T]){
    for (i, &(s, c)) in sins_cosins.iter().enumerate() {
        if s.abs() < T::epsilon() {
            if c.is_sign_negative(){
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
    eps: T
) -> Result<(), SolverErrors>{
    let fade = T::from(0.75).unwrap();
    let min_to_fade = T::from(1.2).unwrap();
    let mut overshoot_counter = T::one();
    let mut prev_r = T::max_value();
    let mut r = compute_solution_norm(sup, diag, sub, rhs, &x_buffer).unwrap();
    for _ in 0..iter{
        let mut contin = false;
        for i in 0..n{
            let xi_ai_dotproduct = 
            if n == 1{
                diag[0] * x_buffer[0]
            } else if i == 0 {
                diag[0] * x_buffer[0] + sup[0] * x_buffer[1]
            } else if i == n - 1 {
                diag[n-1] * x_buffer[n-1] + sub[n - 2] * x_buffer[n-2]
            } else {
                sub[i-1] * x_buffer[i - 1] + diag[i] * x_buffer[i] + sup[i] * x_buffer[i+1]
            };
            let w = T::one()/(overshoot_counter + if is_zero_eps_mag(prev_r, r) {T::one()} else {r/prev_r});
            let numerator = (rhs[i] - xi_ai_dotproduct) * w;
            if is_zero_eps_mag(ai_ai_prod[i], numerator){
                return Err(SolverErrors::DivisionByZero);
            }
            let c = numerator / ai_ai_prod[i];
            if n == 1 {
                x_buffer[0] = x_buffer[0] + c * diag[0];
            } else if i == 0{
                x_buffer[0..=1].iter_mut().zip([diag[0], sup[0]]).for_each(|(element, a)| *element = *element + c * a);
            } else if i == n - 1{
                x_buffer[n-2..=n-1].iter_mut().zip([sub[n-2], diag[n-1]]).for_each(|(element, a)| *element = *element + c * a);
            } else {
                x_buffer[i-1..=i+1].iter_mut().zip([sub[i-1], diag[i], sup[i]]).for_each(|(element, a)| *element = *element + c * a );
            }
            if overshoot_counter > min_to_fade{
                overshoot_counter = overshoot_counter * fade;
            }
            if prev_r > r{
                overshoot_counter = overshoot_counter + T::one();
            }
            prev_r = r;
            r = compute_solution_norm(sup, diag, sub, rhs, &x_buffer).unwrap();
            if eps < r{
                contin = true;
            }
        }
        if contin == false{
            break;
        }
    }
    Ok(())
}