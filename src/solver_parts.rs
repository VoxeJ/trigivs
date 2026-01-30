use crate::solver_error::SolverErrors;
use num_traits::float::Float;
use crate::prelude::*;

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
pub fn rotate_rhs<T: Float>(rhsi: T, rhsi1: T, s: T, c: T) -> (T, T) {
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
pub fn kaczmarz_body<T: Float>(sub: &[T], diag: &[T], sup: &[T], rhs: &[T], ai_ai_prod: &[T], x_buffer: &mut [T], n: usize, iter: usize, eps: T) -> Result<(), SolverErrors>{
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