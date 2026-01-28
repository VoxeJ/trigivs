use crate::solver_error::SolverErrors;
use num_traits::float::Float;

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
