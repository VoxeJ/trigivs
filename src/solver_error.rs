use thiserror::Error;

#[derive(Error, Debug)]
pub enum SolverErrors {
    /// There is a mismatch in diagonal sizing. Superdiagonal and subdiagonal must be equal in length and exactly one element shorter than the main diagonal
    /// 
    #[error("Invalid diagonal sizing")]
    InvalidDiagonals,

    /// There is a mismatch in RHS sizing. RHS must be the same length as the main diagonal
    /// 
    #[error("Invalid RHS sizing")]
    InvalidRhsSizing,

    /// Attempt of divizion by zero occured
    /// 
    #[error("Division by zero")]
    DivisionByZero,
}