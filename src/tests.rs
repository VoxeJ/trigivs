use approx::assert_abs_diff_eq;
use crate::prelude::*;

#[test]
fn test_determinant(){
    let sup = [4., 7., 7., 100.];
    let diag = [1., 3., 6., 4., 12.];
    let sub = [2., 5., 10., 90.];

    let result = compute_tridiag_determinant(&sup, &diag, &sub);
    let expected = 586080.;

    assert_abs_diff_eq!(result, expected, epsilon=1e-6);
}

#[test]
fn test_1diag_solver(){
    let diag = [2.];
    let rhs = [10.];
    
    let result = solve_givens(&[], &diag, &[], &rhs).unwrap();
    let expected = [5.];

    assert_abs_diff_eq!(result.as_ref() as &[f64], expected.as_ref(), epsilon=1e-6);
}

#[test]
fn test_2diag_solver(){
    let sup = [-4.];
    let diag = [3., 2.];
    let sub = [5.];
    let rhs = [-3., 21.];
    
    let result = solve_givens(&sup, &diag, &sub, &rhs).unwrap();
    let expected = [3., 3.];

    assert_abs_diff_eq!(result.as_ref() as &[f64], expected.as_ref(), epsilon=1e-6);
}

#[test]
fn test_solver(){
    let sup = [4., 7., 7., 100.];
    let diag = [1., 3., 6., 4., 12.];
    let sub = [2., 5., 10., 90.];
    let rhs = [-7., 17., -20., 514., -300.];
    
    let result = solve_givens(&sup, &diag, &sub, &rhs).unwrap();
    let expected = [1., -2., 3., -4., 5.];

    assert_abs_diff_eq!(result.as_ref() as &[f64], expected.as_ref(), epsilon=1e-6);
}

#[test]
fn test_zero_division(){
    let sup = [4., 7., 0., 100.];
    let diag = [1., 3., 0., 4., 12.];
    let sub = [2., 0., 10., 90.];
    let rhs = [-7., 17., -20., 514., -300.];
    
    let result = solve_givens(&sup, &diag, &sub, &rhs);
    
    assert!(matches!(result, Err(SolverErrors::DivisionByZero)));
}

#[test]
fn test_1diag_precomp(){
    let diag = [2.];

    let rhs1 = [10.];
    let rhs2 = [-8.];
    
    let precomp = precompute_givens(&[], &diag, &[]).unwrap();

    let result1 = precomp.solve_givens_rhs(&rhs1).unwrap();
    let result2 = precomp.solve_givens_rhs(&rhs2).unwrap();

    let expected1 = [5.];
    let expected2 = [-4.];

    assert_abs_diff_eq!(result1.as_ref() as &[f64], expected1.as_ref(), epsilon=1e-6);
    assert_abs_diff_eq!(result2.as_ref() as &[f64], expected2.as_ref(), epsilon=1e-6);
}

#[test]
fn test_2diag_precomp(){
    let sup = [-4.];
    let diag = [3., 2.];
    let sub = [5.];

    let rhs1 = [-3., 21.];
    let rhs2  = [-23., 5.];
    
    let precomp = precompute_givens(&sup, &diag, &sub).unwrap();

    let result1 = precomp.solve_givens_rhs(&rhs1).unwrap();
    let result2 = precomp.solve_givens_rhs(&rhs2).unwrap();

    let expected1 = [3., 3.];
    let expected2 = [-1., 5.];

    assert_abs_diff_eq!(result1.as_ref() as &[f64], expected1.as_ref(), epsilon=1e-6);
    assert_abs_diff_eq!(result2.as_ref() as &[f64], expected2.as_ref(), epsilon=1e-6);
}

#[test]
fn test_precomp(){
    let sup = [4., 7., 7., 100.];
    let diag = [1., 3., 6., 4., 12.];
    let sub = [2., 5., 10., 90.];
    
    let rhs1 = [-7., 17., -20., 514., -300.];
    let rhs2 = [43., 57., -72., 450., -1740.];
    
    let precomp = precompute_givens(&sup, &diag, &sub).unwrap();
    
    let result1 = precomp.solve_givens_rhs(&rhs1).unwrap();
    let result2 = precomp.solve_givens_rhs(&rhs2).unwrap();

    let expected1 = [1., -2., 3., -4., 5.];
    let expected2 = [3., 10., 3., -20., 5.];

    assert_abs_diff_eq!(result1.as_ref() as &[f64], expected1.as_ref(), epsilon=1e-6);
    assert_abs_diff_eq!(result2.as_ref() as &[f64], expected2.as_ref(), epsilon=1e-6);
}

#[test]
fn test_precom_direct_eq_solver(){
    let sup = [4., 7., 7., 100.];
    let diag = [1., 3., 6., 4., 12.];
    let sub = [2., 5., 10., 90.];
    let rhs = [-7., 17., -20., 514., -300.];
    
    let precomp = precompute_givens(&sup, &diag, &sub).unwrap();

    let result1 = solve_givens(&sup, &diag, &sub, &rhs).unwrap();
    let result2 = precomp.solve_givens_rhs(&rhs).unwrap();

    assert_abs_diff_eq!(result1.as_ref() as &[f64], result2.as_ref(), epsilon=1e-6);
}

#[cfg(feature = "alloc")]
#[test]
fn test_invalid_diag(){
    let sup = [4., 7., 7., 100.];
    let diag = [1., 3., 6., 4., 12.];
    let sub = [2., 5., 10.];
    let rhs = [-7., 17., -20., 514., -300.];

    let result = solve_givens(&sup, &diag, &sub, &rhs);

    assert!(matches!(result, Err(SolverErrors::InvalidDiagonals)));
}

#[cfg(feature = "alloc")]
#[test]
fn test_invalid_rhs(){
    let sup = [4., 7., 7., 100.];
    let diag = [1., 3., 6., 4., 12.];
    let sub = [2., 5., 10., 90.];
    let rhs = [-7., 17., -20., 514.];
    
    let result = solve_givens(&sup, &diag, &sub, &rhs);

    assert!(matches!(result, Err(SolverErrors::InvalidRhsSizing)));
}

#[cfg(feature = "alloc")]
#[test]
fn test_invalid_diag_precomp(){
    let sup = [4., 7., 7., 100.];
    let diag = [1., 3., 6., 4., 12.];
    let sub = [2., 5., 10.];

    let result = precompute_givens(&sup, &diag, &sub);

    assert!(matches!(result, Err(SolverErrors::InvalidDiagonals)));
}

#[cfg(feature = "alloc")]
#[test]
fn test_invalid_rhs_precomp(){
    let sup = [4., 7., 7., 100.];
    let diag = [1., 3., 6., 4., 12.];
    let sub = [2., 5., 10., 90.];
    
    let rhs = [-7., 17., -20., 514.];
    
    let precomp = precompute_givens(&sup, &diag, &sub).unwrap();
    let result = precomp.solve_givens_rhs(&rhs);

    assert!(matches!(result, Err(SolverErrors::InvalidRhsSizing)));
}