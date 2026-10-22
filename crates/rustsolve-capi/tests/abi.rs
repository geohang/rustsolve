//! The C ABI as a C caller sees it (layouts, status codes, handle states).

use std::ffi::CStr;
use std::ptr;

use rustsolve_capi::*;

/// tridiagonal SPD matrix 2, -1, both triangles, as i32 CSC
fn tridiag(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut cp = vec![0i32];
    let mut ri = Vec::new();
    let mut v = Vec::new();
    for j in 0..n {
        if j > 0 {
            ri.push(j as i32 - 1);
            v.push(-1.0);
        }
        ri.push(j as i32);
        v.push(2.0);
        if j + 1 < n {
            ri.push(j as i32 + 1);
            v.push(-1.0);
        }
        cp.push(ri.len() as i32);
    }
    (cp, ri, v)
}

#[test]
fn layouts_match_the_header() {
    assert_eq!(std::mem::size_of::<RsOptions>(), 56);
    assert_eq!(std::mem::size_of::<RsInfo>(), 112);
    assert_eq!(std::mem::align_of::<RsOptions>(), 8);
    assert_eq!(rs_abi_version(), 1);
}

#[test]
fn defaults_new_update_solve() {
    unsafe {
        let mut o = std::mem::zeroed::<RsOptions>();
        assert_eq!(rs_default_options(&mut o), RS_OK);
        assert_eq!((o.kind, o.threads, o.solve_threads, o.check_symmetry), (0, 1, 1, 1));
        let mut s = ptr::null_mut();
        assert_eq!(rs_new(&o, &mut s), RS_OK);
        let n = 50;
        let (cp, ri, v) = tridiag(n);
        assert_eq!(rs_update_i32(s, n as i64, cp.as_ptr(), ri.as_ptr(), v.as_ptr()), RS_OK);
        // A x = A * ones: x must come back as ones, two columns with a gap
        let mut b = vec![0.0; 2 * (n + 3)];
        for k in 0..2 {
            b[k * (n + 3)] = 1.0;
            b[k * (n + 3) + n - 1] = 1.0;
        }
        assert_eq!(rs_solve(s, 2, b.as_mut_ptr(), (n + 3) as i64), RS_OK);
        for k in 0..2 {
            for i in 0..n {
                assert!((b[k * (n + 3) + i] - 1.0).abs() < 1e-12);
            }
        }
        let mut info = RsInfo::default();
        assert_eq!(rs_get_info(s, &mut info), RS_OK);
        assert_eq!((info.n, info.nnz_a, info.reanalyzed, info.fail_step), (n as i64, (3 * n - 2) as i64, 1, -1));
        let mut rc = 0.0;
        assert_eq!(rs_rcond_estimate(s, &mut rc), RS_OK);
        assert!(rc > 0.0 && rc < 1.0);
        assert_eq!(CStr::from_ptr(rs_last_error(s)).to_str().unwrap(), "");
        rs_free(s);
    }
}

#[test]
fn a_failed_update_leaves_no_factor() {
    unsafe {
        let mut s = ptr::null_mut();
        assert_eq!(rs_new(ptr::null(), &mut s), RS_OK);
        let n = 20;
        let (cp, ri, v) = tridiag(n);
        assert_eq!(rs_update_i32(s, n as i64, cp.as_ptr(), ri.as_ptr(), v.as_ptr()), RS_OK);
        let mut neg = v.clone();
        for x in neg.iter_mut() {
            *x = -*x;
        }
        assert_eq!(rs_update_i32(s, n as i64, cp.as_ptr(), ri.as_ptr(), neg.as_ptr()), RS_NOT_POSITIVE_DEFINITE);
        let msg = CStr::from_ptr(rs_last_error(s)).to_str().unwrap().to_string();
        assert!(msg.starts_with("NOT_POSITIVE_DEFINITE"), "{msg}");
        let mut info = RsInfo::default();
        rs_get_info(s, &mut info);
        assert!(info.fail_column >= 0);
        let mut b = vec![1.0; n];
        assert_eq!(rs_solve(s, 1, b.as_mut_ptr(), n as i64), RS_NOT_FACTORIZED);
        let mut nan = v.clone();
        nan[3] = f64::NAN;
        assert_eq!(rs_update_i32(s, n as i64, cp.as_ptr(), ri.as_ptr(), nan.as_ptr()), RS_NON_FINITE_INPUT);
        assert_eq!(rs_solve(s, 1, b.as_mut_ptr(), n as i64), RS_NOT_FACTORIZED);
        // an input refused before any factorization drops the factor too
        assert_eq!(rs_update_i32(s, n as i64, cp.as_ptr(), ri.as_ptr(), v.as_ptr()), RS_OK);
        let mut bad = ri.clone();
        bad[0] = -1;
        assert_eq!(rs_update_i32(s, n as i64, cp.as_ptr(), bad.as_ptr(), v.as_ptr()), RS_INVALID_INPUT);
        assert_eq!(rs_solve(s, 1, b.as_mut_ptr(), n as i64), RS_NOT_FACTORIZED);
        assert_eq!(rs_solve(s, 0, ptr::null_mut(), n as i64), RS_NOT_FACTORIZED);
        // the same through the i64 entry point and a null pointer, then recovery
        let (cp64, ri64): (Vec<i64>, Vec<i64>) = (cp.iter().map(|&v| v as i64).collect(), ri.iter().map(|&v| v as i64).collect());
        assert_eq!(rs_update_i64(s, n as i64, cp64.as_ptr(), ri64.as_ptr(), v.as_ptr()), RS_OK);
        assert_eq!(rs_update_i64(s, n as i64, cp64.as_ptr(), ptr::null(), v.as_ptr()), RS_INVALID_INPUT);
        assert_eq!(rs_solve(s, 1, b.as_mut_ptr(), n as i64), RS_NOT_FACTORIZED);
        assert_eq!(rs_update_i64(s, n as i64, cp64.as_ptr(), ri64.as_ptr(), v.as_ptr()), RS_OK);
        let mut ones = vec![0.0; n];
        ones[0] = 1.0;
        ones[n - 1] = 1.0;
        assert_eq!(rs_solve(s, 1, ones.as_mut_ptr(), n as i64), RS_OK);
        assert!(ones.iter().all(|x| (x - 1.0).abs() < 1e-12));
        rs_free(s);
    }
}

#[test]
fn bad_arguments() {
    unsafe {
        assert_eq!(rs_new(ptr::null(), ptr::null_mut()), RS_INVALID_INPUT);
        let mut o = std::mem::zeroed::<RsOptions>();
        rs_default_options(&mut o);
        o.kind = 7;
        let mut s = ptr::null_mut();
        assert_eq!(rs_new(&o, &mut s), RS_INVALID_INPUT);
        assert!(s.is_null());
        // thresholds must be finite (a NaN would switch the rcond check off)
        rs_default_options(&mut o);
        o.min_rcond = f64::NAN;
        assert_eq!(rs_new(&o, &mut s), RS_INVALID_INPUT);
        rs_default_options(&mut o);
        o.singular_pivot_ratio = f64::INFINITY;
        assert_eq!(rs_new(&o, &mut s), RS_INVALID_INPUT);
        // -inf must not pass for the negative "default" sentinel, and
        // symmetry_tol has no sentinel: it must be finite and >= 0
        for (field, v) in [(0, f64::NEG_INFINITY), (1, f64::NEG_INFINITY), (2, f64::NAN),
                           (2, f64::NEG_INFINITY), (2, f64::INFINITY), (2, -1.0)] {
            rs_default_options(&mut o);
            match field {
                0 => o.min_rcond = v,
                1 => o.singular_pivot_ratio = v,
                _ => o.symmetry_tol = v,
            }
            assert_eq!(rs_new(&o, &mut s), RS_INVALID_INPUT, "field {field} = {v}");
        }
        assert!(s.is_null());
        // a finite negative threshold is the default
        rs_default_options(&mut o);
        o.min_rcond = -1.0;
        o.singular_pivot_ratio = -1.0;
        assert_eq!(rs_new(&o, &mut s), RS_OK);
        rs_free(s);
        s = ptr::null_mut();
        assert_eq!(rs_new(ptr::null(), &mut s), RS_OK);
        let mut b = vec![1.0; 4];
        assert_eq!(rs_solve(s, 1, b.as_mut_ptr(), 4), RS_NOT_FACTORIZED);
        let cp: Vec<i64> = vec![0, 1, -2];
        let ri: Vec<i64> = vec![0];
        let v = vec![1.0];
        assert_eq!(rs_update_i64(s, 2, cp.as_ptr(), ri.as_ptr(), v.as_ptr()), RS_INVALID_INPUT);
        assert_eq!(rs_update_i64(s, 2, ptr::null(), ri.as_ptr(), v.as_ptr()), RS_INVALID_INPUT);
        assert_eq!(rs_solve(ptr::null_mut(), 1, b.as_mut_ptr(), 4), RS_INVALID_INPUT);
        assert_eq!(CStr::from_ptr(rs_status_name(RS_PATTERN_CHANGED)).to_str().unwrap(), "PATTERN_CHANGED");
        assert_eq!(CStr::from_ptr(rs_status_name(99)).to_str().unwrap(), "UNKNOWN");
        rs_free(s);
        rs_free(ptr::null_mut());
    }
}

#[test]
fn compressed_rows_lu_from_i64_arrays() {
    // [[4, 1], [2, 3]] as CSR: rows (4, 1), (2, 3)
    unsafe {
        let mut o = std::mem::zeroed::<RsOptions>();
        rs_default_options(&mut o);
        o.kind = 2;
        o.compressed_rows = 1;
        let mut s = ptr::null_mut();
        assert_eq!(rs_new(&o, &mut s), RS_OK);
        let rp: Vec<i64> = vec![0, 2, 4];
        let ci: Vec<i64> = vec![0, 1, 0, 1];
        let v = vec![4.0, 1.0, 2.0, 3.0];
        assert_eq!(rs_update_i64(s, 2, rp.as_ptr(), ci.as_ptr(), v.as_ptr()), RS_OK);
        let mut b = vec![5.0, 5.0]; // A * (1, 1)
        assert_eq!(rs_solve(s, 1, b.as_mut_ptr(), 2), RS_OK);
        assert!((b[0] - 1.0).abs() < 1e-14 && (b[1] - 1.0).abs() < 1e-14, "{b:?}");
        rs_free(s);
    }
}
