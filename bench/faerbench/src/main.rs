//! faerbench: rustsolve on sparse systems exported from finite-element models.
//!
//! faerbench bench <case_dir> <out_dir> [--threads N] [--repeats R] [--variants a,b]
//!
//! A case directory holds meta.json, indptr.i64, indices.i64, data.f64 (full
//! symmetric CSR) and rhs.f64 (n x nrhs, column-major); see bench/python/caseio.py.
//! One JSON line per (case, variant) goes to <out_dir>/results.jsonl and the
//! solutions to <out_dir>/<case>__<variant>.f64.

use rustsolve::{Error, Kind, Nodal, Options, Ordering, Solver};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

struct Case {
    name: String,
    n: usize,
    nrhs: usize,
    cp: Vec<usize>,
    ri: Vec<usize>,
    v: Vec<f64>,
    rhs: Vec<f64>,
}

fn read_i64(p: &Path) -> Vec<usize> {
    let b = fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect()
}

fn read_f64(p: &Path) -> Vec<f64> {
    let b = fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn write_f64(p: &Path, v: &[f64]) {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(p, b).unwrap();
}

fn meta_usize(meta: &str, key: &str) -> usize {
    let k = format!("\"{key}\":");
    let i = meta.find(&k).unwrap_or_else(|| panic!("no {key} in meta")) + k.len();
    meta[i..].trim_start().chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
}

fn load(dir: &Path) -> Case {
    let meta = fs::read_to_string(dir.join("meta.json")).unwrap();
    let (n, nrhs) = (meta_usize(&meta, "n"), meta_usize(&meta, "nrhs"));
    let rhs = read_f64(&dir.join("rhs.f64"));
    assert_eq!(rhs.len(), n * nrhs);
    let name = format!(
        "{}__{}",
        dir.parent().unwrap().file_name().unwrap().to_string_lossy(),
        dir.file_name().unwrap().to_string_lossy()
    );
    Case { name, n, nrhs, cp: read_i64(&dir.join("indptr.i64")), ri: read_i64(&dir.join("indices.i64")), v: read_f64(&dir.join("data.f64")), rhs }
}

/// max over columns of ||A x - b|| / ||b|| (A symmetric: CSR == CSC)
fn relres(c: &Case, vals: &[f64], x: &[f64], b: &[f64], nrhs: usize) -> f64 {
    let n = c.n;
    let mut worst: f64 = 0.0;
    for k in 0..nrhs {
        let (xk, bk) = (&x[k * n..(k + 1) * n], &b[k * n..(k + 1) * n]);
        let (mut rn, mut bn) = (0.0, 0.0);
        for i in 0..n {
            let s: f64 = (c.cp[i]..c.cp[i + 1]).map(|p| vals[p] * xk[c.ri[p]]).sum();
            rn += (s - bk[i]).powi(2);
            bn += bk[i] * bk[i];
        }
        let v = if bn > 0.0 { (rn / bn).sqrt() } else { rn.sqrt() };
        worst = if v.is_nan() || worst.is_nan() { f64::NAN } else { worst.max(v) };
    }
    worst
}

fn opts(variant: &str, threads: usize) -> Options {
    let (kind, nodal, ordering) = match variant {
        "llt_auto" => (Kind::Llt, Nodal::Auto, Ordering::Amd),
        "llt_simplicial" => (Kind::Llt, Nodal::Simplicial, Ordering::Amd),
        "llt_supernodal" => (Kind::Llt, Nodal::Supernodal, Ordering::Amd),
        "llt_nd" => (Kind::Llt, Nodal::Auto, Ordering::NestedDissection),
        "ldlt_auto" => (Kind::Ldlt, Nodal::Auto, Ordering::Amd),
        "lu_auto" => (Kind::Lu, Nodal::Auto, Ordering::Amd),
        v => panic!("unknown variant {v}"),
    };
    // compare the factorization itself: the LU rcond check runs separately below
    Options { kind, nodal, ordering, factor_threads: threads, solve_threads: threads, min_rcond: Some(0.0), ..Options::default() }
}

fn err_json(e: &Error) -> String {
    format!("{{\"code\":\"{}\",\"detail\":\"{}\"}}", e.name(), e.to_string().replace('"', "'"))
}

fn fmt(x: f64) -> String {
    if x.is_finite() { format!("{x:.6e}") } else { format!("\"{x}\"") }
}

fn min_time<F: FnMut() -> R, R>(repeats: usize, mut f: F) -> (f64, R) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..repeats {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(r);
    }
    (best, out.unwrap())
}

fn bench(c: &Case, out: &Path, variants: &[String], threads: usize, repeats: usize) {
    let mut log = fs::OpenOptions::new().create(true).append(true).open(out.join("results.jsonl")).unwrap();
    for v in variants {
        let o = opts(v, threads);
        let mut line = format!(
            "{{\"case\":\"{}\",\"variant\":\"{v}\",\"threads\":{threads},\"n\":{},\"nnz\":{},\"nrhs\":{}",
            c.name, c.n, c.ri.len(), c.nrhs
        );
        let (t_an, (mut s, r)) = min_time(repeats, || {
            let mut s = Solver::new(o).unwrap();
            let r = s.analyze(c.n, &c.cp, &c.ri);
            (s, r)
        });
        line += &format!(",\"t_analyze\":{t_an:.6e}");
        let emit = |line: String, log: &mut fs::File| {
            writeln!(log, "{line}").unwrap();
            println!("{line}");
        };
        if let Err(e) = r {
            emit(line + &format!(",\"error_stage\":\"analyze\",\"error\":{}}}", err_json(&e)), &mut log);
            continue;
        }
        let t = Instant::now();
        let f = s.factorize(&c.cp, &c.ri, &c.v).map(|i| i.clone());
        line += &format!(",\"t_factor\":{:.6e}", t.elapsed().as_secs_f64());
        let info = match f {
            Ok(i) => i,
            Err(e) => {
                let mut x = c.rhs.clone();
                let after = s.solve(&mut x, c.nrhs).map(|_| "OK").unwrap_or_else(|e| e.name());
                emit(line + &format!(",\"error_stage\":\"factorize\",\"error\":{},\"solve_after_error\":\"{after}\"}}", err_json(&e)), &mut log);
                continue;
            }
        };
        let (t_refac, _) = min_time(repeats, || s.factorize(&c.cp, &c.ri, &c.v).map(|_| ()).unwrap());
        let (t_block, (x, solve_err)) = min_time(repeats, || {
            let mut x = c.rhs.clone();
            let r = s.solve(&mut x, c.nrhs);
            (x, r)
        });
        let (t_cols, _) = min_time(repeats, || {
            let mut x = c.rhs.clone();
            for k in 0..c.nrhs {
                let _ = s.solve(&mut x[k * c.n..(k + 1) * c.n], 1);
            }
            x
        });
        let rr = relres(c, &c.v, &x, &c.rhs, c.nrhs);
        let t = Instant::now();
        let rc = s.rcond_estimate().unwrap_or(f64::NAN);
        let t_rc = t.elapsed().as_secs_f64();
        line += &format!(
            ",\"t_refactor\":{t_refac:.6e},\"t_solve_block\":{t_block:.6e},\"t_solve_cols\":{t_cols:.6e},\"len_l\":{},\"supernodal\":{},\"min_pivot\":{},\"max_pivot\":{},\"pivot_ratio\":{},\"neg_pivots\":{},\"rcond_est\":{},\"t_rcond\":{t_rc:.6e},\"relres\":{}",
            info.nnz_factor, info.supernodal, fmt(info.min_pivot), fmt(info.max_pivot), fmt(info.pivot_ratio), info.neg_pivots, fmt(rc), fmt(rr)
        );
        if let Err(e) = solve_err {
            line += &format!(",\"error_stage\":\"solve\",\"error\":{}", err_json(&e));
        }
        write_f64(&out.join(format!("{}__{v}.f64", c.name)), &x);
        emit(line + "}", &mut log);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut threads, mut repeats) = (1usize, 3usize);
    let mut variants: Vec<String> = ["llt_auto", "llt_nd", "ldlt_auto", "lu_auto"].iter().map(|s| s.to_string()).collect();
    let mut pos = vec![];
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                threads = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--repeats" => {
                repeats = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--variants" => {
                variants = args[i + 1].split(',').map(|s| s.to_string()).collect();
                i += 2;
            }
            a => {
                pos.push(a.to_string());
                i += 1;
            }
        }
    }
    match pos.first().map(|s| s.as_str()) {
        Some("bench") => {
            let c = load(Path::new(&pos[1]));
            let out = Path::new(&pos[2]);
            fs::create_dir_all(out).unwrap();
            bench(&c, out, &variants, threads, repeats);
        }
        _ => {
            eprintln!("usage: faerbench bench <case_dir> <out_dir> [--threads N] [--repeats R] [--variants a,b]");
            std::process::exit(2);
        }
    }
}
