//! Symbolic analyses shared between solvers of one process.
//!
//! pyGIMLi builds a new solver for every matrix it assembles (one per 2.5D
//! wavenumber and forward run), all on the same mesh. The analysis depends on
//! the pattern alone, so a solver may take one made by another solver for an
//! identical pattern: same options, same n, same pointers and indices,
//! compared entry by entry. Numeric factors are never shared.

use std::sync::{Arc, Mutex};

use crate::{Analysis, AnalysisKey};

/// Analyses kept: a forward run alternates between at most a couple of
/// patterns (the forward mesh, the P2 primary mesh).
const CAPACITY: usize = 2;

static CACHE: Mutex<Vec<Arc<Analysis>>> = Mutex::new(Vec::new());

pub(crate) fn find(key: AnalysisKey, n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Option<Arc<Analysis>> {
    let mut c = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    let pos = c.iter().position(|a| a.key == key && a.same_pattern(n, col_ptr, row_idx))?;
    // most recently used last
    let a = c.remove(pos);
    c.push(a.clone());
    Some(a)
}

pub(crate) fn put(a: Arc<Analysis>) {
    let mut c = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if c.len() >= CAPACITY {
        c.remove(0);
    }
    c.push(a);
}

/// Drop every shared analysis (frees their memory).
pub fn clear_shared_analyses() {
    CACHE.lock().unwrap_or_else(|p| p.into_inner()).clear();
}
