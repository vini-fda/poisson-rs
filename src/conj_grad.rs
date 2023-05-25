use crate::csr::{CSRMatrix, norm_squared, dot_product};

/// Solves the linear system Ax = b using the conjugate gradient method.
/// This function in optimized for a sparse matrix A in CSR format.
pub fn conjugate_gradient(a: &CSRMatrix, b: &[f32], x: &mut [f32]) {
    let mut r = vec![0.0; b.len()];
    a.vmul(x, &mut r);
    for i in 0..b.len() {
        r[i] = b[i] - r[i];
    }
    const TOL: f32 = 1e-10;
    if norm_squared(&r) < TOL {
        return;
    }
    let mut p = r.clone();
    let mut rnew = r.clone();
    for _ in 0..a.nrows() {
        let mut q = vec![0.0; b.len()];
        a.vmul(&p, &mut q); // q = Ap
        let alpha = norm_squared(&r) / dot_product(&p, &q);
        for i in 0..b.len() {
            x[i] += alpha * p[i];
            rnew[i] -= alpha * q[i];
        }
        if norm_squared(&rnew) < TOL {
            return;
        }
        let beta = norm_squared(&rnew) / norm_squared(&r);
        r = rnew.clone();
        for i in 0..b.len() {
            p[i] = rnew[i] + beta * p[i];
        }
    }
}