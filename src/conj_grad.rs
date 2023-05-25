use crate::csr::{CSRMatrix, norm_squared, dot_product};

/// Solves the linear system Ax = b using the conjugate gradient method.
/// This function in optimized for a sparse matrix A in CSR format.
pub fn conjugate_gradient(a: &CSRMatrix, b: &[f32], x: &mut [f32]) {
    let n = b.len();
    let mut r = vec![0.0; n];
    a.vmul(x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }
    const TOL: f32 = 1e-10;
    if norm_squared(&r) < TOL {
        return;
    }
    let mut p = r.clone();
    let mut rnew = r.clone();
    let mut q = vec![0.0; n];
    for _ in 0..a.nrows() {
        a.vmul(&p, &mut q); // q = Ap
        let alpha = norm_squared(&r) / dot_product(&p, &q);
        for i in 0..n {
            x[i] += alpha * p[i];
            rnew[i] = r[i] - alpha * q[i];
        }
        if norm_squared(&rnew) < TOL {
            return;
        }
        let beta = norm_squared(&rnew) / norm_squared(&r);
        r[..n].copy_from_slice(&rnew[..n]);
        for i in 0..n {
            p[i] = rnew[i] + beta * p[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::ThreadRng, Rng};

    fn random_vector(n: usize, rng: &mut ThreadRng) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(rng.gen_range(0.0..=1.0));
        }
        v
    }
    #[test]
    fn test_conj_grad() {
        use crate::csr::poisson_matrix;
        use crate::conj_grad::conjugate_gradient;
        let mut rng = rand::thread_rng();
        let m = 100;
        let dt = 0.01;
        let alpha = 1.0;
        let a = poisson_matrix(m, dt, alpha);
        let mut b = vec![0.0; m*m];
        // m x m random values
        let x_soln = random_vector(m*m, &mut rng);
        // b = Ax
        a.vmul(&x_soln, &mut b);
        let b = b;
        let mut x = vec![0.0; m*m];
        println!("Solving with conjugate gradient...");
        conjugate_gradient(&a, &b, &mut x);
        let mut err = 0.0;
        for i in 0..m*m {
            err += (x[i] - x_soln[i]) * (x[i] - x_soln[i]);
        }
        err = err.sqrt();
        const MAX_ERR: f32 = 1e-4;
        println!("Absolute error: {}", err);
        assert!(err < MAX_ERR);
    }
}