pub struct CSRMatrix {
    pub values: Vec<f32>,
    pub col_ind: Vec<usize>,
    pub row_ptr: Vec<usize>,
}

impl CSRMatrix {
    pub fn new(values: Vec<f32>, col_ind: Vec<usize>, row_ptr: Vec<usize>) -> Self {
        Self {
            values,
            col_ind,
            row_ptr,
        }
    }

    /// Multiplies the sparse matrix by dense vector v and stores the result in `result`.
    pub fn vmul(&self, v: &[f32], result: &mut [f32]) {
        // assumes the number of rows in the matrix is equal to the length of v
        // and the length of result.
        for (i, y) in result.iter_mut().enumerate() {
            *y = 0.0;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                *y += self.values[k] * v[self.col_ind[k]];
            }
        }
    }

    pub fn nrows(&self) -> usize {
        self.row_ptr.len() - 1
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    // assumes a.len() == b.len()
    let n = a.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// L2 squared norm of a vector.
pub fn norm_squared(a: &[f32]) -> f32 {
    dot_product(a, a)
}

/// Returns a sparse matrix for the 2D Poisson equation with Dirichlet boundary conditions.
/// dt is the time step, h is the grid spacing, and alpha is the diffusion coefficient.
/// 
/// The discretization is implicit Euler, and assumes a square grid with m x m grid points.
pub fn poisson_matrix(m: usize, dt: f32, h: f32, alpha: f32) -> CSRMatrix {
    let k = alpha * dt / (h*h);
    let diag = 1.0 + 4.0 * k;
    let off_diag = -k;
    let nnz = m * (5 * m - 2) - 2;
    let mut values = Vec::with_capacity(nnz);
    let mut col_ind = Vec::with_capacity(nnz);
    let mut row_ptr = Vec::with_capacity(m*m + 1);
    let mut index = 0;
    for i in 0..(m*m) {
        row_ptr.push(index);
        // i,i-m -> off-diagonal entry with -k
        if i >= m {
            values.push(off_diag);
            col_ind.push(i - m);
            index += 1;
        }
        // i,i-1 -> off-diagonal entry with -k
        if i >= 1 {
            values.push(off_diag);
            col_ind.push(i - 1);
            index += 1;
        }
        // i,i -> diagonal entry with 1 + 4k
        values.push(diag);
        col_ind.push(i);
        index += 1;
        // i,i+1 -> off-diagonal entry with -k
        if i < m*m - 1 {
            values.push(off_diag);
            col_ind.push(i + 1);
            index += 1;
        }
        // i,i+m -> off-diagonal entry with -k
        if i < m * (m - 1) {
            values.push(off_diag);
            col_ind.push(i + m);
            index += 1;
        }

    }
    row_ptr.push(nnz);
    CSRMatrix::new(values, col_ind, row_ptr)
}



#[cfg(test)]
mod tests {
    #[test]
    fn test_vmul_zero() {
        use crate::csr::poisson_matrix;
        let m = 100;
        let h = 0.1;
        let dt = 0.1;
        let alpha = 1.0;
        let a = poisson_matrix(m, dt, h, alpha);
        // m x m random values
        let x = vec![0.0; m*m];
        let mut b = vec![1.0; m*m];
        // b = Ax
        a.vmul(&x, &mut b);
        // check that b = 0
        assert!(b.iter().all(|&v| v == 0.0));
    }
}