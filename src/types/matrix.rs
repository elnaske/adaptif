#![allow(dead_code, clippy::unwrap_used, reason = "TODO")]

use std::marker::PhantomData;

use crate::types::Float;

pub trait MatrixOps<'a, F: Float> {
    fn get_2d(&self, row: usize, col: usize) -> Option<&F>;
    fn shape(&self) -> Shape;

    fn transpose(&'a self) -> Transpose<'a, Self, F>
    where
        Self: Sized, // TODO: should the trait require Sized?
    {
        Transpose {
            matrix: self,
            unused: PhantomData,
        }
    }

    // TODO: replace Option with Result
    fn matmul<'rhs, M>(&self, rhs: &M, out: &mut [F]) -> Option<()>
    where
        M: MatrixOps<'rhs, F>,
    {
        if self.shape().cols != rhs.shape().rows {
            return None;
        }

        if out.len() != self.shape().rows * rhs.shape().cols {
            return None;
        }

        let (n_rows, inner, n_cols) = (self.shape().rows, self.shape().cols, rhs.shape().cols);

        for i in 0..n_rows {
            #[allow(clippy::indexing_slicing, reason = "output shape validated")]
            for j in 0..n_cols {
                out[i * n_cols + j] = F::zero();
            }
        }

        for i in 0..n_rows {
            for k in 0..inner {
                let lhs = *self.get_2d(i, k).unwrap();

                #[allow(clippy::indexing_slicing, reason = "inputs validated")]
                for j in 0..n_cols {
                    out[i * n_cols + j] += lhs * *rhs.get_2d(k, j).unwrap();
                }
            }
        }

        Some(())
    }
}

#[derive(Debug, Clone)]
pub struct Transpose<'a, M, F>
where
    M: MatrixOps<'a, F>,
    F: Float,
{
    matrix: &'a M,
    unused: PhantomData<F>,
}

impl<'a, M: MatrixOps<'a, F>, F: Float> MatrixOps<'_, F> for Transpose<'a, M, F> {
    fn get_2d(&self, row: usize, col: usize) -> Option<&F> {
        self.matrix.get_2d(col, row)
    }

    fn shape(&self) -> Shape {
        Shape {
            rows: self.matrix.shape().cols,
            cols: self.matrix.shape().rows,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

#[derive(Debug, Clone)]
pub struct Matrix<F: Float> {
    inner: Box<[F]>,
    shape: Shape,
}
impl<F: Float> Matrix<F> {
    pub fn new(shape: Shape) -> Self {
        Self {
            inner: vec![F::zero(); shape.rows * shape.cols].into_boxed_slice(),
            shape,
        }
    }
}
impl<F: Float> MatrixOps<'_, F> for Matrix<F> {
    fn get_2d(&self, row: usize, col: usize) -> Option<&F> {
        self.inner.get(row * self.shape.cols + col)
    }

    fn shape(&self) -> Shape {
        self.shape
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::{all_approx_equal, approx_equal};

    fn matrix() -> Matrix<f64> {
        let mut mat = Matrix::new(Shape { rows: 2, cols: 3 });
        mat.inner = (1..=6).into_iter().map(f64::from).collect();
        mat
    }
    #[test]
    fn get_2d_works() {
        let mat = matrix();

        assert!(approx_equal(*mat.get_2d(1, 2).unwrap(), 6.0, 1e-6));
    }

    #[test]
    fn transpose_works() {
        let mat = matrix();

        assert!(approx_equal(
            *mat.get_2d(0, 1).unwrap(),
            *mat.transpose().get_2d(1, 0).unwrap(),
            1e-6
        ));
    }

    #[test]
    fn matmul_works() {
        let mat = matrix();
        let rhs = mat.transpose();

        let mut res = [0.0; 4];

        let _ = mat.matmul(&rhs, &mut res);

        assert!(all_approx_equal(
            res.iter(),
            [14.0, 32.0, 32.0, 77.0].iter()
        ));
    }
}
