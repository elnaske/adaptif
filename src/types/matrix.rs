#![allow(dead_code, reason = "TODO")]

use crate::types::Float;
use num_traits::Zero as _;

pub trait MatrixOps<'a> {
    type Item: Float;

    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item>;
    fn shape(&self) -> Shape;

    fn transpose(&'a self) -> Transpose<'a, Self>
    where
        Self: Sized, // TODO: should the trait itself require Sized?
    {
        Transpose { matrix: self }
    }

    // TODO: replace Option with Result
    fn matmul<'rhs, M>(&self, rhs: &M, out: &mut [Self::Item]) -> Option<()>
    where
        M: MatrixOps<'rhs, Item = Self::Item>,
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
                out[i * n_cols + j] = Self::Item::zero();
            }
        }

        for i in 0..n_rows {
            for k in 0..inner {
                let lhs = *self.get_2d(i, k)?;

                #[allow(clippy::indexing_slicing, reason = "inputs validated")]
                for j in 0..n_cols {
                    out[i * n_cols + j] += lhs * *rhs.get_2d(k, j)?;
                }
            }
        }

        Some(())
    }
}

#[derive(Debug, Clone)]
pub struct Transpose<'a, M>
where
    M: MatrixOps<'a>,
{
    matrix: &'a M,
}

impl<'a, M: MatrixOps<'a>> MatrixOps<'_> for Transpose<'a, M> {
    type Item = M::Item;

    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item> {
        self.matrix.get_2d(col, row)
    }

    fn shape(&self) -> Shape {
        Shape {
            rows: self.matrix.shape().cols,
            cols: self.matrix.shape().rows,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}
impl Shape {
    pub fn new(rows: usize, cols: usize) -> Option<Self> {
        if rows == 0 || cols == 0 {
            None
        } else {
            Some(Self { rows, cols })
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
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
impl<F: Float> MatrixOps<'_> for Matrix<F> {
    type Item = F;

    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item> {
        if row >= self.shape.rows || col >= self.shape.cols {
            None
        } else {
            self.inner.get(row * self.shape.cols + col)
        }
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
        let mut mat = Matrix::new(Shape::new(2, 3).unwrap());
        mat.inner = (1..=6).into_iter().map(f64::from).collect();
        mat
    }

    #[test]
    fn get_2d_works() {
        let mat = matrix();

        assert!(approx_equal(*mat.get_2d(1, 2).unwrap(), 6.0, 1e-6));
    }

    #[test]
    fn get_2d_bounds() {
        let mat = matrix();

        for (r, c) in [
            (0, 0),
            (mat.shape().rows - 1, 0),
            (0, mat.shape().cols - 1),
            (mat.shape().rows - 1, mat.shape().cols - 1),
        ] {
            mat.get_2d(r, c).unwrap();
        }

        for (r, c) in [
            (mat.shape().rows, 0),
            (0, mat.shape().cols),
            (mat.shape().rows, mat.shape().cols),
        ] {
            assert!(mat.get_2d(r, c).is_none());
        }
    }

    #[test]
    fn shape_works() {
        let shape = Shape::new(2, 3).unwrap();
        assert_eq!(shape.rows(), 2);
        assert_eq!(shape.cols(), 3);

        let mat: Matrix<f64> = Matrix::new(shape);
        assert_eq!(mat.shape(), shape);

        for (r, c) in [(0, 0), (0, 1), (1, 0)] {
            assert!(Shape::new(r, c).is_none());
        }
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

    #[test]
    fn matmul_validate_shapes() {
        let mat = matrix();
        let mut res = [0.0; 4];

        // [2,3] * [3,4] -> [2,4]
        mat.matmul(&Matrix::new(Shape::new(3, 4).unwrap()), &mut [0.0; 2 * 4])
            .unwrap();

        // [3,2] * [2,5] -> [3,5]
        mat.transpose()
            .matmul(&Matrix::new(Shape::new(2, 5).unwrap()), &mut [0.0; 3 * 5])
            .unwrap();

        // [2,3] * [2,3] -> inner values don't match
        assert!(mat.matmul(&mat, &mut res).is_none());

        // [3,2] * [3,2] -> inner values don't match
        assert!(mat.transpose().matmul(&mat.transpose(), &mut res).is_none());
    }

    #[test]
    fn matmul_validate_out_buf() {
        let mat = matrix();
        let rhs = mat.transpose();

        let mut too_small = [0.0; 3];
        let mut right_size = [0.0; 4];
        let mut too_big = [0.0; 5];

        mat.matmul(&rhs, &mut right_size).unwrap();

        assert!(mat.matmul(&rhs, &mut too_small).is_none());
        assert!(mat.matmul(&rhs, &mut too_big).is_none());
    }
}
