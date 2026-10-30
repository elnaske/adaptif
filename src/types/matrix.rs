#![allow(dead_code, reason = "TODO")]

use std::ops::{Deref, DerefMut, DivAssign, Index, IndexMut};

use crate::types::Float;
use num_traits::Zero as _;

pub trait MatrixOps<'a> {
    type Item: Float;

    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item>;
    fn shape(&self) -> Shape;

    fn transpose(&'a self) -> Transpose<'a, Self>
    where
        Self: Sized,
    {
        Transpose { matrix: self }
    }

    // TODO: replace Option with Result
    fn matmul<'rhs, M>(&'a self, rhs: &M, out: &mut [Self::Item]) -> Option<()>
    where
        M: MatrixOps<'rhs, Item = Self::Item>,
    {
        if self.shape().cols != rhs.shape().rows {
            return None;
        }

        if out.len() != self.shape().rows * rhs.shape().cols {
            return None;
        }

        out.fill(Self::Item::zero());

        let (n_rows, inner, n_cols) = (self.shape().rows, self.shape().cols, rhs.shape().cols);

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

pub trait MatrixOpsMut<'a>: MatrixOps<'a> {
    fn get_2d_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Item>;

    /// In place element-wise subtraction with another matrix.
    ///
    /// Note that we can't use ``Sub`` or ``SubAssign`` for this because the dimensions
    /// of the two matrices have to match, so we need to return a Result or Option,
    /// which ``Sub::sub()`` doesn't.
    fn sub<'rhs, M>(&mut self, rhs: &M) -> Option<()>
    where
        M: MatrixOps<'rhs, Item = Self::Item>,
    {
        if self.shape() == rhs.shape() {
            let shape = self.shape();
            for row in 0..shape.rows {
                for col in 0..shape.cols {
                    *self.get_2d_mut(row, col)? -= *rhs.get_2d(row, col)?;
                }
            }
            Some(())
        } else {
            None
        }
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

#[derive(Debug, Clone, PartialEq)]
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

    pub fn from_slice(arr: &[F], shape: Shape) -> Option<Self> {
        // if arr.len() == shape.rows * shape.cols {
        //     Some(Self {
        //         inner: arr.to_vec().into_boxed_slice(),
        //         shape,
        //     })
        // } else {
        //     None
        // }

        (arr.len() == shape.rows * shape.cols).then(|| Self {
            inner: arr.to_vec().into_boxed_slice(),
            shape,
        })
    }

    #[cfg(test)]
    pub fn inner(&self) -> &[F] {
        &self.inner
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
impl<F: Float> MatrixOpsMut<'_> for Matrix<F> {
    fn get_2d_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Item> {
        if row >= self.shape.rows || col >= self.shape.cols {
            None
        } else {
            self.inner.get_mut(row * self.shape.cols + col)
        }
    }
}
impl<F: Float> DivAssign<F> for Matrix<F> {
    fn div_assign(&mut self, rhs: F) {
        #[allow(clippy::indexing_slicing, reason = "i < self.inner.len()")]
        for i in 0..self.inner.len() {
            self.inner[i] /= rhs;
        }
    }
}

// TODO: matmul's out param should really impl MatrixOpsMut, but the lifetimes for that are weird,
// so this is a temporary workaround
impl<F: Float> Deref for Matrix<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for Matrix<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
impl<F: Float> Index<MatIdx> for Matrix<F> {
    type Output = F;

    #[allow(clippy::indexing_slicing, reason = "Indexing should panic OOB")]
    fn index(&self, index: MatIdx) -> &Self::Output {
        &self.inner[index.0 * self.shape.cols + index.1]
    }
}
impl<F: Float> IndexMut<MatIdx> for Matrix<F> {
    #[allow(clippy::indexing_slicing, reason = "Indexing should panic OOB")]
    fn index_mut(&mut self, index: MatIdx) -> &mut Self::Output {
        &mut self.inner[index.0 * self.shape.cols + index.1]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MatIdx(pub usize, pub usize);

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
