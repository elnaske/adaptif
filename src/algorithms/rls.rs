use std::ops::{Deref, DerefMut};

use crate::types::buffers::NoiseWindow;
use crate::types::matrix::{Matrix, MatrixOps as _, MatrixOpsMut as _, Shape};
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, Float, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone, PartialEq)]
/// Recursive least squares algorithm.
pub struct Rls<F: Float> {
    /// Forgetting factor, often referred to as `lambda`, used for weight updates.
    forgetting_factor: F,
    /// Positive scalar value, often referred to as `delta`, used for initalizing the inverse correlation `p_matrix`.
    p_init_scale: F,
    /// Inverse correlation matrix, referred to as P[n], for RLS updates. Size is determined by the filter's
    /// window size. The matrix size is M x M, where M corresponds to the filter's window size.
    inverse_corr_matrix: InverseCorrMatrix<F>,
    /// Kalman Gain vector used in updating filter coefficients
    /// Initialized as none because past history is unnecessary. Size is determined by the filter's
    /// window size.
    kalman_gain: KalmanGain<F>,
}

/// Updates the filter weights using the following algorithm.
///
/// The Kalman gain vector, ``k_n`` is calculated as:
///
/// $``k_n`` = \frac{P_{n-1} ``x_n``}
/// {\lambda + ``x_n^T`` P_{n-1} ``x_n``}$.
///
/// The filter weights are updated as:
///
/// $``w_n`` = w_{n-1} + ``k_n`` ``e_n``$.
///
/// The inverse correlation matrix is updated as:
///
/// $``P_n`` = \frac{1}{\lambda}
/// \left(P_{n-1} - ``k_n`` ``x_n^T`` P_{n-1}\right)$.
///
/// where `e_n` is the scalar error for the current sample,
/// and $``x_n``$ is a vector of length `window_size` of the
/// most recent noise reference samples.
impl<F: Float> Algorithm<F> for Rls<F> {
    fn update_step(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_window: NoiseWindow<F>,
    ) {
        // Updates p_matrix on first iteration once n is known
        // No scenario where one is initialized and the other isn't
        // TODO: remove once proper init is implemented
        // if self.inverse_corr_matrix.is_empty() {
        if self.inverse_corr_matrix.shape().rows() != *weights.window_size() {
            self.inverse_corr_matrix =
                InverseCorrMatrix::new(weights.window_size(), self.p_init_scale);
            self.kalman_gain = KalmanGain::new(weights.window_size());
        }

        self.kalman_gain.update(
            &self.inverse_corr_matrix,
            &noise_window,
            self.forgetting_factor,
        );

        for (w, k) in weights.iter_mut().zip(self.kalman_gain.iter().copied()) {
            *w += k * (*error);
        }

        self.inverse_corr_matrix
            .update(&self.kalman_gain, &noise_window, self.forgetting_factor);
    }
}

impl<F: Float> Rls<F> {
    /// # Errors
    ///
    /// Returns an error if forgetting factor <= 0.0 or > 1.0.
    /// Returns an error if init scale <= 0.0.
    #[allow(clippy::missing_panics_doc, reason = "TODO")]
    pub fn new(forgetting_factor: F, p_init_scale: F) -> Result<Self> {
        if forgetting_factor <= F::zero() || forgetting_factor > F::one() {
            return Err(Error::InvalidForgettingFactorRange);
        }

        if p_init_scale <= F::zero() {
            return Err(Error::NonPositivePInitScale);
        }
        Ok(Rls {
            forgetting_factor,
            p_init_scale,
            inverse_corr_matrix: InverseCorrMatrix {
                inner: Matrix::new(Shape::new(1, 1).expect("TODO")),
            },
            kalman_gain: KalmanGain {
                inner: Matrix::new(Shape::new(1, 1).expect("TODO")),
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
/// M-dimensional vector of Kalman gains, where M is the filter's window size.
struct KalmanGain<F: Float> {
    inner: Matrix<F>,
}
impl<F: Float> KalmanGain<F> {
    pub fn new(window_size: WindowSize) -> Self {
        KalmanGain {
            inner: Matrix::new(Shape::new(*window_size, 1).expect("window_size >= 1")),
        }
    }

    /// Updates the Kalman gain vector used in the RLS weight update.
    ///
    /// Kalman gain vector update calculated as:
    ///
    /// $``k_n`` = \frac{P_{n-1} ``x_n``}
    /// {\lambda + ``x_n^T`` P_{n-1} ``x_n``}$.
    ///
    /// Existing Kalman gain buffer gets reused to store the numerator
    /// value in update calculation to avoid reallocating a new vector.
    pub fn update(
        &mut self,
        inverse_corr_matrix: &InverseCorrMatrix<F>,
        noise_ref: &NoiseWindow<F>,
        lambda: F,
    ) {
        // We reuse the kalman gain buffer to store the numerator
        // to avoid an extra allocation for the output parameter of the matmul.
        inverse_corr_matrix
            .matmul(noise_ref, self)
            .expect("[window_size, window_size] @ [window_size, 1] -> [window_size, 1]");

        let xt_p_x = self.iter().zip(noise_ref.iter()).map(|(p, x)| *p * x).sum();

        self.inner /= lambda + xt_p_x;
    }
}
impl<F: Float> Deref for KalmanGain<F> {
    type Target = Matrix<F>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F: Float> DerefMut for KalmanGain<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Inverse Correlation Matrix with shape M * M, where M is the filter's window size.
struct InverseCorrMatrix<F: Float> {
    inner: Matrix<F>,
}
impl<F: Float> InverseCorrMatrix<F> {
    /// Creates an M x M-dimensional Identity matrix multiplied by a positive scalar `p_init_scale`.
    /// I.e. `p_init_scale` is the resulting value along main diagonal and all other values are initialized
    /// to zero.
    ///
    /// Serves as inverse correlation matrix in RLS algorithm, where M is the filter's window size.
    pub fn new(window_size: WindowSize, p_init_scale: F) -> Self {
        let mut p = Matrix::new(
            Shape::new(*window_size, *window_size)
                .expect("WindowSize type ensures window_size >= 1"),
        );

        for i in 0..(*window_size) {
            if let Some(elem) = p.get_2d_mut(i, i) {
                *elem = p_init_scale;
            }
        }

        InverseCorrMatrix { inner: p }
    }

    /// Updates the inverse correlation matrix used in the RLS weight update.
    /// Utilizes current Kalman gain ``k_n``.
    ///
    /// Inverse correlation matrix update calculated as:
    ///
    /// $``P_n`` = \frac{1}{\lambda}
    /// \left(P_{n-1} - ``k_n`` ``x_n^T`` P_{n-1}\right)$.
    ///
    /// The matrix is stored as a flat buffer and updated in place.
    fn update(&mut self, kalman_gain: &KalmanGain<F>, noise_ref: &NoiseWindow<F>, lambda: F) {
        // We calculate [x^T_n p_{n-1}] by computing the dot product of
        // ``noise_ref`` with each columnn in ``inverse_corr_matrix``
        //
        // NOTE(elias): tried using MatrixOps::matmul() for this and
        // execution time for RLS literally doubled. This is likely because
        // we couldn't modify P in place, so we have a bunch of extra calculations
        // that we need to carry out and store, whereas the approach below
        // simplifies the overall computation greatly.
        for col in 0..noise_ref.len() {
            let p_col = self.iter().copied().skip(col).step_by(noise_ref.len());
            let xt_p_col = noise_ref.iter().zip(p_col).map(|(x, p)| x * p).sum::<F>();

            // takes result^ and computes lambda^-1 * [p_{n-1} - k(xt_p column)]
            for (row, k_i) in kalman_gain.iter().copied().enumerate() {
                let p_i = self
                    .get_2d_mut(row, col)
                    // TODO(ben): How is this enforced?
                    .expect("Index should always be valid based on the type invariants");
                *p_i = (*p_i - k_i * xt_p_col) / lambda;
            }
        }
    }
}
impl<F: Float> Deref for InverseCorrMatrix<F> {
    type Target = Matrix<F>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F: Float> DerefMut for InverseCorrMatrix<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{all_approx_equal, block_noise_buffer_from},
        types::{BlockSize, FilterWeights, WindowSize, buffers::BlockError},
    };

    #[test]
    fn init_p_matrix_works() {
        let n = WindowSize::new(3).unwrap();
        let p_matrix = InverseCorrMatrix::new(n, 10.0);

        let expected = Matrix::from_slice(
            &[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            Shape::new(3, 3).unwrap(),
        )
        .unwrap();

        assert_eq!(*p_matrix, expected);
    }

    #[test]
    fn update_kalman_gain_works() {
        let mut k = KalmanGain {
            inner: Matrix::new(Shape::new(2, 1).unwrap()),
        };
        let p = InverseCorrMatrix {
            inner: Matrix::from_slice(&[1.0, 0.0, 0.0, 1.0], Shape::new(2, 2).unwrap()).unwrap(),
        };
        let x_n = block_noise_buffer_from(
            &[1.0, 2.0],
            WindowSize::new(2).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        k.update(&p, &x_n.get_window(0).unwrap(), 1.0);

        let expected = [1.0 / 6.0, 1.0 / 3.0];

        assert!(all_approx_equal(k.inner().iter(), expected.iter()));
    }

    #[test]
    fn update_p_matrix_works() {
        let mut p = InverseCorrMatrix {
            inner: Matrix::from_slice(&[1.0, 0.0, 0.0, 1.0], Shape::new(2, 2).unwrap()).unwrap(),
        };
        let k = KalmanGain {
            inner: Matrix::from_slice(&[1.0 / 6.0, 1.0 / 3.0], Shape::new(2, 1).unwrap()).unwrap(),
        };

        let x_n = block_noise_buffer_from(
            &[1.0, 2.0],
            WindowSize::new(2).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        p.update(&k, &x_n.get_window(0).unwrap(), 1.0);

        let expected = InverseCorrMatrix {
            inner: Matrix::from_slice(
                &[5.0 / 6.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0],
                Shape::new(2, 2).unwrap(),
            )
            .unwrap(),
        };

        assert!(all_approx_equal(p.inner().iter(), expected.inner().iter()));
    }

    #[test]
    fn update_rls_1() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut rls = Rls::new(0.5, 1.0).unwrap();
        let e_n = {
            let mut e_n = BlockError::new(block_size);
            e_n.push(OutputSample(2.0));
            e_n
        };
        let x_n = block_noise_buffer_from(&[1.0, -1.0], window_size, block_size);
        let expected = [0.8, -0.8];
        let mut weights = FilterWeights::new(window_size);

        rls.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_rls_2() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut rls = Rls::new(1.0, 1.0).unwrap();
        let e_n = {
            let mut e_n = BlockError::new(block_size);
            e_n.push(OutputSample(1.0));
            e_n
        };
        let x_n = block_noise_buffer_from(&[5.0, 2.0], window_size, block_size);
        let expected = [5.0 / 30.0, 2.0 / 30.0];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        rls.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn forgetting_factor_range() {
        let p_init_scale = 1.0;
        Rls::new(0.01, p_init_scale).unwrap();
        Rls::new(1.0, p_init_scale).unwrap();

        assert!(matches!(
            Rls::new(0.0, p_init_scale),
            Err(Error::InvalidForgettingFactorRange)
        ));
        assert!(matches!(
            Rls::new(-1.0, p_init_scale),
            Err(Error::InvalidForgettingFactorRange)
        ));

        assert!(matches!(
            Rls::new(2.0, p_init_scale),
            Err(Error::InvalidForgettingFactorRange)
        ));
    }

    #[test]
    fn p_init_scale_range() {
        let good_init_scale = 100.0;
        Rls::new(0.01, good_init_scale).unwrap();

        assert!(matches!(
            Rls::new(0.5, 0.0),
            Err(Error::NonPositivePInitScale)
        ));
        assert!(matches!(
            Rls::new(0.5, -1.0),
            Err(Error::NonPositivePInitScale)
        ));
    }
}
