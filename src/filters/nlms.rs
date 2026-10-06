use std::ops::{Deref, DerefMut};

use super::FilterBase;

use crate::Result;
use crate::algorithms::Nlms;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone)]
pub struct NlmsOptions<F: Float> {
    pub mu: F,
    pub eps: F,
}

#[derive(Debug, Clone)]
pub struct NlmsFilter<F: Float> {
    inner: FilterBase<F, Nlms<F>>,
}
impl<F: Float> NlmsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if mu or eps <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: NlmsOptions<F>, window_size: WindowSize) -> Result<Self> {
        let lms = Nlms::new(options.mu, options.eps)?;
        let filter = FilterBase::new(lms, *window_size)?;
        Ok(Self { inner: filter })
    }

    // TODO: from_weights() ?
}
impl<F: Float> Deref for NlmsFilter<F> {
    type Target = FilterBase<F, Nlms<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for NlmsFilter<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;

    #[test]
    fn nlms_new() {
        let mu = 0.1;
        let eps = 1e-8;
        let window_size = 16;

        let filter = NlmsFilter::new(
            NlmsOptions { mu, eps },
            WindowSize::new(window_size).unwrap(),
        )
        .unwrap();
        let expected_inner = FilterBase::new(Nlms::new(mu, eps).unwrap(), window_size).unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }
}
