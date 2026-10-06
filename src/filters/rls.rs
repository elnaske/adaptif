use std::ops::{Deref, DerefMut};

use super::SampleFilterBase;

use crate::Result;
use crate::algorithms::Rls;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone)]
pub struct RlsOptions<F: Float> {
    pub forgetting_factor: F,
    pub p_init_scale: F,
}

#[derive(Debug, Clone)]
pub struct RlsFilter<F: Float> {
    inner: SampleFilterBase<F, Rls<F>>,
}
impl<F: Float> RlsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if forgetting factor <= 0.0 or > 1.0, if init scale <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: RlsOptions<F>, window_size: WindowSize) -> Result<Self> {
        let lms = Rls::new(options.forgetting_factor, options.p_init_scale)?;
        let filter = SampleFilterBase::new(lms, *window_size)?;
        Ok(Self { inner: filter })
    }

    // TODO: from_weights() ?
}
impl<F: Float> Deref for RlsFilter<F> {
    type Target = SampleFilterBase<F, Rls<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for RlsFilter<F> {
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
        let forgetting_factor = 0.999;
        let p_init_scale = 1e-4;
        let window_size = 16;

        let filter = RlsFilter::new(
            RlsOptions {
                forgetting_factor,
                p_init_scale,
            },
            WindowSize::new(window_size).unwrap(),
        )
        .unwrap();
        let expected_inner = SampleFilterBase::new(
            Rls::new(forgetting_factor, p_init_scale).unwrap(),
            window_size,
        )
        .unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }
}
