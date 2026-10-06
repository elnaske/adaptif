use std::ops::{Deref, DerefMut};

use super::{BlockFilterBase, FilterBase};

use crate::Result;
use crate::algorithms::Lms;
use crate::types::{BlockSize, Float, WindowSize};

#[derive(Debug, Clone)]
pub struct LmsOptions<F: Float> {
    pub mu: F,
}

#[derive(Debug, Clone)]
pub struct LmsFilter<F: Float> {
    inner: FilterBase<F, Lms<F>>,
}
impl<F: Float> LmsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: LmsOptions<F>, window_size: WindowSize) -> Result<Self> {
        let lms = Lms::new(options.mu)?;
        let filter = FilterBase::new(lms, *window_size)?;
        Ok(Self { inner: filter })
    }

    // TODO: from_weights() ?
}
impl<F: Float> Deref for LmsFilter<F> {
    type Target = FilterBase<F, Lms<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for LmsFilter<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone)]
pub struct BlockLmsFilter<F: Float> {
    inner: BlockFilterBase<F, Lms<F>>,
}
impl<F: Float> BlockLmsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(
        options: LmsOptions<F>,
        window_size: WindowSize,
        block_size: BlockSize,
    ) -> Result<Self> {
        let lms = Lms::new(options.mu)?;
        let filter = BlockFilterBase::new(lms, *window_size, *block_size)?;
        Ok(Self { inner: filter })
    }

    // TODO: from_weights() ?
}
impl<F: Float> Deref for BlockLmsFilter<F> {
    type Target = BlockFilterBase<F, Lms<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for BlockLmsFilter<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;

    #[test]
    fn lms_new() {
        let mu = 0.1;
        let window_size = 16;

        let filter =
            LmsFilter::new(LmsOptions { mu }, WindowSize::new(window_size).unwrap()).unwrap();
        let expected_inner = FilterBase::new(Lms::new(mu).unwrap(), window_size).unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }

    #[test]
    fn block_lms_new() {
        let mu = 0.1;
        let window_size = 16;
        let block_size = 8;

        let filter = BlockLmsFilter::new(
            LmsOptions { mu },
            WindowSize::new(window_size).unwrap(),
            BlockSize::new(block_size).unwrap(),
        )
        .unwrap();
        let expected_inner =
            BlockFilterBase::new(Lms::new(mu).unwrap(), window_size, block_size).unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
        assert_eq!(filter.block_size(), block_size);
    }
}
