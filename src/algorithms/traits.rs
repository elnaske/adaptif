use crate::types::buffers::{BlockError, BlockNoiseBuffer, NoiseBuffer};
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, Float};

/// Trait used for implementing algorithms with block-based processing used in conjuction with
/// `FilterBase`.
pub trait Algorithm<F: Float> {
    /// Updates the weights for the next block based on the algorithm's update rules.
    /// This function is called for every processing block by the filter during adapation.
    /// `error` are the cleaned samples from the current block.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    ///
    fn update_step(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_window: impl Iterator<Item = F>, // TODO: replace w/ actual type
    );

    fn update_block(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: &BlockError<F>,
        noise_ref: &BlockNoiseBuffer<F>,
    ) {
        for (window_start, window_error) in error.iter().copied().enumerate() {
            let current_window = noise_ref
                .iter()
                .copied()
                .skip(window_start)
                .take(*weights.window_size());

            self.update_step(weights, OutputSample(window_error), current_window);
        }
    }
}

/// Trait used for implementing algorithms with sample-based processing used in conjuction with
/// `SampleFilter`.
// TODO: remove one algorithms have been moved to block processing
pub trait SampleAlgorithm<F: Float> {
    /// Updates the weights for the next time step based on the algorithm's update rules.
    /// This function is called every processing iteration by the filter during adapation.
    /// `error` is the cleaned sample from the current time step.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    ///
    fn update_sample(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_ref: &NoiseBuffer<F>,
    );
}
