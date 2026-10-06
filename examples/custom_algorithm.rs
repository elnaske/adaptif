use std::error::Error;

use adaptif::algorithms::SampleAlgorithm;
use adaptif::filters::{AdaptiveFilter as _, SampleFilterBase};
use adaptif::types::buffers::NoiseBuffer;
use adaptif::types::signals::{InputSignal, NoiseReference, OutputSample};
use adaptif::types::{FilterWeights, Float};

// Create a struct to hold any required parameters or state
pub struct MyAlgorithm<F: Float> {
    pub alpha: F,
}
// Implement the Algorithm trait so the algorithm can be used with SampleFilterBase
impl<F: Float> SampleAlgorithm<F> for MyAlgorithm<F> {
    // This function is called every iteration during adaptation to update the weights
    fn update_step(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_ref: &NoiseBuffer<F>,
    ) {
        for (w, x) in weights.iter_mut().zip(noise_ref.iter().copied()) {
            *w += self.alpha * (*error) * x;
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Sample inputs
    let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
    let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;

    // Define the algorithm parameters
    let algorithm_cfg = MyAlgorithm { alpha: 1.0 };
    let window_size = 1024;

    // Instantiate the filter using SampleFilterBase and our custom algorithm
    let mut filter = SampleFilterBase::new(algorithm_cfg, window_size)?;

    // Adapt the filter using our algorithm's update rules
    let _output = filter.adapt(&input_signal, &noise_ref)?;

    Ok(())
}
