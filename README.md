# OnlineInference.jl

This Julia module implements a pseudo online sampler called [SMC²](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2012.01046.x) which jointly estimates hidden states as well as model parameters with consistent information sets.

Defining an SMC algorithm is done like so:

```julia
smc_algo = SMC(
    parameter_particles::Int,
    resampling_algorithm::ESSResampler,
    rejuvenation_kernel::PMCMCKernel,
    filter::AbstractFilter
)
```

where the filter must be compatible with the generated state space model. For example, one cannot run a nonlinear model with a Kalman filter. 

When running the algorithm online, you can pass additional key word arguments like `ensemble` which determines whether to parallelize the rejuventation/resampling step via `MCMCThreads()` or `MCMCDistributed()`. Feel free to play around with various configurations to ensure parallel processes provide ample speedup.

For rejuvenation, the only working algorithm is metropolis hastings, however I seek to get Hamiltonian Monte Carlo operational for future iterations.

## Examples

For both examples, I rely heavily on `StaticArrays.jl` to both reduce the memory footprint as well as guarantee efficient computation for Kalman filtered likelihoods.


> [!TIP]
> While I have `CairoMakie.jl` included in the toml, this is purely for testing purposes, and is not a necessary component of compiling the replication package. Feel free to remove it and reinitialize for faster precompilation.

#### Stochastic Volatility

This tried and true example estimates and decomposes volatility into permanent and transitory components according to a random walk.

For this model, the number of state particles is determined according to a conditionally analytical filter (or a Rao-Blackwellized particle filter) where we run 4096 state particles per 100 parameter particles.

Details on this model can be found in [this paper](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1538-4616.2007.00014.x) with a log Normal prior chosen for variance of log volatility. They use a vastly different approach to solve this model, but I would argue (and I am not alone) that a particle filter is far superior.

#### Trend Cycle Decomposition

This more experimental setup estimates a frequency band in addition to the model variances, which decomposes the time series into cycle + trend + noise. Instead of a nonlinear approach, this relies on linear time invariant signal processing and thus can be solved via the Kalman filter.

Since we estimate more parameters, however, this is quite a bit more intense; requiring at least 10x more parameter particles.

For this particular problem, the algorithm tends to get stuch around `t = 244` because the model picks up COVID as a signal as opposed to noise. There are ways around this, but I have not yet controlled for that in this iteration of the model.

Details on the underlying model can be found in [this paper](https://personal.eur.nl/hkvandijk/PDF/Harvey_Trimbur_and_Van_Dijk_2007_JoE_trends_and_cycles.pdf) with a standard prior on the passband. Their use of a Gibbs sampler is clever but incorrect in that parameter estimation is not information consistent; thus SMC² is a far more appropriate means for evaluating forecasts.
