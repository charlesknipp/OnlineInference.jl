## PARTICLE REJUVENATION KERNELS ###########################################################

abstract type AbstractMove end

struct IdentityKernel <: AbstractMove end

function rejuvenate(
    rng::AbstractRNG,
    logdensity,
    kernel::IdentityKernel,
    state::ParticleDistribution;
    kwargs...
)
    return state
end

struct PMCMCKernel{KT<:AbstractMCMC.AbstractSampler} <: AbstractMove
    sampler::KT
    num_chains::Int
end

function rejuvenate(
    rng::AbstractRNG,
    logdensity,
    kernel::PMCMCKernel,
    state::ParticleDistribution;
    ensemble=MCMCSerial(),
    kwargs...
)
    sampler = adapt(kernel.sampler, state)
    N = kernel.num_chains
    initial_params = map(x -> link(prior(logdensity), x.state.params), state.particles)
    chains = @suppress_err AbstractMCMC.sample(
        rng, logdensity, sampler, ensemble, N, length(state); initial_params, kwargs...
    )
    return repackage(chains, logdensity; kwargs...)
end

## PMMH KERNEL #############################################################################

# should this be a subtype of AbstractSampler?
struct RandomWalkKernel <: AbstractMCMC.AbstractSampler end

PMMH(num_chains::Int) = PMCMCKernel(RandomWalkKernel(), num_chains)

function adapt(::RandomWalkKernel, state)
    parameters = hcat(map(x -> x.state.params, state.particles)...)
    scale = (2.38 ^ 2) / size(parameters, 1)
    Σ = cov(parameters, StatsBase.weights(state), 2)
    scaled_Σ = if isposdef(Σ)
        scale * Σ
    else
        (convert(eltype(Σ), 1e-8) * I(size(Σ, 1)))
    end
    return RWMH(MvNormal(zeros(eltype(Σ), size(Σ, 1)), scaled_Σ))
end

# I think this could generalize a bit better
function repackage(
    chains::AbstractVector{<:AbstractVector{T}}, ssmprob; kwargs...
) where {T<:AdvancedMH.Transition}
    return reinitialise(map(x -> x[end].params, chains), ssmprob; kwargs...)
end

## REJUVENATION RESAMPLER ##################################################################

# not sure if this is properly subtyped...
mutable struct MoveResampler{RS,MF<:AbstractMove} <: AbstractConditionalResampler
    const resampler::RS
    const move::MF
    problem
end

# TODO: really shitty typing going on, but I can fix that later
function MoveResampler(threshold::Float64, move::AbstractMove)
    return MoveResampler(ESSResampler(threshold), move, nothing)
end

ess(weights) = inv(sum(abs2, weights))

# we must overload maybe_resample to avoid problems with kwargs
function GeneralisedFilters.maybe_resample(
    rng::AbstractRNG,
    algo::MoveResampler,
    state,
    weights=GeneralisedFilters.get_weights(state);
    kwargs...
)
    ess_tracker = @sprintf("t = %4d\tess = %7.2f", get_iter(algo.problem), ess(weights))
    print("\r" * ess_tracker)
    if GeneralisedFilters.will_resample(algo.resampler, state, weights)
        println("\t(resampling)")
        prop_state = GeneralisedFilters.resample(rng, algo.resampler, state, weights)
        return rejuvenate(rng, algo.problem, algo.move, prop_state; kwargs...)
    else
        return GeneralisedFilters.preserve_sample(state)
    end
end
