module OnlineInference

using AbstractMCMC
using AdvancedMH
using Bijectors
using Distributions
using GeneralisedFilters
using LinearAlgebra
using LogDensityProblems
using LogExpFunctions
using PDMats
using Printf
using Random
using SSMProblems
using StatsBase
using Suppressor

using GeneralisedFilters:
    ParticleDistribution,
    Particle,
    TypelessZero,
    AbstractConditionalResampler

export smc_iter,
    smc_initialise,
    StateSpaceLogDensity,
    run_smc,
    PMMH,
    SMC

## LOG DENSITY PROBLEM #####################################################################

struct StateSpaceLogDensity{BT,FT,PT,YT}
    build::BT
    algo::FT
    prior::PT
    data::YT
end

function LogDensityProblems.logdensity(p::StateSpaceLogDensity, θ)
    # TODO: optimize the bijectors stack
    θinv = invlink(p.prior, θ)
    _, logmarginal = GeneralisedFilters.filter(p.build(θinv), p.algo, p.data)
    return logpdf(p.prior, θinv) + logmarginal
end

LogDensityProblems.dimension(p::StateSpaceLogDensity) = length(p.prior)

function LogDensityProblems.capabilities(::StateSpaceLogDensity)
    return LogDensityProblems.LogDensityOrder{0}()
end

prior(p::StateSpaceLogDensity) = p.prior

## CUSTOM STATE PARTICLES ##################################################################

struct ModelState{MT,θT,XT}
    model::MT
    params::θT
    sample::XT
end

unpack(state::ModelState) = (state.model, state.params, state.sample)

# this only collects the mean of the parameters, not the states
function StatsBase.mean(
    sample::ParticleDistribution{<:Real,<:Particle{PT}}
) where {PT<:ModelState}
    parameters = hcat(map(x -> x.state.params, sample.particles)...)
    return mean(parameters, StatsBase.weights(sample), 2)
end

function reinitialise(parameters, p::StateSpaceLogDensity; kwargs...)
    particles = map(parameters) do θ
        # TODO: optimize the bijectors stack
        θinv = invlink(p.prior, θ)
        logprior = logpdf(prior(p), θinv)
        model = p.build(θinv)
        states, logweight = GeneralisedFilters.filter(model, p.algo, p.data; kwargs...)
        Particle(ModelState(model, θinv, states), logweight + logprior, 0)
    end
    return ParticleDistribution(particles, TypelessZero())
end

include("move_resampler.jl")
include("smc_base.jl")

end