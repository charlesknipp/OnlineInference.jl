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
    batch_tempered_smc,
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
    θinv = maybe_invlink(p, θ)
    return logprior(p, θinv) + logmarginal(p, θinv)
end

# TODO: optimize the bijectors stack
maybe_invlink(p::StateSpaceLogDensity, θ) = invlink(p.prior, θ)

function logmarginal(p::StateSpaceLogDensity, θ)
    _, logmarginal = GeneralisedFilters.filter(p.build(θ), p.algo, p.data)
    return logmarginal
end

logprior(p::StateSpaceLogDensity, θ) = logpdf(p.prior, θ)

LogDensityProblems.dimension(p::StateSpaceLogDensity) = length(p.prior)

function LogDensityProblems.capabilities(::StateSpaceLogDensity)
    return LogDensityProblems.LogDensityOrder{0}()
end

prior(p::StateSpaceLogDensity) = p.prior

get_iter(p::StateSpaceLogDensity) = lastindex(p.data)

function reinitialise(parameters, p::StateSpaceLogDensity; kwargs...)
    particles = map(parameters) do θ
        θinv = invlink(p.prior, θ)
        logprior = logpdf(prior(p), θinv)
        model = p.build(θinv)
        states, logweight = GeneralisedFilters.filter(model, p.algo, p.data; kwargs...)
        Particle(ModelState(model, θinv, states), logweight + logprior, 0)
    end
    return ParticleDistribution(particles, TypelessZero())
end

## DENSITY TEMPERING #######################################################################

struct TemperedLogDensity{T,LDP}
    temperature::T
    logdensity::LDP
end

function LogDensityProblems.logdensity(p::TemperedLogDensity, θ)
    θinv = maybe_invlink(p.logdensity, θ)
    return p.temperature * logmarginal(p.logdensity, θinv) + logprior(p.logdensity, θinv)
end

function LogDensityProblems.dimension(p::TemperedLogDensity)
    return LogDensityProblems.dimension(p.logdensity)
end

function LogDensityProblems.capabilities(p::TemperedLogDensity)
    return LogDensityProblems.capabilities(p.logdensity)
end

prior(p::TemperedLogDensity) = prior(p.logdensity)

get_iter(p::TemperedLogDensity) = get_iter(p.logdensity)

# do not include tempering in the weights here, so we can reuse them later
function reinitialise(parameters, p::TemperedLogDensity; kwargs...)
    reinitialise(parameters, p.logdensity; kwargs...)
end

## CUSTOM STATE PARTICLES ##################################################################

struct ModelState{MT,θT,XT}
    model::MT
    params::θT
    sample::XT
end

unpack(state::ModelState) = (state.model, state.params, state.sample)

# this only collects the mean of the parameters, not the states
function StatsBase.mean(
    sample::ParticleDistribution{<:Number,<:Particle{PT}}
) where {PT<:ModelState}
    parameters = hcat(map(x -> x.state.params, sample.particles)...)
    return mean(parameters, StatsBase.weights(sample), 2)
end

include("move_resampler.jl")
include("smc_base.jl")

end