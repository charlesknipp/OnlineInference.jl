using AbstractMCMC
using CSV
using DataFrames
using Distributions
using GeneralisedFilters
using LinearAlgebra
using OnlineInference
using PDMats
using Random
using SSMProblems
using StaticArrays

#= NOTE: most of these dependencies are in favor of speed; for example, the StaticMvNormal
is something that belongs in an extension. I can also reexport things like MCMCThreads() if
I also plan on doing multithreaded propagation.
=#

const GF = GeneralisedFilters

## STATIC ARRAY UTILITIES ##################################################################

# while not necessary, it speeds up sampling static arrays in nonlinear states
const StaticMvNormal{N,T} = MvNormal{T,PDMat{T,MT},VT} where {
    N,T,MT<:StaticMatrix{N,N,T},VT<:StaticVector{N,T}
}

function PDMats.unwhiten(
    a::PDMat{T,AT}, x::SVector{N,T}
) where {T<:Real,N,AT<:StaticMatrix{N,N,T}}
    return PDMats.chol_lower(cholesky(a)) * x
end

# this should singlehandedly fix sampling from Static MvNormal
function Random.rand(rng::AbstractRNG, d::StaticMvNormal{N,T}) where {N,T<:Real}
    return d.μ + PDMats.unwhiten(d.Σ, (@SVector randn(rng, N)))
end

## LATENT DYNAMICS #########################################################################

struct LocalLevelTrend <: LinearGaussianLatentDynamics end

GF.calc_A(::LocalLevelTrend, ::Integer; kwargs...) = ones(SMatrix{1,1})
GF.calc_b(::LocalLevelTrend, ::Integer; kwargs...) = zeros(SVector{1})
function GF.calc_Q(::LocalLevelTrend, ::Integer; new_outer, kwargs...)
    return PDMat(SMatrix{1,1}(exp(new_outer[1])))
end

## OBSERVATION PROCESS #####################################################################

struct SimpleObservation <: LinearGaussianObservationProcess end

GF.calc_H(::SimpleObservation, ::Integer; kwargs...) = ones(SMatrix{1,1})
GF.calc_c(::SimpleObservation, ::Integer; kwargs...) = zeros(SVector{1})
function GF.calc_R(::SimpleObservation, ::Integer; new_outer, kwargs...)
    return PDMat(SMatrix{1,1}(exp(new_outer[2])))
end

function UCSV(γ::T) where {T<:Real}
    stoch_vol_prior = GF.HomogeneousGaussianPrior(
        zeros(SVector{2,T}), PDMat(10 * SMatrix{2,2,T}(I))
    )

    stoch_vol_process = GF.HomogeneousLinearGaussianLatentDynamics(
        SMatrix{2,2,T}(I), zeros(SVector{2,T}), PDMat(exp(γ) * SMatrix{2,2,T}(I))
    )

    local_level_model = StateSpaceModel(
        GF.HomogeneousGaussianPrior(zeros(SVector{1,T}), PDMat(SMatrix{1,1}(100.0))),
        LocalLevelTrend(),
        SimpleObservation(),
    )

    return HierarchicalSSM(stoch_vol_prior, stoch_vol_process, local_level_model)
end

## MAIN ####################################################################################

prior = MvNormal(1.0I(1))
fred_data = CSV.read("examples/trend-inflation/data.csv", DataFrame)
infl_data = [[val] for val in fred_data.value]

# run SMC² with a Rao-Blackwellised particle filter and multithreaded PMMH rejuvenation
rng = MersenneTwister(1234)
smc = SMC(500, ESSResampler(0.6), PMMH(10), RBPF(BF(2^12), KF()))
sample = run_smc(rng, x -> UCSV(only(x)), prior, smc, infl_data; ensemble=MCMCThreads());
vcat(map(x -> exp.(x.state.params), sample.particles)...)
