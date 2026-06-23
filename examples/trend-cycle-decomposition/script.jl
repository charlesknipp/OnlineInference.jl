using AbstractMCMC
using Bijectors
using CairoMakie
using CSV
using DataFrames
using Distributions
using GeneralisedFilters
using LinearAlgebra
using MatrixEquations
using OnlineInference
using PDMats
using Random
using SSMProblems
using StaticArrays

const GF = GeneralisedFilters

#= NOTE: my abuse of StaticArrays is quite gross. Presumably this will no longer be an issue
once the closure interface gets merged into SSMProblems/GeneralisedFilters.

This example is also relatively unstable. It requires a much lower ESS threshold for SMC². I
could get around this by implementing a tempered SMC algorithm for the first 50 observations
to "jump start" estimation like I do in our paper.
=#

## STATIC ARRAY UTILITIES ##################################################################

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
    return d.μ + PDMats.unwhiten(d.Σ, SVector{N,T}(randn(rng, N)))
end

## BANDPASS TREND PROCESS ##################################################################

function stochastic_cyle(n::Integer, ρ::Real, λ::Real)
    cosλ = cos(λ)
    sinλ = sin(λ)

    ϕ = kron(I(n), ρ * [cosλ sinλ; -sinλ cosλ])
    ϕ += kron(diagm(1 => ones(n - 1)), I(2))

    return SMatrix{2 * n,2 * n}(ϕ)
end

function stochastic_trend(m::Integer)
    return UpperTriangular(ones(SMatrix{m,m}))
end

struct TrendCycleDynamics{ΦT,ZT,ΚT} <: LinearGaussianLatentDynamics
    m::Int
    n::Int
    ϕ::ΦT
    σζ²::ZT
    σκ²::ΚT
end

GF.calc_A(cycle::TrendCycleDynamics, ::Integer; kwargs...) = cycle.ϕ

function GF.calc_b(cycle::TrendCycleDynamics, ::Integer; kwargs...)
    return @SVector zeros(cycle.m + 2 * cycle.n)
end

function GF.calc_Q(cycle::TrendCycleDynamics{ΦT}, ::Integer; kwargs...) where {ΦT}
    Σ = cat(
        Diagonal([zeros(cycle.m - 1); cycle.σζ²]),
        kron(Diagonal([zeros(cycle.n - 1); 1]), cycle.σκ² * I(2));
        dims=(1, 2)
    )
    N = cycle.m + 2 * cycle.n
    return SMatrix{N,N}(Σ)
end

## LINEAR OBSERVATION ######################################################################

struct TrendObservation{ET} <: LinearGaussianObservationProcess
    m::Int
    n::Int
    σε²::ET
end

function GF.calc_H(obs::TrendObservation, ::Integer; kwargs...)
    m, n = obs.m, obs.n
    return SMatrix{1,2 * n + m}([(i == 1) | (i == m + 1) for _ = 1:1, i = 1:(2*n+m)])
end

GF.calc_c(::TrendObservation, ::Integer; kwargs...) = zeros(SVector{1})

function GF.calc_R(obs::TrendObservation, ::Integer; kwargs...)
    return PDMat(SMatrix{1,1}(obs.σε²))
end

## STATE SPACE MODEL #######################################################################

function trend_cycle_model(
    n::Int64, m::Int64, ρ::Real, λ::Real, σκ²::Real, σζ²::Real, σε²::Real, init_state::Real
)
    # transition matrices
    ϕ1 = stochastic_trend(m)
    ϕ2 = stochastic_cyle(n, ρ, λ)

    # state dimension
    N = m + 2 * n

    # solve for optimal initial covariance via a lyapunov equation
    Σϕ = kron(Diagonal([zeros(n - 1); σζ²]), σκ² * I(2))
    Σ0 = SMatrix{N,N}(cat(1000I(m), lyapd(ϕ2, Σϕ), dims=(1, 2)))

    return StateSpaceModel(
        GF.HomogeneousGaussianPrior(
            SVector{N}([init_state; zeros(m - 1 + 2 * n)]), PDMat(Σ0)
        ),
        TrendCycleDynamics(m, n, SMatrix{N,N}(cat(ϕ1, ϕ2, dims=(1, 2))), σζ², σκ²),
        TrendObservation(m, n, σε²)
    )
end

function beta_prior(mode, var)
    temp = ((mode * (1 - mode)) / var) - 1
    return Beta(mode * temp, (1 - mode) * temp)
end

model_builder(θ) = trend_cycle_model(2, 2, θ[1], θ[2], θ[3], θ[4], θ[5], 816.542)

## MAIN ####################################################################################

prior = product_distribution([
    Uniform(0.01, 0.99),
    beta_prior(2π / 20, 0.001),
    LogNormal(),
    LogNormal(),
    LogNormal()
])

fred_data = CSV.read("examples/trend-cycle-decomposition/data.csv", DataFrame)
gdp_data = [[val] for val in fred_data.gdp]

# run SMC² with a Kalman filter and multithreaded PMMH rejuvenation
rng = MersenneTwister(1234)
smc = SMC(1000, ESSResampler(0.3), PMMH(20), KF())
sample = run_smc(rng, model_builder, prior, smc, gdp_data; ensemble=MCMCThreads());

# return the weighted mean of the sample
mean(sample)
