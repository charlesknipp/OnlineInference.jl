## MODULAR SMC METHODS #####################################################################

struct SMC{FT<:GeneralisedFilters.AbstractFilter,RS,KT}
    N::Int
    resampler::RS
    kernel::KT
    filter::FT
end

GeneralisedFilters.num_particles(algo::SMC) = algo.N

function resampler(algo::SMC, problem::StateSpaceLogDensity, iter::Int)
    acc_problem = StateSpaceLogDensity(
        problem.build, problem.algo, problem.prior, problem.data[1:iter]
    )
    return MoveResampler(algo.resampler, algo.kernel, acc_problem)
end

# GF.initialise(rng, prior, algo; kwargs...)
function smc_initialise(rng::AbstractRNG, logdensity::StateSpaceLogDensity, algo; kwargs...)
    particles = map(1:GeneralisedFilters.num_particles(algo)) do _
        parameter = rand(rng, prior(logdensity))
        model = logdensity.build(parameter)
        states = GeneralisedFilters.initialise(rng, SSMProblems.prior(model), algo.filter; kwargs...)
        Particle(ModelState(model, parameter, states), 0)
    end
    return ParticleDistribution(particles, TypelessZero())
end

function smc_propagate(
    rng::AbstractRNG, algo::SMC, iter::Integer, particle, observation; kwargs...
)
    # run the filtering step
    model, params, sample = unpack(particle.state)
    new_sample, log_increment = GeneralisedFilters.step(
        rng, model, algo.filter, iter, sample, observation; kwargs...
    )

    # populate the particle
    return Particle(
        ModelState(model, params, new_sample),
        GeneralisedFilters.log_weight(particle) + log_increment,
        particle.ancestor
    )
end

# GF.step(rng, model, algo, iter, state, observation; kwargs...)
function smc_iter(
    rng::AbstractRNG,
    model::StateSpaceLogDensity,
    algo::SMC,
    iter::Integer,
    state,
    observation;
    kwargs...
)
    rs = resampler(algo, model, iter)
    state = GeneralisedFilters.maybe_resample(rng, rs, state; kwargs...)

    # propagate the particles through a given filter
    particles = map(eachindex(state.particles)) do i
        particle = state.particles[i]
        smc_propagate(rng, algo, iter, particle, observation; kwargs...)
    end

    # populate the entire distribution
    return ParticleDistribution(
        particles, logsumexp(GeneralisedFilters.log_weights(state)) + state.ll_baseline
    )
end

function run_smc(rng::AbstractRNG, build, prior, algo::SMC, observations; kwargs...)
    ssm_logdensity = StateSpaceLogDensity(build, algo.filter, prior, observations)
    init_state = smc_initialise(rng, ssm_logdensity, algo)

    state = smc_iter(rng, ssm_logdensity, algo, 1, init_state, observations[1]; kwargs...)
    for t in 2:length(observations)
        state = smc_iter(rng, ssm_logdensity, algo, t, state, observations[t]; kwargs...)
    end
    return state
end
