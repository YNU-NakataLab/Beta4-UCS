include("condition.jl")

using Random
using Distributions

"""
    B4Classifier

Represents a β4 classifier in the β4-UCS system, storing rule conditions as four-parameter beta distributions (FBR).
"""
mutable struct B4Classifier
    id::Int64                               # Unique identifier for the classifier
    condition::Vector{FBR}                  # Condition represented as a vector of FBRs
    weight_array::Vector{Float64}           # Weights for each action
    fitness::Float64                        # Fitness of the classifier
    correct_matching_array::Vector{Float64} # Correct matches for each action
    experience::Float64                     # Number of times the classifier has been involved in matching
    time_stamp::Int64                       # Last time the classifier was involved in the GA
    correct_set_size::Float64               # Average size of correct sets this classifier has been part of
    correct_set_size_array::Vector{Float64} # History of correct set sizes
    numerosity::Int64                       # Number of micro-classifiers this macro-classifier represents
    matching_degree::Float64                # Degree of match with the current state
end

"""
    B4Classifier(parameters, env, state)

Initialize a new β4 classifier using covering operator (Algorithm 1 line 8-9).
- Creates crisp intervals with a probability P_hash of "Don't Care"
- Implements parameterized covering range r0
"""
function B4Classifier(parameters, env, state)
    condition::Vector{FBR} = Vector{FBR}(undef, env.state_length)
    @simd for i in 1:env.state_length
        if rand() < parameters.P_hash || state[i] == "?"
            # Create "Don't Care" condition (l=0, u=1, α=β=1)
            condition[i] = FBR(1.0, 1.0, 0.0, 1.0)
        else
            # Initialize crisp interval per covering operator
            a::Float64 = 1.0
            b::Float64 = a # Maintain crisp initialization
            d::Float64 = rand(Uniform(0.005, parameters.r0)) # Covering range parameter
            l::Float64 = state[i] - d
            u::Float64 = state[i] + d
            condition[i] = FBR(a, b, l, u)
        end
    end
    clas::B4Classifier = B4Classifier(0, condition, zeros(Float64, env.num_actions), 1, zeros(Float64, env.num_actions), 0, 0, 1, [], 1, 1)
    return clas
end

"""
    apply_crossover!(child_1, child_2)

Perform uniform crossover on FBR parameters.
- Swaps α, β, l, u parameters per dimension with 50% probability
- Maintains ordered conditions post-crossover
"""
function apply_crossover!(child_1::B4Classifier, child_2::B4Classifier)
    @simd for i in 1:length(child_1.condition)
        # Swap shape parameters
        if rand() < 0.5
            child_1.condition[i].a, child_2.condition[i].a = child_2.condition[i].a, child_1.condition[i].a
        end
        if rand() < 0.5
            child_1.condition[i].b, child_2.condition[i].b = child_2.condition[i].b, child_1.condition[i].b
        end

        # Swap interval parameters
        if rand() < 0.5
            child_1.condition[i].l, child_2.condition[i].l = child_2.condition[i].l, child_1.condition[i].l
        end
        if rand() < 0.5
            child_1.condition[i].u, child_2.condition[i].u = child_2.condition[i].u, child_1.condition[i].u
        end

        # Ensure valid intervals after crossover
        @simd for child in (child_1, child_2)
            ordered_condition!(child.condition[i])
        end
    end
end

"""
    apply_mutation!(clas, m0, mu)

Mutate FBR parameters with probability mu.
- Relative mutation for α/β (±50% of current value)
- Absolute mutation for l/u (±m0 range)
- Enforces α,β ≥ 1.0 and 0 ≤ l < u ≤ 1 constraints
"""
function apply_mutation!(clas::B4Classifier, m0::Float64, mu::Float64)
    @simd for i in 1:length(clas.condition)
        if rand() < mu
            # Shape parameter mutation (relative)
            clas.condition[i].a += rand(Uniform(-clas.condition[i].a / 2.0, clas.condition[i].a / 2.0))
            clas.condition[i].b += rand(Uniform(-clas.condition[i].b / 2.0, clas.condition[i].b / 2.0))
            
            # Interval parameter mutation (absolute)
            clas.condition[i].l += 2. * m0 * rand() - m0
            clas.condition[i].u += 2. * m0 * rand() - m0

            # Clamp values to valid ranges
            clas.condition[i].a = max(1.0, clas.condition[i].a)
            clas.condition[i].b = max(1.0, clas.condition[i].b)
            clas.condition[i].l = min(0.995, clas.condition[i].l) # Prevent l >= u
            clas.condition[i].u = max(0.005, clas.condition[i].u)

            ordered_condition!(clas.condition[i])
        end
    end
end

"""
    is_more_general(general, specific, Tol_sub)::Bool

Check if `general` subsumes `specific` using three conditions:
1. Interval inclusion
2. Kurtosis comparison
3. Mode similarity with Tol_sub tolerance
"""
function is_more_general(general::B4Classifier, specific::B4Classifier, Tol_sub::Float64)::Bool
    k::Int64 = 0
    for i in eachindex(general.condition)
        gen_fbr = general.condition[i]
        spec_fbr = specific.condition[i]

        # Condition 1: Interval inclusion
        l_gen = max(0.0, gen_fbr.l)
        u_gen = min(1.0, gen_fbr.u)
        l_spec = max(0.0, spec_fbr.l)
        u_spec = min(1.0, spec_fbr.u)
        
        if !(l_gen ≤ l_spec ≤ u_spec ≤ u_gen)
            return false
        end

        # Condition 2: Kurtosis comparison
        if get_kurtosis(gen_fbr) > get_kurtosis(spec_fbr)
            return false
        end

        # Condition 3: Mode similarity for non-crisp rules
        if is_bell(gen_fbr) && is_bell(spec_fbr)
            mode_diff = abs(get_mode(gen_fbr) - get_mode(spec_fbr))
            if mode_diff > Tol_sub
                return false
            end
        end

        if is_equal(general.condition[i], specific.condition[i])
            k += 1
        end

    end
    if k == length(general.condition)
        return false
    end
    return true
end

"""
    is_equal_classifier(a, b)::Bool

Identity check using unique classifier IDs (not parameter values).
"""
is_equal_classifier(a::B4Classifier, b::B4Classifier)::Bool = a.id == b.id

"Check if two fuzzy classifiers have equal conditions"
function is_equal_condition(a::B4Classifier, b::B4Classifier)::Bool
    for i in eachindex(a.condition)
            if !(is_equal(a.condition[i], b.condition[i]))
                return false
            end
    end
    return true
end
