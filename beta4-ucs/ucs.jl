include("classifier.jl")

"""
    B4UCS

Main controller for β4-UCS algorithm.
"""
mutable struct B4UCS
    env::Environment                 # The environment the system interacts with
    parameters::Parameters           # System parameters
    population::Vector{B4Classifier} # Population of fuzzy classifiers
    time_stamp::Int64                # Current time step
    covering_occur_num::Int64        # Number of times covering occurred
    subsumption_occur_num::Int64     # Number of times subsumption occurred
    global_id::Int64                 # Global ID counter for classifiers
end

"""
    B4UCS(env, parameters)

Initialize β4-UCS with empty population and default counters.
"""
function B4UCS(env, parameters)
    return B4UCS(env, parameters, [], 0, 0, 0, 0)
end

"Main experiment loop for β4-UCS"
function run_experiment(b4ucs::B4UCS)
    # Get current state and correct answer from environment
    curr_state::Vector{Union{Float64, Int64, String}} = state(b4ucs.env)
    curr_answer::Int64 = answer(b4ucs.env, curr_state)

    # Generate match set based on current state
    match_set = generate_match_set(b4ucs, curr_state, curr_answer)

    # Generate correct set from match set
    correct_set = generate_correct_set(match_set, curr_answer)

    # Update parameters of classifiers in match and correct sets
    update_set!(b4ucs, match_set, correct_set, curr_answer)

    # Apply crispification to promote simplicity
    if b4ucs.parameters.do_correct_set_crispification
        do_correct_set_crispification!(b4ucs, correct_set)
    end

    # Run GA on correct set
    run_ga!(b4ucs, correct_set)

    # Increment time step
    b4ucs.time_stamp += 1
end

"""
    generate_match_set(b4ucs::B4UCS, state, answer, do_exploit)::Vector{B4Classifier}

Generate match set [M] for current state.
Implements covering operator when no existing rules adequately match the state.
"""
function generate_match_set(b4ucs::B4UCS, state::Vector{Union{Float64, Int64, String}}, answer::Int64, do_exploit=false)::Vector{B4Classifier}
    # Calculate matching degrees for all classifiers
    set_matching_degree(b4ucs, state)
    
    # Filter classifiers with non-zero matching degree
    match_set::Vector{B4Classifier} = filter(clas -> clas.matching_degree > 0.0, b4ucs.population)
    
    if !do_exploit
        # Check if covering is needed
        if do_covering(b4ucs, match_set, answer)
            # Covering not needed, do nothing
        else
            # Generate a covering classifier
            clas::B4Classifier = generate_covering_classifier(b4ucs, state, answer)
            # Add the new classifier to the population
            push!(b4ucs.population, clas)
            # Remove excess classifiers if necessary
            delete_from_population!(b4ucs)
            # Add the new classifier to the match set
            push!(match_set, clas)
        end
    end
    return match_set
end

"""
    set_matching_degree(b4ucs::B4UCS, state)

Calculate matching degree μ for all classifiers using β4-MFs.
"""
function set_matching_degree(b4ucs::B4UCS, state::Vector{Union{Float64, Int64, String}})
    @simd for clas in b4ucs.population
        clas.matching_degree = get_matching_degree(clas, state)
    end
end

"""
    get_matching_degree(clas, state)::Float64

Calculate classifier's matching degree as product of β4-MF values.
Implements early termination for zero-product cases.
"""
function get_matching_degree(clas::B4Classifier, state::Vector{Union{Float64, Int64, String}})::Float64
    matching_degree::Float64 = 1.0
    @simd for i in 1:length(state)
        # Multiply membership values for each attribute
        matching_degree *= get_membership_value(clas.condition[i], state[i])
        # Early exit if matching degree becomes zero
        if matching_degree == 0.0
            return 0.0
        end
    end
    # Sanity check: matching degree should be between 0 and 1
    if !(0.0 <= matching_degree <= 1.0)
        error("Matching degree should be in [0,1] : mu_A=$(matching_degree)")
    end
    return matching_degree
end

"""
    do_covering(b4ucs::B4UCS, match_set, answer)::Bool

Determine if covering is needed based on accumulated prediction weight.
Returns true if existing match set sufficiently covers correct class (weight ≥ 1).
"""
function do_covering(b4ucs::B4UCS, match_set::Vector{B4Classifier}, answer::Int64)::Bool
    weight_array::Vector{Float64} = zeros(Float64, b4ucs.env.num_actions)
    @simd for clas in match_set
        # Accumulate matching degrees for each action
        weight_array[argmax(clas.weight_array)] += clas.matching_degree
        # If the correct action has sufficient weight, no covering is needed
        if weight_array[answer + 1] >= 1
            return true
        end
    end
    # Covering is needed
    return false
end

"""
    generate_covering_classifier(b4ucs::B4UCS, state, answer)::B4Classifier

Create new classifier using covering operator.
"""
function generate_covering_classifier(b4ucs::B4UCS, state::Vector{Union{Float64, Int64, String}}, answer::Int64)::B4Classifier
    # Create a new fuzzy classifier
    clas::B4Classifier = B4Classifier(b4ucs.parameters, b4ucs.env, state)
    # Set the weight for the correct action to 1
    clas.weight_array[answer + 1] = 1
    # Set the timestamp and ID
    clas.time_stamp = b4ucs.time_stamp
    clas.id = b4ucs.global_id
    b4ucs.global_id += 1
    # Increment the covering counter
    b4ucs.covering_occur_num += 1
    return clas
end

"""
    update_set!(b4ucs::B4UCS, match_set, correct_set, answer)

Update classifier parameters using accuracy-based reinforcement.
Performs fitness calculation and maintains correct set statistics.
"""
function update_set!(b4ucs::B4UCS, match_set::Vector{B4Classifier}, correct_set::Vector{B4Classifier}, answer::Int64)
    # Calculate the total numerosity of the correct set
    set_numerosity::Int64 = mapreduce(clas -> clas.numerosity, +, correct_set)

    @simd for clas in match_set
        # Update experience
        clas.experience = clas.experience + clas.matching_degree

        # Update correct matching array and weights
        @simd for i in 1:b4ucs.env.num_actions
            if i == answer + 1
                clas.correct_matching_array[i] += clas.matching_degree
            end
            clas.weight_array[i] = clas.correct_matching_array[i] / clas.experience
        end
        
        # Sanity check: weights should sum to 1
        if round(sum(clas.weight_array), digits=3) != 1.0
            println("The sum of all the weights is not 1.")
            println(clas.weight_array, [clas.experience], [clas.matching_degree], clas.correct_matching_array)
            exit(1)
        end

        # Update fitness
        clas.fitness = 2 * maximum(clas.weight_array) - sum(clas.weight_array)
    end

    # Update correct set size for classifiers in the correct set
    @simd for clas in correct_set
        push!(clas.correct_set_size_array, set_numerosity)
        clas.correct_set_size = sum(clas.correct_set_size_array) / length(clas.correct_set_size_array)
    end

    # Perform correct set subsumption if enabled
    if b4ucs.parameters.do_correct_set_subsumption
        do_correct_set_subsumption!(b4ucs, correct_set)
    end
end

"""
    run_ga!(b4ucs::B4UCS, correct_set)

Execute genetic algorithm in correct set.
"""
function run_ga!(b4ucs::B4UCS, correct_set::Vector{B4Classifier})
    if isempty(correct_set)
        return
    end

    # Filter out classifiers with negative fitness
    positive_correct_set::Vector{B4Classifier} = filter(clas -> clas.fitness >= 0, correct_set)
    if isempty(positive_correct_set)
        return
    end

    # Check if it's time to run GA based on the average timestamp of the correct set
    if b4ucs.time_stamp - mapreduce(clas -> clas.time_stamp * clas.numerosity, +, correct_set) / mapreduce(clas -> clas.numerosity, +, correct_set) > b4ucs.parameters.theta_GA
        # Update timestamps
        for clas in correct_set
            clas.time_stamp = b4ucs.time_stamp
        end

        # Select parents and create offspring
        parent_1::B4Classifier = select_offspring(b4ucs, positive_correct_set)
        parent_2::B4Classifier = select_offspring(b4ucs, positive_correct_set)
        child_1::B4Classifier = deepcopy(parent_1)
        child_2::B4Classifier = deepcopy(parent_2)

        # Initialize children properties
        child_1.id, child_2.id = b4ucs.global_id, b4ucs.global_id + 1
        b4ucs.global_id += 2
        @simd for child in (child_1, child_2)
            child.numerosity = 1
            child.experience = 0
            child.correct_matching_array = zeros(Float64, b4ucs.env.num_actions)
            empty!(child.correct_set_size_array)
        end

        # Apply crossover with probability chi
        if rand() < b4ucs.parameters.chi
            apply_crossover!(child_1, child_2)
        end

        # Apply mutation and handle subsumption for each child
        @simd for child in (child_1, child_2)
            apply_mutation!(child, b4ucs.parameters.m0, b4ucs.parameters.mu)
            if b4ucs.parameters.do_GA_subsumption
                if does_subsume(parent_1, child, b4ucs.parameters.theta_sub, b4ucs.parameters.F0, b4ucs.parameters.tol_sub)
                    b4ucs.subsumption_occur_num += 1
                    parent_1.numerosity += 1
                elseif does_subsume(parent_2, child, b4ucs.parameters.theta_sub, b4ucs.parameters.F0, b4ucs.parameters.tol_sub)
                    b4ucs.subsumption_occur_num += 1
                    parent_2.numerosity += 1
                else
                    insert_in_population!(b4ucs, child)
                end
            else
                insert_in_population!(b4ucs, child)
            end
            delete_from_population!(b4ucs)
        end
    end
end

"""
    select_offspring(b4ucs::B4UCS, correct_set)::B4Classifier

Select parent classifier using either:
- Fitness-proportional selection (τ=0)
- Tournament selection (τ>0)
"""
function select_offspring(b4ucs::B4UCS, positive_correct_set::Vector{B4Classifier})::B4Classifier
    if b4ucs.parameters.tau == 0.
        # Roulette-Wheel Selection
        fitness_sum = sum([(clas.fitness ^ b4ucs.parameters.nu) * clas.matching_degree for clas in positive_correct_set])
        choice_point = rand() * fitness_sum

        fitness_sum = 0.
        for clas in positive_correct_set
            fitness_sum += (clas.fitness ^ b4ucs.parameters.nu) * clas.matching_degree
            if fitness_sum > choice_point
                return clas
            end
        end
    else
        # Tournament Selection
        parent = nothing
        for clas in positive_correct_set
            if parent == nothing || (parent.fitness ^ b4ucs.parameters.nu) * parent.matching_degree / parent.numerosity < (clas.fitness ^ b4ucs.parameters.nu) * clas.matching_degree / clas.numerosity
                for i in 1:clas.numerosity
                    if rand() < b4ucs.parameters.tau
                        parent = clas
                        break
                    end
                end
            end
        end
        if parent == nothing
            parent = rand(positive_correct_set)
        end
        return parent
    end
end

"""
    insert_in_population!(b4ucs::B4UCS, clas)

Insert classifier into population.
"""
function insert_in_population!(b4ucs::B4UCS, clas::B4Classifier)
    for c in b4ucs.population
        if is_equal_condition(c, clas)
            c.numerosity += 1
            return
        end
    end
    push!(b4ucs.population, clas)
end

"""
    delete_from_population!(b4ucs::B4UCS)

Maintain population size N using fitness-weighted deletion.
Applies deletion vote considering:
- Classifier age
- Fitness relative to population average
- Correct set participation
"""
function delete_from_population!(b4ucs::B4UCS)
    # Calculate the total numerosity of the population
    numerosity_sum::Float64 = mapreduce(clas -> clas.numerosity, +, b4ucs.population)
    # If the population size is within limits, do nothing
    if numerosity_sum <= b4ucs.parameters.N
        return
    end

    # Calculate average fitness of the population, using the power of nu
    average_fitness::Float64 = mapreduce(clas -> clas.fitness^b4ucs.parameters.nu, +, b4ucs.population) / numerosity_sum
    # Calculate the sum of deletion votes for all classifiers
    vote_sum::Float64 = mapreduce(clas -> deletion_vote(clas, average_fitness, b4ucs.parameters.theta_del, b4ucs.parameters.delta, b4ucs.parameters.nu), +, b4ucs.population)

    # Select a classifier for deletion using roulette wheel selection
    choice_point::Float64 = rand() * vote_sum
    vote_sum = 0.

    for clas in b4ucs.population
        vote_sum += deletion_vote(clas, average_fitness, b4ucs.parameters.theta_del, b4ucs.parameters.delta, b4ucs.parameters.nu)
        if vote_sum > choice_point
            # Decrease numerosity of the selected classifier
            clas.numerosity -= 1
            # If numerosity reaches zero, remove the classifier from the population
            if clas.numerosity == 0
                @views filter!(e -> e != clas, b4ucs.population)
            end
            return
        end
    end
end

"""
    do_correct_set_subsumption!(b4ucs::B4UCS, correct_set)

Perform subsumption in correct set.
"""
function do_correct_set_subsumption!(b4ucs::B4UCS, correct_set::Vector{B4Classifier})
    # Find the most general subsumer in the correct set
    cl::Any = nothing
    for c in correct_set
        if could_subsume(c, b4ucs.parameters.theta_sub, b4ucs.parameters.F0)
            if cl == nothing || is_more_general(c, cl, b4ucs.parameters.tol_sub)
                cl = c
            end
        end
    end
    # If a subsumer is found, subsume more specific classifiers
    if cl != nothing
        for c in correct_set
            if is_more_general(cl, c, b4ucs.parameters.tol_sub)
                b4ucs.subsumption_occur_num += 1
                cl.numerosity = cl.numerosity + c.numerosity
                # Remove subsumed classifier from correct set and population
                @views filter!(e->e!=c, correct_set)
                @views filter!(e->e!=c, b4ucs.population)
            end
        end
    end
end

"""
    do_correct_set_crispification!(b4ucs::B4UCS, correct_set)

Apply Ockham's Razor-inspired crispification.
Converts fuzzy rules to crisp intervals when:
- Experience > θ_sub
- Fitness > F0
"""
function do_correct_set_crispification!(b4ucs::B4UCS, correct_set::Vector{B4Classifier})
    for clas in correct_set
        if clas.experience > b4ucs.parameters.theta_sub && clas.fitness > b4ucs.parameters.F0
            # Select random non-crisp dimension
            fuzzy_dims::Vector{Int64} = [i for i in eachindex(clas.condition) if !(clas.condition[i].a == 1.0 && clas.condition[i].b == 1.0)]
            
            if isempty(fuzzy_dims) == false
                k::Int64 = rand(fuzzy_dims)
                # Convert to crisp interval (α=β=1)
                clas.condition[k].a = 1.0
                clas.condition[k].b = 1.0

                # Reset experience to prevent repeated crispification
                clas.experience = 0
                clas.correct_matching_array = zeros(Float64, b4ucs.env.num_actions)
                empty!(clas.correct_set_size_array)
            end
        end
    end
end

"""
    could_subsume(clas, θ_sub, F0)::Bool

Check if classifier meets subsumption prerequisites:
- Experience > θ_sub
- Fitness > F0
"""
could_subsume(clas::B4Classifier, theta_sub::Int, F0::Float64)::Bool = clas.experience > theta_sub && clas.fitness > F0

"""
    does_subsume(potential, target, θ_sub, F0, Tol_sub)::Bool

Check if potential classifier can subsume target.
"""
does_subsume(potential::B4Classifier, target::B4Classifier, theta_sub::Int, F0::Float64, tol_sub::Float64)::Bool = could_subsume(potential, theta_sub, F0) && is_more_general(potential, target, tol_sub)

"""
    deletion_vote(clas, average_fitness, θ_del, δ)::Float64

Calculate deletion vote.
"""
function deletion_vote(clas::B4Classifier, average_fitness::Float64, theta_del::Int, delta::Float64, nu::Float64)::Float64
    # Initial vote is based on the classifier's correct set size and numerosity
    vote::Float64 = clas.correct_set_size * clas.numerosity
    
    # If the classifier is experienced enough and its fitness is below average
    if clas.experience > theta_del && clas.fitness^nu < delta * average_fitness
        # Increase the vote, making it more likely to be deleted
        # The vote is scaled by the ratio of average fitness to the classifier's fitness
        vote *= average_fitness / max(clas.fitness^nu, 1e-12)
    end
    return vote
end

"""
    generate_correct_set(match_set, answer)::Vector{B4Classifier}

Form correct set [C] by selecting classifiers predicting the correct class.
"""
function generate_correct_set(match_set::Vector{B4Classifier}, answer::Int)::Vector{B4Classifier}
    # Filter the match set to include only classifiers whose highest weight corresponds to the correct answer
    return filter(clas -> argmax(clas.weight_array) == answer + 1, match_set)
end


