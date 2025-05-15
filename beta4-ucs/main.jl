using ArgParse
using Random
using CSV
using DataFrames
using Dates

function parse_commandline()
    s = ArgParseSettings(description="The β4-UCS classifier system")
    @add_arg_table s begin
        "--num_trials"
            help = "The number of trials"
            arg_type = Int64
            default = 30
        "--epoch", "-e"
            help = "The number of epochs"
            arg_type = Int64
            default = 50
        "--csv"
            help = "A CSV file for classification"
            arg_type = String
            default = nothing
        "-a", "--all"
            help = "Whether to conduct experiments using all CSV files"
            arg_type = Bool
            default = false
        "-N"
            help = "The maximum size of the population"
            arg_type = Int64
            default = 2000
        "--F0"
            help = "The fitness threshold"
            arg_type = Float64
            default = 0.99
        "--nu"
            help = "The exponent in the power function for parent selection"
            arg_type = Float64
            default = 1.0
        "--theta_GA"
            help = "The threshold for the GA application in a correct set"
            arg_type = Int64
            default = 50
        "--chi"
            help = "The probability of applying crossover"
            arg_type = Float64
            default = 0.8
        "--mu"
            help = "The probability of mutating one allele"
            arg_type = Float64
            default = 0.04
        "--theta_del"
            help = "The experience threshold over which the fitness of a classifier may be considered in its deletion probability"
            arg_type = Int64
            default = 50
        "--delta"
            help = "The fraction of the mean fitness of the population below which the fitness of a classifier may be considered in its vote for deletion"
            arg_type = Float64
            default = 0.1
        "--theta_sub"
            help = "The experience of a classifier required to be a subsumer"
            arg_type = Int64
            default = 50
        "--r0"
            help = "The maximum value of a spread in the covering operator"
            arg_type = Float64
            default = 1.0
        "--m0"
            help = "The maximum change of a lower value or a upper value in mutation"
            arg_type = Float64
            default = 0.1
        "--tau"
            help = "The tournament size for selection [Butz et al., 2003] (set 0 to use the roulette-wheel selection)"
            arg_type = Float64
            default = 0.4
        "--do_GA_subsumption"
            help = "Whether offspring are to be tested for possible logical subsumption by parents"
            arg_type = Bool
            default = true
        "--do_correct_set_subsumption"
            help = "Whether correct sets are to be tested for subsuming classifiers"
            arg_type = Bool
            default = true
        "--do_correct_set_crispification"
            help = "Whether correct sets are to be crispified"
            arg_type = Bool
            default = true
        "--P_hash"
            help = "The probability of Don't Care in covering"
            arg_type = Float64
            default = 0.33
        "--theta_exploit"
            help = "The experience threshold for class inference"
            arg_type = Float64
            default = 10.0
        "--tol_sub"
            help = "Subsumption tolerance"
            arg_type = Float64
            default = 0.01
        "--all_metrics"
            help = "Whether to output all metrics (if false, Precision, Recall, and F1 are not shown in the log)"
            arg_type = Bool
            default = true
    end
    return parse_args(s)
end

function main_csv(args; now_str=Dates.format(Dates.now(), "Y-m-d-H-M-S"))
    env = Environment(args)
    param = Parameters(args)
    helper = Helper(env)

    println("[ Settings ]")
    println("    Environment = ", get_environment_name(env))
    println("         #Epoch = ", args["epoch"])
    println("          #Inst = ", size(env.all_data, 1))
    println("           #Fea = ", env.state_length)
    println("            #Cl = ", env.num_actions)

    println("\n[ β4-UCS General Parameters ]")
    println("                 N = ", param.N)
    println("                F0 = ", param.F0)
    println("                 ν = ", param.nu)
    println("              θ_GA = ", param.theta_GA)
    println("                 χ = ", param.chi)
    println("                 μ = ", param.mu)
    println("             θ_del = ", param.theta_del)
    println("                 δ = ", param.delta)
    println("             θ_sub = ", param.theta_sub)
    println("                m0 = ", param.m0)
    println("                r0 = ", param.r0)
    println("         θ_exploit = ", param.theta_exploit)
    println("               P_# = ", param.P_hash)
    println("           Tol_sub = ", param.tol_sub)
    println("   doGASubsumption = ", Bool(param.do_GA_subsumption))
    println("   doCSSubsumption = ", Bool(param.do_correct_set_subsumption))
    println("doCSCrispification = ", Bool(param.do_correct_set_crispification))

    println("\n[ β4-UCS Optional Settings ]")
    if param.tau != 0.
        println("                 τ = ", param.tau)
    end

    @time for n in 1:args["num_trials"]
        start_time = time()
        ucs::B4UCS = B4UCS(env, param)
        env.seed = n
        Random.seed!(env.seed)
        shuffle_train_and_test_data!(env)
        println("\n[ Seed $(env.seed) / $(args["num_trials"]) ]\n")
        time_list = Vector{Float64}(undef, 1)
        summary_list = Array{Any}(undef, args["epoch"] + 1, 13)

         for e in 1:args["epoch"]
            # Training Phase
            env.is_exploit = false
            num_iter = size(env.train_data, 1)
            for i in 1:num_iter
                run_experiment(ucs)
            end

            # Test Phase
            env.is_exploit = true
            train_accuracy::Float64, train_precision::Float64, train_recall::Float64, train_f1::Float64 = get_scores_per_epoch(env.seed, ucs, env.train_data)
            test_accuracy::Float64, test_precision::Float64, test_recall::Float64, test_f1::Float64 = get_scores_per_epoch(env.seed, ucs, env.test_data)
            popsize::Int64 = length(ucs.population)

            # Output log
            summary_list = output_full_score_for_csv(e, args["epoch"], env, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1, popsize, ucs.covering_occur_num, ucs.subsumption_occur_num, summary_list, args["all_metrics"])
            ucs.covering_occur_num = ucs.subsumption_occur_num = 0.
        end

        elapsed_time = time() - start_time
        time_list[1] = elapsed_time

        if !args["all"]
            dir_path = "./beta4-ucs/result/" * get_environment_name(env) * "/" * now_str * "/seed" * string(env.seed)
        else
            dir_path = "./all_" * now_str * "/beta4-ucs/" *  basename(get_environment_name(env)) * "/seed" * string(env.seed)
        end
        mkpath(dir_path)
        make_one_column_csv(joinpath(dir_path, "time.csv"), time_list)
        make_matrix_csv(joinpath(dir_path, "classifier.csv"), make_classifier_list(helper, ucs))
        make_matrix_csv(joinpath(dir_path, "summary.csv"), summary_list)
        make_matrix_csv(joinpath(dir_path, "parameter.csv"), make_parameter_list(args))
    end
end

function main_all_csv(args)
    now = Dates.now()
    now_str = Dates.format(now, "Y-m-d")
    dir_path::String = "./dataset/"
    csv_list_array::Vector{String} = [
    
        # 25 datasets used in the main article
        "caesarian",
        "cancer",
        "12carry",
        "chk3_5",
        "cmux",
        "column_3C_weka",
        "diabetes",
        "ecoli",
        "fruit",
        "glass",
        "horse-colic",
        "mammographic_masses",
        "11mop",
        "multiple",
        "20mux",
        "npha",
        "Paddy-Leaf",
        "pistachio",
        "pumpkin",
        "raisin",
        "soybean",
        "tae",
        "wine",
        "wpbc",
        "yeast"
    ]

    for csv::String in csv_list_array
        args["csv"] = dir_path * csv * ".csv"
        main_csv(args; now_str)
    end
end


args = parse_commandline()
args = parse_commandline()
include("../environment/real_world.jl")
include("parameters.jl")
include("helper.jl")
include("ucs.jl")

if args["csv"] == nothing && args["all"]
    main_all_csv(args)
elseif args["csv"] != nothing && !args["all"]
    main_csv(args)
else
    error("Set either args[\"csv\"] or args[\"all\"]")
end