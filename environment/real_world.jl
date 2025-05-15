```
Real-World Problems
```

using CSV, DataFrames, Random

mutable struct Environment
    seed::Int64
    is_exploit::Bool
    num_actions::Int64
    state_length::Int64
    row_index::Int64
    train_data::Array{Union{Float64, Int64, String}, 2}
    test_data::Array{Union{Float64, Int64, String}, 2}
    all_data::Array{Union{Float64, Int64, String}, 2}
    file_path::String
    index_array::Vector{Int64}
end

function Environment(args)::Environment
    train_data, test_data, all_data = get_train_and_test_data_and_all_data(args["csv"])
    num_actions, state_length = get_data_information(args["csv"])
    return Environment(1, false, num_actions, state_length, 0, train_data, test_data, all_data, args["csv"], Vector(1:size(train_data, 1)))
end

function get_train_and_test_data_and_all_data(filename::String)::Tuple{Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}}
    all_data = CSV.File(filename; header=false) |> DataFrame
    all_data = Matrix(all_data)

    train_data_ratio::Float64 = 0.9 # Shuffle-Split Cross Validation (Training:Testing = 9:1)

    train_data_length::Int64 = Int(floor(size(all_data, 1) * train_data_ratio))
    test_data_length::Int64 = size(all_data, 1) - train_data_length

    train_data = Array{Any}(undef, train_data_length, size(all_data, 2))
    test_data = Array{Any}(undef, test_data_length, size(all_data, 2))

    return train_data, test_data, all_data
end

function get_data_information(filename::String)::Tuple{Int64, Int64}
    all_data = CSV.read(filename, DataFrame, header=false)
    num_actions::Int64 = length(Set(all_data[:,size(all_data, 2)]))
    state_length::Int64 = size(all_data, 2) - 1
    return num_actions, state_length
end

function normalize_columns(data_train, data_test)::Tuple{Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}}
      @simd for j = 1:(size(data_train, 2) - 1)
        col = data_train[:, j]
        # Calculate min and max for each column
        col_without_missing = col[col .!= "?"]
        col_without_missing = map(x -> parse(Float64, string(x)), col_without_missing)
        col_min = minimum(col_without_missing)
        col_max = maximum(col_without_missing)
        for data in [data_train, data_test]
             @simd for i = 1:size(data, 1)
                if data[i, j] != "?"
                    # Normalize data to the range 0 to 1
                    v = parse(Float64, string(data[i, j]))
                    if (col_min == col_max)
                        data[i, j] = 0.5
                    else
                        data[i, j] = max(0, min(1, (v - col_min) / (col_max - col_min)))
                    end
                else
                    data[i, j] = "?"
                end
            end
        end
    end
    return data_train, data_test
end

# Per epoch
function shuffle_index_array_and_reset_row_index!(self::Environment)
    rng = MersenneTwister(self.seed)
    if self.row_index % size(self.train_data, 1) == 0
        self.index_array = Vector(1:size(self.train_data, 1))
        shuffle!(rng, self.index_array)
        self.row_index = 0
    end
end

# Per seed
function shuffle_train_and_test_data!(self::Environment)
    rng = MersenneTwister(self.seed)
    train_data_length::Int64 = size(self.train_data, 1)
    test_data_length::Int64 = size(self.test_data, 1)

    train_data = Array{Any}(undef, train_data_length, size(self.all_data, 2))
    test_data = Array{Any}(undef, test_data_length, size(self.all_data, 2))

    # Generate a random permutation of row indices using shuffle
    perm::Vector{Int64} = shuffle(rng, 1:train_data_length + test_data_length)

    # Rearrange the rows of the array using the permutation
    self.all_data = self.all_data[perm, :]

    # Generate train data
    for i = 1:train_data_length
        train_data[i, :] = self.all_data[i, :]
    end
    # Generate test data
    for i = 1: test_data_length
        test_data[i, :] = self.all_data[train_data_length + i, :]
    end

    self.train_data, self.test_data = normalize_columns(train_data, test_data)

end

function state(self::Environment)::Vector{Union{Float64, Int64, String}}
    if self.is_exploit == false
        # Use train data
        shuffle_index_array_and_reset_row_index!(self)
        self.row_index += 1
        return self.train_data[self.index_array[self.row_index], 1:size(self.train_data,2)-1]
    else
        error("This function is not available during testing.")
    end
end

function answer(self::Environment, state::Vector{Union{Float64, Int64, String}})::Int64
    if self.is_exploit == false
        # Use train data
        return Int64(self.train_data[self.index_array[self.row_index], size(self.train_data, 2)])
    else
        error("This function is not available during testing.")
    end
end


function get_environment_name(self::Environment)::String
    return "$(self.file_path)"
end

function make_matrix_csv(filename::String, list)
    tbl = Tables.table(list)
    CSV.write(filename, tbl, header=false)
end
