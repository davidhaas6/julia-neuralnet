# A multiclass single layer NN used for classifying the MNIST dataset
# David Haas
# 12/5/19
using HDF5, Distributions, Random
using Dates  # weights timestamp
using Plots, Images  # demo

# TODO Analyze error on each digit
# TODO determine if you need a bias node
# TODO Calculate bias and variance for the training error


#   ============================
#   === Neural net functions ===
#   ============================

# A numerically stable softmax -- no NaNs
function softmax(k, activations)
    log_C = -maximum(activations)
    return exp(activations[k] + log_C) / sum(exp(ak + log_C) for ak in activations)
end

# Calcualtes the error of the weights over a subset of the training data
function error(weights, input, target, test_idx)
    K = size(target, 2)
    D = size(input, 2)
    N = length(test_idx)

    output_act = zeros(K)
    y = zeros(K)
    err = 0
    err_rate = 0

    for n = 1:N
        x = input[test_idx[n], :]
        t = target[test_idx[n], :]
        
        for k = 1:K
            output_act[k] = sum(x[i] * weights[k,i] for i = 1:D)
        end
        for k = 1:K
            y[k] = softmax(k, output_act)
            if y[k] == 0
                y[k] += eps()
            elseif y[k] == 1
                y[k] -= eps()
            end
            err += t[k] * log(y[k])
        end

        if argmax(t) != argmax(y)
            err_rate += 1
        end
    end
    err_rate /= N

    return -err, err_rate
end

# Train the neural network to generate a weight matrix
function train(input, target; batch_size = 256, alpha = 0.001)
    N, K = size(target)
    D = size(input, 2)

    # Initialize weight and activation vectors
    weights = randn(K, D)  # TODO: Do I add a bias?
    best_weights = weights
    output_act = zeros(K)

    # Initialize training and holdout datasets
    N_train = N - round(Int64, N / 3) # number in training set
    idx = shuffle(1:N)      
    train_idx = idx[1:N_train]
    test_idx = idx[(N_train + 1):N]
    pdf = Uniform(1, N_train) # For sampling the batches

    # Initialize Adam parameters
    tau = 1
    m1_t = zeros(size(weights)...)
    v1_t = zeros(size(weights)...)
    B1 = 0.9
    B2 = 0.999
    epsilon = 1e-8

    # Initialize stop conditions and error calculations
    stop = false
    error_history = zeros(1)
    erate_history = zeros(1)
    push!(error_history, error(weights, input, target, test_idx)[1])
    push!(erate_history, error(weights, input, target, test_idx)[2])
    window_size = 10
    
    # Display training settings
    println("Training for batch_size = $(batch_size), alpha = $(alpha), K = $(K)")

    while !stop
        grad = zeros(K, D)

        # Calculate batch gradient
        for n = 1:batch_size
            sample = train_idx[round(Int64, rand(pdf, 1)[1])]
            x = input[sample,:]
            t = target[sample,:]

            # Forward Propagation
            y = zeros(K)
            for k = 1:K
                output_act[k] = sum(x[i] * weights[k,i] for i = 1:D)
            end
            for k = 1:K
                y[k] = softmax(k, output_act)
            end            
            
            # Calculate gradient -- dE/dw
            for j = 1:D
                for i = 1:K
                    grad[i,j] += (y[i] - t[i]) * x[j]
                end
            end
        end

        # Update weights with Adam
        for i = 1:D
            for k = 1:K
                g_t = grad[k,i]
                m1_t[k,i] = B1 * m1_t[k,i] + (1 - B1) * g_t
                v1_t[k,i] = B2 * v1_t[k,i] + (1 - B2) * g_t^2
                mhat_t = m1_t[k,i] / (1 - B1^tau)
                vhat_t = v1_t[k,i] / (1 - B2^tau)
                weights[k,i] -= alpha * mhat_t / (sqrt(vhat_t) + epsilon)
            end
        end

        # Calculate error
        validation_error, validation_error_rate = error(weights, input, target, test_idx)
        if validation_error < minimum(error_history)
            best_weights = weights
        end
        push!(error_history, validation_error)
        push!(erate_history, validation_error_rate)

        if tau % 20 == 0 || tau == 1
            println("\nIteration $(tau):")
            println("\tVal error = $(round(validation_error))")
            println("\tVal error rate = $(round(validation_error_rate*100,digits=2))%")
        end
        
        # Test stop conditions
        if tau > (window_size * 10)
            recent_err = mean(error_history[end-window_size:end])
            past_err = mean(error_history[end-window_size*2:end-window_size]) 

            recent_erate = mean(erate_history[end-window_size:end])
            past_erate = mean(erate_history[end-window_size*2:end-window_size]) 
            stop = recent_err > past_err && recent_erate > past_erate 
        end

        tau += 1
    end

    # Print the final training results 
    v_err, v_erate = error(best_weights, input, target, test_idx)
    t_err, t_erate = error(best_weights, input, target, train_idx)
    printstyled("\n","="^30, " CONVERGED ", "="^30, bold=true, color=:yellow)
    printstyled("\nRESULTS:\n", bold=true, color=:yellow)
    printstyled("\tTraining error = $(round(t_err))\n", color=:light_cyan)
    printstyled("\tTraining error rate = $(round(t_erate*100, digits=3))%\n", color=:light_cyan)
    printstyled("\n\tValidation error = $(round(v_err))\n", color=:light_green)
    printstyled("\tValidation error rate = $(round(v_erate*100, digits=3))%\n", color=:light_green)

    return best_weights, error_history, erate_history
end


#   ==========================
#   === Analysis functions ===
#   ==========================

# Runs forward propagation... used for the demo
function predict(image, weights)
    K,D = size(weights)
    output_act = zeros(K)
    y = zeros(K)
    for k = 1:K
        output_act[k] = sum(image[i] * weights[k,i] for i = 1:D)
    end
    for k = 1:K
        y[k] = softmax(k, output_act)
    end      
    return argmax(y)
end

# Runs the algorithm on a number of samples, plotting the prediction
# along with the image
function demo(weights; num_samples=100)
    K = size(weights, 1)
    _, _, images, labels = load_data(num_digits = K)
    
    # Pick random samples
    idx = shuffle(1:size(labels,1))

    for i = 1:num_samples
        sample = images[idx[i],:]
        label = argmax(labels[idx[i], :])-1
        img = Gray.(normedview(N0f8, reshape(sample, 28, 28)))
        titl = string("Predicted: ", predict(sample, weights)-1, " Label: ", label)
        p = plot(img, title=titl, xaxis=false, yaxis=false)
        display(p)
        sleep(1)
    end
end


#   ========================
#   === Helper functions ===
#   ========================

# Loads in the training and test data from mnist.h5 and preprocesses it
function load_data(;num_digits=10)
    # Read in data
    train_images = h5read("mnist.h5", "train/images")
    train_labels = h5read("mnist.h5", "train/labels")
    test_images = h5read("mnist.h5", "test/images")
    test_labels = h5read("mnist.h5", "test/labels")

    if num_digits != 10
        train_images = train_images[:,:, train_labels .< num_digits]
        train_labels = train_labels[train_labels .< num_digits]
        test_images = test_images[:,:, test_labels .< num_digits]
        test_labels = test_labels[test_labels .< num_digits]
    end

    # flatten images into 784-D vector
    train_images = reshape(train_images, size(train_images, 1) * size(train_images, 2), size(train_images, 3))'
    test_images = reshape(test_images, size(test_images, 1) * size(test_images, 2), size(test_images, 3))'

    # Encode labels as one-hot vectors
    onehot_train = zeros(UInt8, length(train_labels), num_digits)
    for (i, label) in enumerate(train_labels)
        onehot = zeros(num_digits)'
        onehot[label + 1] = 1
        onehot_train[i,:] = onehot
    end

    onehot_test = zeros(UInt8, length(test_labels), num_digits)
    for (i, label) in enumerate(test_labels)
        onehot = zeros(num_digits)'
        onehot[label + 1] = 1
        onehot_test[i,:] = onehot
    end
    
    return train_images, onehot_train, test_images, onehot_test
end

# Saves the weights and the validation error history to an h5 file
# Writes the training metadata to metadata.txt as well
function save_weights(weights, test_err_rate, batch_size, alpha, val_err_hist, val_rate_hist)
    K = size(weights,1)

    # Save h5
    folder = "./singlelayer-weights/"
    timestamp = Dates.format(Dates.now(), "mm_dd-HH_MM")
    path = folder * "weights-" * timestamp * ".h5"
    h5write(path, "weights", weights)
    h5write(path, "validation-err-hist", val_err_hist)
    h5write(path, "validation-rate-hist", val_err_hist)
    println("Data saved to $path")

    # Write metadata
    open(folder * "metadata.txt", "a") do file
        write(file, "\nweights-" * timestamp * ".h5\n");
        write(file, "\tTest error rate = $(round(test_err_rate*100, digits=2))%\n")
        write(file, "\tBatch size = $batch_size\n\tAlpha = $alpha\n\tK = $K\n")
    end
end

# === Main ===
# runs the whole shebang
# weights can either be a K-by-D matrix of weights or a path to an h5 file
# with the weights matrix stored in "weights" 
function main(;weights=false)
    # Run pre-trained data and quit
    if weights != false
        @assert weights isa String || weights isa Array{Float64,2}
        if weights isa String
            @assert endswith(weights, ".h5")
            weights = h5read(weights, "weights")
        end
        _, _, test_images, test_labels = load_data(num_digits=size(weights,1))
        test_err, test_erate = error(weights,test_images,test_labels,1:size(test_labels,1))
        printstyled("\tTest error = $(round(test_err, digits=5))\n", color=:light_magenta)
        printstyled("\tTest error rate = $(round(test_erate*100, digits=3))%\n", color=:light_magenta)
        return;
    end

    # Hyperparameters
    batch_size = 16348
    alpha = 2

    # Train
    train_images, train_labels, test_images, test_labels = load_data(num_digits=10)
    @time weights, v_hist, v_rate_hist = train(train_images, train_labels, batch_size=batch_size, alpha=alpha)

    # Test
    test_err, test_erate = error(weights,test_images,test_labels,1:size(test_labels,1))
    printstyled("\n\tTest error = $(round(test_err, digits=5))\n", color=:light_magenta)
    printstyled("\tTest error rate = $(round(test_erate*100, digits=3))%\n", color=:light_magenta)

    # Save
    save_weights(weights, test_erate, batch_size, alpha, v_hist, v_rate_hist);

    return weights
end

final_weights = main()