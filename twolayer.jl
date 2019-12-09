# A multiclass two layer NN used for classifying the MNIST dataset
# David Haas
# 12/5/19

using HDF5, Distributions, Random
using Dates  # weights timestamp
using Plots, Images  # demo


#   ============================
#   === Neural net functions ===
#   ============================

sigmoid(a) = 1/(1+exp(-a))
sigmoid_prime(a) = sigmoid(a) * (1 - sigmoid(a))

# A numerically stable softmax -- no NaNs
# Inputs:
#   activations: a vector of the activation values for the nodes in the softmax layer
#   k: The index of the node to run softmax on
# Outputs:
#   The result of the softmax calculation
function softmax(k, activations)
    log_C = -maximum(activations)
    return exp(activations[k] + log_C) / sum(exp(ak + log_C) for ak in activations)
end

# Calcualtes the cross-entropy error of the weights over a subset of the training data w/ multithreading
# Inputs:
#   weights: a tuple of the 1st and 2nd layer weights
#   input: the input data
#   target: the labels for the input data
#   test_idx: the indicies in input to calculate the error over
# Outputs:
#   The cross-entropy error and error rate over the input
function error(weights, input, target, test_idx)
    N = length(test_idx)
    err = Threads.Atomic{Float64}(0)
    err_rate = Threads.Atomic{Int}(0)

    Threads.@threads for n = 1:N
        x = input[test_idx[n], :]
        t = target[test_idx[n], :]
        
        y, _, _ = forward_prop(weights,x)

        # Calculate cross-entropy error for the sample and add to total error
        for k in 1:length(y)
            if y[k] == 0
                y[k] += eps()
            elseif y[k] == 1
                y[k] -= eps()
            end
            Threads.atomic_add!(err, Float64(t[k] * log(y[k])))
        end

        if argmax(t) != argmax(y)
            Threads.atomic_add!(err_rate, 1)
        end
    end

    error = -err[]
    error_rate = err_rate[] / N
    return error, error_rate
end

# Forward propagates the input/weight combinations to the output nodes
# Inputs:
#   weights: a tuple of the 1st and 2nd layer weights
#   x: the input sample
# Outputs:
#   the outputs of the softmax layer, hidden layer, and the hidden layer activations
function forward_prop(weights, x)
    w1, w2 = weights
    M,D = size(w1)
    K = size(w2,1)
    hidden_act = zeros(M)
    output_act = zeros(K)
    z = zeros(length(hidden_act)+1)
    y = zeros(length(output_act))

    z[1] = 1  # bias
    for m = 1:M
        hidden_act[m] = sum(x[i] * w1[m,i] for i = 1:D)
        z[m+1] = sigmoid(hidden_act[m])
    end

    # Layer 1->2
    for k = 1:K
        output_act[k] = sum(z[i] * w2[k,i] for i = 1:M+1)
    end
    for k = 1:K
        y[k] = softmax(k, output_act)
    end           

    return y, z, hidden_act
end

# Train the neural network to generate a weight matrix
# Inputs:
#   input: the input data
#   target: the labels for the input data
#   M: the number of hidden nodes
#   batch_size: the batch size for stochastic gradient descent
#   alpha: the learning rate for Adam
#   data_usage: the percentage of the input data to train off of
# Outputs:
#   A tuple of the 1st and 2nd layer weights,
#   Vectors of the validation error and error rate throughout training
#   The number of iterations until convergence
function train(input, target, M, batch_size, alpha; data_usage=100)
    # Scale data usage
    nrows = Int(round(size(target,1) * data_usage/100))
    target = target[1:nrows,:]
    input = input[1:nrows,:]
    
    # Get dimensions
    N, K = size(target)
    D = size(input, 2)

    # Initialize weight and activation vectors
    weights1 = randn(M, D)
    weights2 = randn(K, M+1)
    best_weights = weights1, weights2

    # Initialize training and holdout datasets
    N_train = N - round(Int64, N / 3) # number in training set
    idx = shuffle(1:N)      
    train_idx = idx[1:N_train]
    test_idx = idx[(N_train + 1):N]
    pdf = Uniform(1, N_train) # For sampling the batches

    # Initialize Adam parameters
    tau = 1
    m1_t = zeros(size(weights1))
    v1_t = zeros(size(weights1))
    m2_t = zeros(size(weights2))
    v2_t = zeros(size(weights2))
    B1 = 0.9
    B2 = 0.999
    epsilon = 1e-8

    # Initialize stop conditions and error calculations
    stop = false
    error_history = zeros(1)
    erate_history = zeros(1)
    init_err, init_erate = error(best_weights, input, target, test_idx)
    push!(error_history, init_err)
    push!(erate_history, init_erate)
    window_size = 3
    validation_period = 50  # Checks validation error ever X epochs
    
    # Display training settings
    printstyled("\n","="^20, " Training Info ", "="^20, "\n", bold=true)
    println("Model parameters:")
    println("\tM = $M\n\tAlpha = $alpha\n\tBatch size = $batch_size\n\tK = $K")
    println("\nTraining parameters:")
    println("\tValidation period = $validation_period iters")
    println("\tSample usage = $(data_usage)%")
    println("\tWindow size = $window_size")
    println("\tThreads = $(Threads.nthreads())")

    printstyled("\n", "="^20, " Begin ", "="^20, "\n", bold=true)
    while !stop
        print(".")
        grad1 = zeros(size(weights1))
        grad2 = zeros(size(weights2))

        # Calculate batch gradient
        batch_indicies = train_idx[round.(Int, rand(pdf, batch_size))]
        for idx in batch_indicies
            x = input[idx,:]
            t = target[idx,:]

            # Forward Propagation
            y, z, hidden_act = forward_prop((weights1, weights2), x)

            # Backpropagation
            # Layer 2
            for i = 1:M+1
                for j = 1:K
                    if i == 1
                        grad2[j,i] += (y[j] - t[j])
                    else
                        grad2[j,i] += (y[j] - t[j]) * z[i]
                    end
                end
            end

            # Layer 1
            for j = 1:M
                delta_j = sigmoid_prime(hidden_act[j]) * sum(weights2[k,j] * (y[k] - t[k]) for k in 1:K)
                for i = 1:D
                    grad1[j,i] += delta_j * x[i]  
                end
            end

        end

        # Update weights with Adam
        # layer 2
        for i = 1:M+1
            for j = 1:K
                g_t = grad2[j,i]
                m2_t[j,i] = B1*m2_t[j,i] + (1-B1)*g_t
                v2_t[j,i] = B2*v2_t[j,i] + (1-B2)*g_t^2
                mhat_t = m2_t[j,i] / (1-B1^tau)
                vhat_t = v2_t[j,i] / (1-B2^tau)
                weights2[j,i] -= alpha * mhat_t / (sqrt(vhat_t) + epsilon)
            end
        end

        # layer 1
        for i = 1:D
            for j = 1:M
                g_t = grad1[j,i]
                m1_t[j,i] = B1*m1_t[j,i] + (1-B1)*g_t
                v1_t[j,i] = B2*v1_t[j,i] + (1-B2)*g_t^2
                mhat_t = m1_t[j,i] / (1-B1^tau)
                vhat_t = v1_t[j,i] / (1-B2^tau)
                weights1[j,i] -= alpha * mhat_t / (sqrt(vhat_t) + epsilon)
            end
        end

        # Calculate error
        if tau % validation_period == 0
            weights = weights1, weights2
            validation_error, validation_error_rate = error(weights, input, target, test_idx)
            if validation_error < minimum(error_history)
                best_weights = copy(weights)
            end
            push!(error_history, validation_error)
            push!(erate_history, validation_error_rate)

            println("\nIteration $(tau):")
            println("\tVal error = $(round(validation_error,digits=3))")
            println("\tVal error rate = $(round(validation_error_rate*100,digits=2))%")
            
            # Test stop conditions
            if tau > (window_size * validation_period * 3)
                recent_err = mean(error_history[end-window_size:end])
                past_err = mean(error_history[end-window_size*2:end-window_size]) 

                recent_erate = mean(erate_history[end-window_size:end])
                past_erate = mean(erate_history[end-window_size*2:end-window_size]) 
                stop = recent_err > past_err && recent_erate > past_erate 
            end
        end

        tau += 1
    end

    # Print the final training results
    v_err, v_erate = error(best_weights, input, target, test_idx)
    t_err, t_erate = error(best_weights, input, target, train_idx)
    printstyled("\n","="^20, " CONVERGED ", "="^20, bold=true, color=:yellow)
    printstyled("\nRESULTS:\n", bold=true, color=:yellow)
    printstyled("\tTraining error = $(round(t_err, digits=3))\n", color=:light_cyan)
    printstyled("\tTraining error rate = $(round(t_erate*100, digits=3))%\n", color=:light_cyan)
    printstyled("\n\tValidation error = $(round(v_err, digits=3))\n", color=:light_green)
    printstyled("\tValidation error rate = $(round(v_erate*100, digits=3))%\n\n", color=:light_green)

    return best_weights, error_history, erate_history, tau
end


#   ==========================
#   === Analysis functions ===
#   ==========================

# Runs forward propagation and returns the most likely output
# Inputs:
#   x: the data sample to predict the class of
#   weights: a tuple containing the weights for the model
# Outputs:
#   The predicted class of x
function predict(x, weights)
    y, _, _ = forward_prop(weights,x)
    return argmax(y)
end

# Runs the algorithm on a number of samples, plotting the prediction and image
# Inputs:
#   weights: a tuple containing the weights for the model
#   num_samples: The number of samples to display and predict
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

# Loads in weights and calculates the test set accuracy
# Inputs:
#   weights_path: The path to an .h5 file of the weights
# Outputs:
#   The error and error rate of the test set
function test_model(weights_path)
    @assert endswith(weights_path, ".h5")

    # Load data and model
    weights = h5read(weights_path, "w1"), h5read(weights_path, "w2")
    _, _, test_images, test_labels = load_data()

    println("Testing model...")
    test_err, test_erate = error(weights,test_images,test_labels,1:size(test_labels,1))
    printstyled("Test error = $(round(test_err, digits=2))\n", color=:light_magenta)
    printstyled("Test error rate = $(round(test_erate*100, digits=3))%\n\n", color=:light_magenta)

    return test_err, test_erate
end

# Calculates the error rate for each digit
# Inputs:
#   weights: either a tuple containing weights or a path to a .h5 file containing the weights
#   path: True if weights is a path to an h5 file
function digit_errors(weights; path=false)
    if path
        weights = h5read(weights, "w1"), h5read(weights, "w2")
    end

    _, _, test_images, test_labels = load_data()
    num_wrong = zeros(10)'
    num_samples = zeros(10)'
    
    for n = 1:size(test_images,1)
        # Load data in
        x = test_images[n,:]
        t = test_labels[n,:]
        digit = argmax(t)-1

        if (predict(x, weights)-1) != digit
            num_wrong[digit+1] += Float64(1)
        end

        num_samples[digit+1] += Float64(1)
    end
    error_rate = num_wrong ./ num_samples * 100
    
    for digit in 1:10
        println("Error rate for $(digit-1) = $(round(error_rate[digit],digits=2))%")
    end
end


#   ========================
#   === Helper functions ===
#   ========================

# Loads in the training and test data from mnist.h5 and preprocesses it
# Inputs:
#   num_digits: The number of digits (max 10) to load in
# Outputs:
#   A 4-element tuple containing the training and test images and labels
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
# Inputs:
#   weights: a tuple containing the weights for the model
#   test_err_rate: The error rate of the model over the test set
#   batch_size: The batch size the model was trained on
#   alpha: The alpha the model was trained on 
#   M: The number of hidden nodes the model contains
#   data_usage: The percent of the input data the model was trained on
#   val_err_hist: A vector of the error history as the model was training
#   val_rate_hist: A vector of the error rate history as the model was training
#   niter: The number of iterations the model took to converge
function save_weights(weights, test_err_rate, batch_size, alpha, M, data_usage, val_err_hist, val_rate_hist, niter)
    K = size(weights[2],1)

    # Ensure folder exists
    folder = "./twolayer-weights/"
    if !isdir(folder)
        mkdir(folder)
    end

    # Save h5
    timestamp = Dates.format(Dates.now(), "mm_dd-HH_MM")
    path = folder * "weights-" * timestamp * ".h5"
    h5write(path, "w1", weights[1])
    h5write(path, "w2", weights[2])
    h5write(path, "validation-err-hist", val_err_hist)
    h5write(path, "validation-rate-hist", val_err_hist)
    println("Data saved to $path")

    # Write metadata
    open(folder * "metadata.txt", "a") do file
        write(file, "\nweights-" * timestamp * ".h5\n");
        write(file, "\tTest error rate = $(round(test_err_rate*100, digits=2))%\n")
        write(file, "\tBatch size = $batch_size\n\tAlpha = $alpha\n\tM = $M\n\tK = $K\n")
        write(file, "\tData usage = $(data_usage)%\n")
        write(file, "\tNum iter = $niter\n")
        write(file,"\n")
    end
end

# === Main ===
# Loads in MNIST data, trains a model, tests it, then saves it.
# Inputs:
#   M: The number of hidden nodes in the network
#   alpha: The learning rate of Adam
#   batch_size: The number of samples to use in a batch
#   data_usage: The percent of the data to train off of
# Outputs:
#   A tuple of the weights the model converged on
function main(M, alpha, batch_size; data_usage=100)
    # Train
    train_images, train_labels, test_images, test_labels = load_data()
    @time model = train(train_images, train_labels, M, batch_size, alpha, data_usage=data_usage)
    weights, v_hist, v_rate_hist, n_iter = model

    # Test
    test_err, test_erate = error(weights,test_images,test_labels,1:size(test_labels,1))
    printstyled("\n\tTest error = $(round(test_err, digits=5))\n", color=:light_magenta)
    printstyled("\tTest error rate = $(round(test_erate*100, digits=3))%\n", color=:light_magenta)

    # Save
    save_weights(weights, test_erate, batch_size, alpha, M, data_usage, v_hist, v_rate_hist, n_iter);
    println("\n\n")
    return weights
end