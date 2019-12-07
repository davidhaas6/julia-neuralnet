# A multiclass single layer NN used for classifying the MNIST dataset
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
function softmax(k, activations)
    log_C = -maximum(activations)
    return exp(activations[k] + log_C) / sum(exp(ak + log_C) for ak in activations)
end

# Calcualtes the error of the weights over a subset of the training data
function error(weights, input, target, test_idx)
    N = length(test_idx)
    err = 0
    err_rate = 0

    for n = 1:N
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
            err += t[k] * log(y[k])
        end

        if argmax(t) != argmax(y)
            err_rate += 1
        end
    end
    err_rate /= N
    err /= N
    return -err, err_rate
end

# Runs forward propagation
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
# M -> number of hidden nodes
# data_usage is in percent
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
    window_size = 7
    
    # Display training settings
    println("Training for batch_size = $batch_size, alpha = $alpha, K = $K, M = $M")

    while !stop
        grad1 = zeros(size(weights1))
        grad2 = zeros(size(weights2))

        # Calculate batch gradient
        for n = 1:batch_size
            sample_idx = train_idx[round(Int64, rand(pdf, 1)[1])]
            x = input[sample_idx,:]
            t = target[sample_idx,:]

            # Forward Propagation
            y, z, hidden_act = forward_prop((weights1, weights2), x)
            
            # Backpropagate
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
        weights = weights1, weights2
        validation_error, validation_error_rate = error(weights, input, target, test_idx)
        if validation_error < minimum(error_history)
            best_weights = copy(weights)
        end
        push!(error_history, validation_error)
        push!(erate_history, validation_error_rate)

        #if tau % 20 == 0 || tau == 1
            println("\nIteration $(tau):")
            println("\tNorm val error = $(round(validation_error,digits=3))")
            println("\tVal error rate = $(round(validation_error_rate*100,digits=2))%")
        #end
        
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
    printstyled("\tTraining error = $(round(t_err, digits=3))\n", color=:light_cyan)
    printstyled("\tTraining error rate = $(round(t_erate*100, digits=3))%\n", color=:light_cyan)
    printstyled("\n\tValidation error = $(round(v_err, digits=3))\n", color=:light_green)
    printstyled("\tValidation error rate = $(round(v_erate*100, digits=3))%\n", color=:light_green)

    return best_weights, error_history, erate_history
end


#   ==========================
#   === Analysis functions ===
#   ==========================

# Runs forward propagation... used for the demo
function predict(x, weights)
    y, _, _ = forward_prop(weights,x)
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
function save_weights(weights, test_err_rate, batch_size, alpha, M, val_err_hist, val_rate_hist)
    K = size(weights[2],1)

    # Save h5
    folder = "./twolayer-weights/"
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
            w1 = h5read(weights, "w1")
            w2 = h5read(weights, "w2")
            weights = w1, w2
        end
        _, _, test_images, test_labels = load_data(num_digits=size(weights,1))
        test_err, test_erate = error(weights,test_images,test_labels,1:size(test_labels,1))
        printstyled("\tTest error = $(round(test_err, digits=5))\n", color=:light_magenta)
        printstyled("\tTest error rate = $(round(test_erate*100, digits=3))%\n", color=:light_magenta)
        return;
    end

    # Hyperparameters   m=500, bs=1025, alpha=0.01 seemed somewhat promising ... 59% in 37 iter
    batch_size = 64
    alpha = .01
    M = 1000

    # Train
    train_images, train_labels, test_images, test_labels = load_data(num_digits=10)
    @time weights, v_hist, v_rate_hist = train(train_images, train_labels, M, batch_size, alpha, data_usage=25)

    # Test
    test_err, test_erate = error(weights,test_images,test_labels,1:size(test_labels,1))
    printstyled("\n\tTest error = $(round(test_err, digits=5))\n", color=:light_magenta)
    printstyled("\tTest error rate = $(round(test_erate*100, digits=3))%\n", color=:light_magenta)

    # Save
    save_weights(weights, test_erate, batch_size, alpha, M, v_hist, v_rate_hist);

    return weights
end

final_weights = main()