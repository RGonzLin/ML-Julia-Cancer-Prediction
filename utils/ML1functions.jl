# To use the functions contained in this file: include("ML1functions.jl")

using Flux
using Flux.Losses
using Statistics
using DelimitedFiles
using Plots
using Random
using ScikitLearn
using UrlDownload
using DataFrames
using CSV

using ScikitLearn:predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import ensemble:VotingClassifier

function oneHotEncoding(feature::AbstractArray{<:Any,1},      
    classes::AbstractArray{<:Any,1})

    # Define global variables
    global categoricalOutput

    if size(classes)[1] == 2 # For 2 categories   
    # 1st category encoded as 0, and 2nd category encoded as 1
    encodedFeatures = feature .== classes[2]
    # Change type from BitVector to BitMatrix
    encodedFeatures = reshape(encodedFeatures,(size(Int.(encodedFeatures))[1],1))

    elseif size(classes)[1] > 2 # For more than 2 categories
        # Initialize counter
        c = 0
        for class in classes 
            if c == 0
                # Create a clasification vector containing the first class
                categoricalOutput = feature .== class
            else
                # Create a clasification vector for the curret class
                currentClass = feature .== class
                # Create a classification vector containing all classes
                append!(categoricalOutput, currentClass)
            end
            c += 1
        end
        # Define number of rows and columns 
        rows = Int(size(categoricalOutput)[1]/size(classes)[1])
        columns = size(classes)[1]
        encodedFeatures = reshape(categoricalOutput,(rows,columns)) 
    end

    # Return encoded features
    return encodedFeatures

end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature,unique(feature));

function oneHotEncoding(feature::AbstractArray{Bool,1})
    
    return reshape(feature,(0,size(feature)[1]))
    
end

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    
    # Calculate min and max values for each column
    minValues = minimum(dataset,dims=1)
    maxValues = maximum(dataset,dims=1) 
    
    # Pack min and max values into a tuple
    minMaxValues = (minValues,maxValues)
    
    # Return tuple with min and max values
    return minMaxValues

end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    
    # Calculate mean and standard deviation
    meanValues = mean(dataset,dims=1) 
    stdValues = std(dataset,dims=1) 
    
    # Pack mean and std values into a tuple
    meanStdValues = (meanValues,stdValues)
    
    # Return tuple with mean and std values
    return meanStdValues

end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    
    # Normalize between 0 and 1
    normInput = (dataset .- normalizationParameters[1])./(normalizationParameters[2] .- normalizationParameters[1])
    
    # Store dimensions of array
    dimensions = size(normInput)

    # Vectorize array for easy indexing
    normInput = vec(normInput)

    # Replace NaN values (product of division by 0 when max==min) by zeroes
    x=1
    for i in normInput
        if isequal(i,NaN) #i == NaN 
            normInput[x] = 0
        end 
        x += 1 
    end

    # Reshape array to match original dimensions
    dataset = reshape(normInput, dimensions)
    
    # Return min-max normalized array
    return dataset

end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
   
    minMaxValues = calculateMinMaxNormalizationParameters(dataset)
    
    return normalizeMinMax!(dataset, minMaxValues)
    
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    
    # Normalize between 0 and 1
    normInput = (dataset .- normalizationParameters[1])./(normalizationParameters[2] .- normalizationParameters[1])
    
    # Store dimensions of array
    dimensions = size(normInput)

    # Vectorize array for easy indexing
    normInput = vec(normInput)

    # Replace NaN values (product of division by 0 when max==min) by zeroes
    x=1
    for i in normInput
        if isequal(i,NaN)
            normInput[x] = 0
        end 
        x += 1 
    end

    # Reshape array to match original dimensions
    minMaxNorm = reshape(normInput, dimensions)
    
    # Return min-max normalized array
    return minMaxNorm

end

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    
    minMaxValues = calculateMinMaxNormalizationParameters(dataset)
    
    return normalizeMinMax!(dataset, minMaxValues)
    
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    
    # Normalize about zero-mean
    normInput = (dataset .- normalizationParameters[1])./normalizationParameters[2]
    
    # Store dimensions of array
    dimensions = size(normInput)

    # Vectorize array for easy indexing
    normInput = vec(normInput)

    # Replace NaN values (product of division by 0 when std=0) by zeroes
    x=1
    for i in normInput
        if isequal(i,NaN)
            normInput[x] = 0
        end 
        x += 1 
    end

    # Reshape array to match original dimensions
    dataset = reshape(normInput, dimensions)
    
    # Return zero-mean normalized array
    return dataset

end  

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    
    meanStdValues = calculateZeroMeanNormalizationParameters(dataset)
    
    return normalizeZeroMean!(dataset, meanStdValues)  

end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    
    # Normalize about zero-mean
    normInput = (dataset .- normalizationParameters[1])./normalizationParameters[2]
    
    # Store dimensions of array
    dimensions = size(normInput)

    # Vectorize array for easy indexing
    normInput = vec(normInput)

    # Replace NaN values (product of division by 0 when std=0) by zeroes
    x=1
    for i in normInput
        if isequal(i,NaN)
            normInput[x] = 0
        end 
        x += 1 
    end

    # Reshape array to match original dimensions
    meanStdNorm = reshape(normInput, dimensions)
    
    # Return zero-mean normalized array
    return meanStdNorm

end 

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    
    meanStdValues = calculateZeroMeanNormalizationParameters(dataset)
    
    return normalizeZeroMean!(dataset, meanStdValues)  

end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
    
    if size(outputs)[2] == 1
        
        outputs = outputs .>= threshold
        
        return outputs
        
    else
        
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        
        outputs = falses(size(outputs));
        
        outputs[indicesMaxEachInstance] .= true
        
        return collect(outputs)
        
    end
    
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    
    correctlyAssigned = targets .== outputs
    
    acc = sum(correctlyAssigned)/size(correctlyAssigned)[1]
    
    return acc
    
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    
    outputs = vec(outputs)
    targets = vec(targets)
    
    accuracy(outputs,targets)
    
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    
    clasifiedOutputs = threshold .<= outputs
    
    accuracy(clasifiedOutputs,targets)
    
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    
    outputs = classifyOutputs(outputs;threshold)
    
    classComparison = targets .== outputs
    correctClassifications = all(classComparison, dims=2)
    accuracy = mean(correctClassifications)
    
end

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    
    # Initialize ANN
    ann = Chain();
    
    # Initialize the number of imput layers to the next layer
    numInputsLayer = numInputs
    
    # Create hidden layers
    for numOutputsLayer in topology 
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) )      
        numInputsLayer = numOutputsLayer
    end
    
    # Create output layer
    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    
    # Return ANN model
    return ann
    
end

function trainClassANN(topology::AbstractArray{<:Int,1},      
                    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    
    # Define inputs, targets and classes
    inputs = dataset[1]
    targets = dataset[2]
    
    # Build the ANN
    ann = buildClassANN(size(inputs)[2],topology,size(targets)[2];transferFunctions)
    
    # Define missing parameters 
    ps = Flux.params(ann) 
    opt = ADAM(learningRate)
    loss(x, y) = Losses.crossentropy(ann(x), y)
    lossHistory = []
    
    # Training
    for epoch in 1:maxEpochs
        Flux.train!(loss, ps, [(inputs', targets')], opt)
        trainLoss = loss(inputs', targets')
        push!(lossHistory, trainLoss)
        # Break out of the training loop if the desired loss value is reached
        if trainLoss <= minLoss
            println("Epoch: $epoch , loss: $trainLoss")
            break
        end
        # Print status every 100 epochs
        if mod(epoch,100) == 0
            println("Epoch: $epoch , loss: $trainLoss")
        end
    end 
end

function trainClassANN(topology::AbstractArray{<:Int,1},      
                    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};      
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),      
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    
    # Build the ANN
    ann = buildClassANN(size(inputs)[2],topology,1;transferFunctions)
    
    # Define missing parameters 
    ps = Flux.params(ann) 
    opt = ADAM(learningRate)
    loss(x, y) = Losses.crossentropy(ann(x), y)
    lossHistory = []
    
    # Training
    for epoch in 1:maxEpochs
        Flux.train!(loss, ps, [(inputs', targets')], opt)
        trainLoss = loss(inputs', targets')
        push!(lossHistory, trainLoss)
        # Break out of the training loop if the desired loss value is reached
        if trainLoss <= minLoss
            println("Epoch: $epoch , loss: $trainLoss")
            break
        end
        # Print status every 100 epochs
        if mod(epoch,100) == 0
            println("Epoch: $epoch , loss: $trainLoss")
        end
    end
end

function holdOut(N::Int, P::Real)
    
    #Create an empty list of to store the indices
    indices = []
    
    #Fill list with indices up to N
    for i in range(start=1,step=1,stop=N)
        indices = append!(indices,i)
    end
    
    #Perform a random permutation of the indices 
    permIndices = indices[randperm(N)]
    
    #Identify splitting position
    splittingPos = Int64(round(N*P))
    
    #Divide indices for training and testing datasets (test indices, train indices)
    testTrainIndices = (permIndices[1:splittingPos], permIndices[splittingPos+1:length(permIndices)])
    
    return testTrainIndices
    
end

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    
    #Create an empty list of to store the indices
    indices = []
    
    #Fill list with indices up to N
    for i in range(start=1,step=1,stop=N)
        indices = append!(indices,i)
    end
    
    #Perform a random permutation of the indices 
    permIndices = indices[randperm(N)]
    
    #Identify first and seccond splitting positions
    splittingPos1 = Int64(round(N*Pval))
    splittingPos2 = splittingPos1 + Int64(round(N*Ptest))
    
    #Divide indices for training and testing datasets (val indices, test indices, train indices)
    valTestTrainIndices = (permIndices[1:splittingPos1], permIndices[splittingPos1+1:splittingPos2],
        permIndices[splittingPos2+1:length(permIndices)])
    
    return valTestTrainIndices
    
end

function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false) 
    
    # Define inputs and targets
    trainingInputs = trainingDataset[1]
    trainingTargets = trainingDataset[2]
    # If a validation set is provided, create validation variables
    if size(validationDataset[1]) != (0,0)
        validationInputs = validationDataset[1]
        validationTargets = validationDataset[2]
    end
    # If a test set is provided, create test variables
    if size(testDataset[1]) != (0,0)
        testInputs = testDataset[1]
        testTargets = testDataset[2]
    end
    
    # Build the ANN
    ann = buildClassANN(size(trainingInputs)[2],topology,size(trainingTargets)[2];transferFunctions)
    
    # Define missing parameters 
    ps = Flux.params(ann) 
    opt = ADAM(learningRate)
    lossHistoryTrain = []
    lossHistoryValidation = []
    lossHistoryTest = []
    cELoss(x, y) = Losses.crossentropy(ann(x), y)
    bCELoss(x, y) = Losses.binarycrossentropy(ann(x), y)
    
    # Initialize decreasing validation loss counter
    decresingValidation = 0
    
    # Initialize deepcopy variable (dc)
    # Deepcopy exists? dc=0 --> NO, dc=1--> YES
    dc = 0
    
    # Training and validation
    for epoch in 0:maxEpochs
        # Set binary cross entropy loss for binary classification, set cross entropy loss otherwise
        # Train
        if size(trainingTargets)[2] == 1 
            Flux.train!(bCELoss, ps, [(trainingInputs', trainingTargets')], opt)
            trainLoss = bCELoss(trainingInputs', trainingTargets')
        else
            Flux.train!(cELoss, ps, [(trainingInputs', trainingTargets')], opt)
            trainLoss = cELoss(trainingInputs', trainingTargets')
        end
        # Evaluate training loss
        push!(lossHistoryTrain, trainLoss)
        # Evaluate test set if one was provided
        if size(testDataset[1]) != (0,0)
            # Set binary cross entropy loss for binary classification, set cross entropy loss otherwise
            if size(testTargets[2]) == 0
                testLoss = bCELoss(testInputs', testTargets')
            else
                testLoss = cELoss(testInputs', testTargets')
            end
            push!(lossHistoryTest, testLoss)
        end
        # Evaluate validation loss if a validation set is provided
        if size(validationDataset[1]) != (0,0)
            # Set binary cross entropy loss for binary classification, set cross entropy loss otherwise
            if size(validationTargets[2]) == 0
                validationLoss = bCELoss(validationInputs', validationTargets')
            else
                validationLoss = cELoss(validationInputs', validationTargets')
            end
            push!(lossHistoryValidation, validationLoss)
            # Increase or re-initialize decreasing validation loss counter
            if epoch >= 2 && lossHistoryValidation[epoch] > lossHistoryValidation[epoch-1]
                decresingValidation += 1
            else
                decresingValidation = 0
            end
            # Break out of the training loop if the desired validation loss does not
            # improve in the specified amount of cicles 
            if decresingValidation >= maxEpochsVal
                if size(validationDataset[1]) != (0,0)
                    println("Epoch: $epoch , training loss: $trainLoss, validation loss: $validationLoss")
                    if size(testDataset[1]) != (0,0)
                        println("test loss: $testLoss")
                    end
                    println("Validation loss did not improve in the last $maxEpochsVal epochs")
                else
                    println("Epoch: $epoch , training loss: $trainLoss")
                    if size(testDataset[1]) != (0,0)
                        println("test loss: $testLoss")
                    end
                    println("Validation loss did not improve in the last $maxEpochsVal epochs")
                end
                break
            # Else, if the desired validation loss improved, create a deepcopy of the ann
            # and flip deepcopy variable to 1
            else
                savedAnn = deepcopy(ann)
                dc = 1
            end
        end
        # Break out of the training loop if the desired train loss value is reached
        if trainLoss <= minLoss
            if size(validationDataset[1]) != (0,0)
                println("Epoch: $epoch , training loss: $trainLoss, validation loss: $validationLoss")
                if size(testDataset[1]) != (0,0)
                    println("test loss: $testLoss")
                end
                println("Desired train loss value reached")
            else
                println("Epoch: $epoch , training loss: $trainLoss")
                if size(testDataset[1]) != (0,0)
                    println("test loss: $testLoss")
                end
                println("Desired train loss value reached")
            end
            break
        end
        # Print status every 100 epochs
        if mod(epoch,100) == 0 && showText == true
            if size(validationDataset[1]) != (0,0)
                println("Epoch: $epoch , training loss: $trainLoss, validation loss: $validationLoss")
                if size(testDataset[1]) != (0,0)
                    println("test loss: $testLoss")
                end
            else 
                println("Epoch: $epoch , training loss: $trainLoss")
                if size(testDataset[1]) != (0,0)
                    println("test loss: $testLoss")
                end
            end
        end
    end

    # To avoid overwriting the deepcopy if a break triggered, only make a deepcopy here if the
    # maximum dc has not been flipped to 1 (e.g. dc=0)
    if dc == 0
        savedAnn = deepcopy(ann)
    end
    
    # Return the saved ANN and loss histories
    return savedAnn, lossHistoryTrain, lossHistoryValidation, lossHistoryTest
    
end

function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)
    
    # Convert target vectors to 1x0 matrices
    trainingDataset[2] = reshape(trainingDataset[2],(size(trainingDataset[2]),0))
    validationDataset[2] = reshape(validationDataset[2],(size(validationDataset[2]),0))
    testDataset[2] = reshape(testDataset[2],(size(testDataset[2]),0))
    
    # Call the main trainClassANN function
    trainClassANN(topology,trainingDataset,validationDataset,testDataset,transferFunctions,maxEpochs, 
        minLoss,learningRate,maxEpochsVal,showText)
    
end

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    
    # Initialize variables
    confMatrix = [[0 0],[0 0]] # [[TN,FP][FN,TP]]
    
    # Get number ones (true values) and zeros (false values) in outputs vector
    onesOutputs = sum(outputs)
    zerosOutputs = size(outputs)[1] - onesOutputs
    
    # Compare outputs and targets
    comparison = outputs .== targets
    onesComparison = sum(comparison)
    zerosComparison = size(comparison)[1]-onesComparison
    
    #TP
    confMatrix[2][2] = sum(outputs.*targets)
    #TN
    confMatrix[1][1] = onesComparison - confMatrix[2][2]
    #FN
    confMatrix[2][1] = zerosOutputs - confMatrix[1][1]
    #FP
    confMatrix[1][2] = zerosComparison - confMatrix[2][1]
        
    # Calculate and return metrics 
    acc = (confMatrix[1][1] + confMatrix[2][2]) / size(outputs)[1] # Accuracy
    eR = (confMatrix[1][2] + confMatrix[2][1]) / size(outputs)[1] # Error rate
    sens = confMatrix[2][2] / (confMatrix[2][1] + confMatrix[2][2]) # Sensitivity 
    spec = confMatrix[1][1] / (confMatrix[1][2] + confMatrix[1][1]) # Specificity 
    posPredVal = confMatrix[2][2] / (confMatrix[2][2] + confMatrix[1][2]) # Positive predictive value
    negPredVal = confMatrix[1][1] / (confMatrix[1][1] + confMatrix[2][1]) # Negative predictive value
    fScore = (2 * sens * posPredVal) / (sens + posPredVal)
    
    # Exceptions
    # If every pattern is TN
    if confMatrix[1][1] == size(outputs)[1]
        sens = 1.0
        posPredVal = 1.0
    end
    # If every pattern is TP
    if confMatrix[2][2] == size(outputs)[1]
        spec = 1.0
        negPredVal = 1.0
    end
    # If a metric could not be calculated 
    if isequal(acc,NaN) #acc == NaN
        acc = 0.0
    elseif isequal(eR,NaN) #eR == NaN
        eR = 0.0
    elseif isequal(sens,NaN) #sens == NaN
        sens = 0.0
    elseif isequal(spec,NaN) #spec == NaN
        spec = 0.0
    elseif isequal(posPredVal,NaN) #posPredVal == NaN
        posPredVal = 0.0
    elseif isequal(negPredVal,NaN) #negPredVal == NaN
        negPredVal = 0.0
    elseif isequal(fScore,NaN) #fScore == NaN
        fScore = 0.0
    end
    # If sens and posPredVal equal 0
    if sens == 0 && posPredVal == 0
        fScore = 0.0
    end
    
    # Return metrics and filled confusion matrix
    return acc, eR, sens, spec, posPredVal, negPredVal, fScore, confMatrix
    
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    
    # Given the treshold, assign values in outputs vector as possitive (1) or negative (0)
    for i in 1:size(outputs)[1]
        if outputs[i] >= threshold
            outputs[i] = 1
        else
            outputs[i] = 0
        end
    end
    
    # Convert elements of outputs vector into boolean values
    outputs = Bool.(outputs)
    
    # Call main confusionMatrix function
    confusionMatrix(outputs,targets)
    
end 

function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    
    metricsAndConfMatrix = confusionMatrix(outputs,targets)
    
    display(metricsAndConfMatrix)
    
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    
    metricsAndConfMatrix = confusionMatrix(outputs,targets,threshold=threshold)
    
    display(metricsAndConfMatrix)
    
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
            
    # If only one column, call main confusionMatrix function
    if size(outputs)[2] == size(targets)[2] && size(outputs)[2] == 1
        return confusionMatrix(outputs,targets) 
        
    # If more than two columns...
    elseif size(outputs)[2] == size(targets)[2] && size(outputs)[2] > 2
        # Reserve memory for metrics
        sens=zeros(size(outputs)[2]);spec=zeros(size(outputs)[2]);
        posPredVal=zeros(size(outputs)[2]);negPredVal=zeros(size(outputs)[2]);
        fScore = zeros(size(outputs)[2]) 
        # Calculate and store the metrics for each category, in a one-against-all manner
        for i in 1:size(outputs)[2]
            metrics = confusionMatrix(outputs[:,i],targets[:,i])
            sens[i]=metrics[3];spec[i]=metrics[4];posPredVal[i]=metrics[5];
            negPredVal[i]=metrics[6];fScore[i]=metrics[7]
        end
        # Reserve memory for confusion matrix
        confMatrix = Int64.(zeros(size(outputs)[2],size(outputs)[2]))
        # Convert one-hot-encoded classes to numerical classes
        numericalClassesOutputs = Int64.(zeros(size(outputs)[1]))
        numericalClassesTargets = Int64.(zeros(size(outputs)[1]))
        for i in 1:size(outputs)[1]
            for j in 1:size(outputs)[2]
                if outputs[i,j] == true
                    numericalClassesOutputs[i] = j
                end
                if targets[i,j] == true
                    numericalClassesTargets[i] = j
                end
            end
        end
        # Fill confusion matrix
        for i in 1:size(numericalClassesOutputs)[1]
            confMatrix[numericalClassesOutputs[i],numericalClassesTargets[i]] += 1
        end
        # Weighted or macro
        # Weighted
        c = zeros(size(outputs)[2]) # Class intance counter
        if weighted == true
            for i in 1:size(outputs)[2]
                c[i] = sum(confMatrix[:,i]) # Count all instances of a class and store
            end
            w = c./sum(c) # Weights (percentages represented by each class)
            # Weighted metrics calculation
            sens=sum(sens.*w);spec=sum(spec.*w);posPredVal=sum(posPredVal.*w);
            negPredVal=sum(negPredVal.*w);fScore=sum(fScore.*w)
        # Macro
        else
            sens=mean(sens);spec=mean(spec);posPredVal=mean(posPredVal);
            negPredVal=mean(negPredVal);fScore=mean(fScore) #acc=mean(acc);eR=mean(eR);
        end
    # Return metrics and filled confusion matrix
    return sens, spec, posPredVal, negPredVal, fScore, confMatrix
        
    # If two columns, Return an error message
    else 
        return print("Invalid number of columns")
    end

end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
    boolOutputs = classifyOutputs(outputs)
    
    # Call appropriate confusionMatrix function
    return confusionMatrix(boolOutputs,targets,weighted=weighted)

end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    
    @assert(all([in(output, unique(targets)) for output in outputs]))
    
    # Obtain unique categories
    uniqueOutputCategories = unique(outputs)
    uniqueTargetCategories = unique(targets)
    uniqueCategories = unique(vcat(uniqueOutputCategories, uniqueTargetCategories))
    
    # Create vectors to contain numerical categories 
    numericalOutputs = Int64.(zeros(size(outputs)))
    numericalTargets = Int64.(zeros(size(outputs)))
    
    # Given the correspondance of an outputs and targets vectors element to the unique categories
    # assign a numerical version of the category to that element and store in numericalOutputs and
    # numericalTargets vectors
    for i in 1:size(outputs)[1]
        for j in 1:size(uniqueCategories)[1]
            if outputs[i] == uniqueCategories[j]
                numericalOutputs[i] = j
            end
            if targets[i] == uniqueCategories[j]
                numericalTargets[i] = j
            end
        end
    end
    
    # Concatenate numerical outputs and targets
    numericalOutputsAndTargets = vcat(numericalOutputs, numericalTargets)
    
    # One-hot encode the concatenated vector 
    oneHotOutputsAndTargets = oneHotEncoding(numericalOutputsAndTargets, unique(numericalOutputsAndTargets))

    # Separate one-hot encoded matrix into outputs and targets components
    oneHotOutputs = oneHotOutputsAndTargets[1:size(numericalOutputs)[1],:]
    oneHotTargets = oneHotOutputsAndTargets[size(numericalOutputs)[1]+1:size(oneHotOutputsAndTargets)[1],:]

    # Call appropriate confusionMatrix function
    return confusionMatrix(oneHotOutputs,oneHotTargets,weighted=weighted)
    
end 

function crossvalidation(N::Int64, k::Int64)
    
    # Create a vector using going from 1 to k
    subsets = collect(1:k)
    
    # Initialize a vector to contain several subsets
    subsetsRep = []
    
    # Fill the vector
    while size(subsetsRep)[1] < N
       append!(subsetsRep, subsets) 
    end
    
    # Prune the vector
    subsetsRep = subsetsRep[1:N]
    
    # Shuffle the vector 
    shuffle!(subsetsRep) 
    
    return subsetsRep
    
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    
    # Initialize indices vector
    indices = []
    
    for i in 1:size(targets)[2]
        classCount = sum(targets[:,i])
        append!(indices,crossvalidation(classCount,k))
    end
    
    return indices
    
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    
    oneHotEncodedTargets = oneHotEncoding(targets)
    
    return(crossvalidation(oneHotEncodedTargets,k))
    
end

function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})
    
    # Define global variables
    global categoricalOutput
    
    if size(classes)[1] == 2 # For 2 categories   
        # 1st category encoded as 0, and 2nd category encoded as 1
        encodedFeatures = feature .== classes[2]
        # Change type from BitVector to BitMatrix
        encodedFeatures = reshape(encodedFeatures,(size(Int.(encodedFeatures))[1],1))

    elseif size(classes)[1] > 2 # For more than 2 categories
        # Initialize counter
        c = 0
        for class in classes 
            if c == 0
                # Create a clasification vector containing the first class
                categoricalOutput = feature .== class
            else
                # Create a clasification vector for the curret class
                currentClass = feature .== class
                # Create a classification vector containing all classes
                append!(categoricalOutput, currentClass)
            end
            c += 1
        end
        # Define number of rows and columns 
        rows = Int(size(categoricalOutput)[1]/size(classes)[1])
        columns = size(classes)[1]
        encodedFeatures = reshape(categoricalOutput,(rows,columns)) 
    end
    
    # Return encoded features
    return encodedFeatures
    
end

function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices::Array{Int64,1}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    maxEpochsVal::Int=20)

    # Get the number of partitions
    k = maximum(kFoldIndices)
    
    # Initialize vectors to contain k partitions
    kSetsInputs=[[] for i=1:k]
    kSetsTargets = [[] for i=1:k]
    
    # Fill vectors with K matrices
    for i in 1:length(kFoldIndices)
        push!(kSetsInputs[kFoldIndices[i]],trainingDataset[1][i,:])
        push!(kSetsTargets[kFoldIndices[i]],trainingDataset[2][i,:])
    end
    
    # Create vector to store final losses for each training cycle
    trainLosses = []
    testLosses = []
    
    # Assign training and test sets, and train
    for i in 1:k
        # Assignation (deepcopy kSets to avoid permanent info lost when deleating)
        trainInputs = hcat(vcat(deleteat!(deepcopy(kSetsInputs),i)...)...)'  
        trainTargets = hcat(vcat(deleteat!(deepcopy(kSetsTargets),i)...)...)'
        testInputs = hcat(kSetsInputs[i]...)'
        testTargets = hcat(kSetsTargets[i]...)'
        # Create training and test tupples
        trainingDataset = (trainInputs,trainTargets)
        testDataset = (testInputs,testTargets)
        # Train
        resultsANN = trainClassANN(topology,trainingDataset,testDataset=testDataset,
            transferFunctions=transferFunctions,maxEpochs=maxEpochs,minLoss=minLoss,
            learningRate=learningRate,maxEpochsVal=maxEpochsVal)
        # Save final losses
        append!(trainLosses,last(resultsANN[2]))
        append!(testLosses,last(resultsANN[4]))
    end
      
    # Calculate mean losses
    meanTrainLoss = mean(trainLosses)
    stdDevTrainLoss = std(trainLosses)
    meanTestLoss = mean(testLosses)
    stdDevTestLoss = std(testLosses)
    
    # Return mean losses
    return meanTrainLoss, stdDevTrainLoss, meanTestLoss, stdDevTestLoss
    
end

function modelCrossValidation(modelType::Symbol,
        modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2},
        targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1})
    
    # Define global variables
    global model
    
    # Model selection and creation
    
    if modelType == :ANN # Artificial Neural Network
        
        encodedTargets = oneHotEncoding(targets)
        
        return trainClassANNacc(modelHyperparameters["topology"],(inputs,encodedTargets),crossValidationIndices,
                maxEpochs=modelHyperparameters["maxEpochs"],learningRate=modelHyperparameters["learningRate"],
                maxEpochsVal=modelHyperparameters["maxEpochsVal"])
        
    else
        
        if modelType == :SVM # Support Vector Machine
        
            #@sk_import svm: SVC
            model = SVC(kernel=modelHyperparameters["kernel"],degree=modelHyperparameters["degree"],
                gamma=modelHyperparameters["gamma"],C=modelHyperparameters["C"])
        
        elseif modelType == :DecisionTree # Desicion tree
        
            #@sk_import tree: DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"],
                random_state=modelHyperparameters["randomState"])
        
        elseif modelType == :kNN # k-Nearest Neighbours
        
            #@sk_import neighbours: KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=modelHyperparameters["nNeighbors"])
        
        end
        
        # Get the number of partitions
        k = maximum(kFoldIndices)
    
        # Initialize vectors to contain k partitions
        kSetsInputs=[[] for i=1:k]
        kSetsTargets = [[] for i=1:k]
    
        # Fill vectors with K matrices
        for i in 1:length(kFoldIndices)
            push!(kSetsInputs[kFoldIndices[i]],inputs[i,:])
            push!(kSetsTargets[kFoldIndices[i]],targets[i,:])
        end
        
        # Create vector to store final metrics for each training cycle
        trainMetrics = []
        testMetrics = []
    
        # Assign training and test sets, and train
        for i in 1:k
            # Assignation (deepcopy kSets to avoid permanent info loss when deleating)
            trainInputs = hcat(vcat(deleteat!(deepcopy(kSetsInputs),i)...)...)'  
            trainTargets = vcat(vcat(deleteat!(deepcopy(kSetsTargets),i)...)...)
            testInputs = hcat(kSetsInputs[i]...)'
            testTargets = vcat(kSetsTargets[i]...)
            
            # Train
            fit!(model,trainInputs,trainTargets)
            trainOutputs = predict(model,trainInputs)
            # Test
            testOutputs = predict(model,testInputs)
            # Calculate metrics for this iteration
            kTrainMetrics = confusionMatrix(trainOutputs,trainTargets)
            kTestMetrics = confusionMatrix(testOutputs,testTargets)
            # Save chosen metric (accuracy in this case)
            append!(trainMetrics,kTrainMetrics[1])
            append!(testMetrics,kTestMetrics[1])
        end
        
        # Calculate mean metrics and their standard deviation
        meanTrainMetric = mean(trainMetrics)
        stdDevTrainMetric = std(trainMetrics)
        meanTestMetric = mean(testMetrics)
        stdDevTestMetric = std(testMetrics)
    
        # Return mean metrics
        return meanTrainMetric, stdDevTrainMetric, meanTestMetric, stdDevTestMetric
        
    end
    
end

function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
        modelsHyperParameters::AbstractArray{Dict{String,Any},1},     
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
        kFoldIndices::Array{Int64,1})
    
    # Define global variables
    global model
    
    # Get the number of partitions
    k = maximum(kFoldIndices)
    
    # Initialize vectors to contain k partitions
    kSetsInputs=[[] for i=1:k]
    kSetsTargets = [[] for i=1:k]
    
    # Fill vectors with K matrices
    for i in 1:length(kFoldIndices)
        push!(kSetsInputs[kFoldIndices[i]],trainingDataset[1][i,:])
        push!(kSetsTargets[kFoldIndices[i]],trainingDataset[2][i])
    end
    
    # Create vector to store final test metrics for each training cycle
    testMetrics = []
        
    for i in 1:k
        
        # Assignation (deepcopy kSets to avoid permanent info lost when deleating)
        trainInputs = hcat(vcat(deleteat!(deepcopy(kSetsInputs),i)...)...)'
        trainTargets = vcat(vcat(deleteat!(deepcopy(kSetsTargets),i)...)...)
        testInputs = hcat(kSetsInputs[i]...)'
        testTargets = vcat(kSetsTargets[i]...)
        
        for j in 1:length(estimators)
            
            # Define current model type and hyperparameters
            modelType = estimators[j]
            modelHyperparameters = modelsHyperParameters[j]
            
            if modelType == :SVM # Support Vector Machine
                
                model = SVC(kernel=modelHyperparameters["kernel"],degree=modelHyperparameters["degree"],
                    gamma=modelHyperparameters["gamma"],C=modelHyperparameters["C"])
        
            elseif modelType == :DecisionTree # Desicion tree
        
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"],
                    random_state=modelHyperparameters["randomState"])
        
            elseif modelType == :kNN # k-Nearest Neighbours
        
                model = KNeighborsClassifier(n_neighbors=modelHyperparameters["nNeighbors"])
                
            end
            
            # Train
            fit!(model,trainInputs,trainTargets)
            # Test
            testOutputs = predict(model,testInputs)
            kTestMetrics = confusionMatrix(testOutputs,testTargets)
            # Save chosen metric (accuracy in this case)
            append!(testMetrics,kTestMetrics[1])
            
        end
            
    end
    
    # Calculate final metric according to selected ensemble scheme (majority voting in this case)
    meanTestMetric = mean(testMetrics)
    
    # Return mean metrics
    return meanTestMetric 
    
end

function trainClassEnsemble(baseEstimator::Symbol, 
        modelsHyperParameters::AbstractArray{Dict{String,Any},1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},     
        kFoldIndices::Array{Int64,1}; NumEstimators::Int=100)
    
    # Replicate
    estimators=[baseEstimator for i=1:NumEstimators]
    replicatedModelsHyperParameters=[modelsHyperParameters for i=1:NumEstimators]
    
    # Call main trainClassEnsemble
    trainClassEnsemble(estimators,replicatedModelsHyperParameters,trainingDataset,kFoldIndices)
    
end

function partition(dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; Ptest=0.0::Real, Pval::Real=0.0)
    
    # Get dataset size 
    n = size(dataset[1])[1]
    
    if Pval == 0.0 || Ptest == 0.0

        # Grab biggest P value
        Pmax = max(Pval,Ptest)
        
        # Create indices for split
        testTrainIndices = holdOut(n, Pmax)
        
        # Create sets 
        # As having a test/train split is more common than having a validation/train one, name conventions below 
        # use the name 'train' for the split, even if the actual split is of the validation/train kind
        
        #Inputs
        testInput = []
        for i in testTrainIndices[1]
            append!(testInput,dataset[1][i,:])
        end
        testInput = reshape(testInput,(size(dataset[1])[2]),Int64(size(testInput)[1]/size(dataset[1])[2]))'

        trainInput = []
        for i in testTrainIndices[2]
            append!(trainInput,dataset[1][i,:])
        end
        trainInput = reshape(trainInput,(size(dataset[1])[2]),Int64(size(trainInput)[1]/size(dataset[1])[2]))'

        # Targets        
        testTarget = []
        for i in testTrainIndices[1]
            append!(testTarget,dataset[2][i,:])
        end
        testTarget = reshape(testTarget,(size(dataset[2])[2],Int64(size(testTarget)[1]/size(dataset[2])[2])))'    

        trainTarget = []
        for i in testTrainIndices[2]
            append!(trainTarget,dataset[2][i,:])
        end
        trainTarget = reshape(trainTarget,(size(dataset[2])[2],Int64(size(trainTarget)[1]/size(dataset[2])[2])))' 
            
        # Convert to appropriate types
        testInput = convert(Matrix{Float32}, testInput)
        trainInput = convert(Matrix{Float32}, trainInput)
        testTarget = convert(BitMatrix, testTarget)
        trainTarget = convert(BitMatrix, trainTarget)
        
        # Create test and train tupples 
        testTrainTupples = [(testInput,testTarget),(trainInput,trainTarget)]
        
        if Pval == 0.0
            # Return test and train tupples 
            return testTrainTupples

        elseif Ptest == 0.0
            # Change name of train tupples to reflect the true nature of the split (validation/train)
            valTrainTupples = testTrainTupples
            # Return validation and train tupples
            return valTrainTupples
        
        end
        
    else
            
        # Create indices for split
        valTestTrainIndices = holdOut(n, Pval, Ptest)
        
        # Create sets
        
        #Inputs
        valInput = []
        for i in valTestTrainIndices[1]
            append!(valInput,dataset[1][i,:])
        end
        valInput = reshape(valInput,(size(dataset[1])[2]),Int64(size(valInput)[1]/size(dataset[1])[2]))'
        
        testInput = []
        for i in valTestTrainIndices[2]
            append!(testInput,dataset[1][i,:])
        end
        testInput = reshape(testInput,(size(dataset[1])[2]),Int64(size(testInput)[1]/size(dataset[1])[2]))'

        trainInput = []
        for i in valTestTrainIndices[3]
            append!(trainInput,dataset[1][i,:])
        end
        trainInput = reshape(trainInput,(size(dataset[1])[2]),Int64(size(trainInput)[1]/size(dataset[1])[2]))'

        # Targets
        valTarget = []
        for i in valTestTrainIndices[1]
            append!(valTarget,dataset[2][i,:])
        end
        valTarget = reshape(valTarget,(size(dataset[2])[2],Int64(size(valTarget)[1]/size(dataset[2])[2])))'  
        
        testTarget = []
        for i in valTestTrainIndices[2]
            append!(testTarget,dataset[2][i,:])
        end
        testTarget = reshape(testTarget,(size(dataset[2])[2],Int64(size(testTarget)[1]/size(dataset[2])[2])))'    

        trainTarget = []
        for i in valTestTrainIndices[3]
            append!(trainTarget,dataset[2][i,:])
        end
        trainTarget = reshape(trainTarget,(size(dataset[2])[2],Int64(size(trainTarget)[1]/size(dataset[2])[2])))' 
            
        # Convert to appropriate types
        valInput = convert(Matrix{Float32}, valInput)
        testInput = convert(Matrix{Float32}, testInput)
        trainInput = convert(Matrix{Float32}, trainInput)
        valTarget = convert(BitMatrix, valTarget)
        testTarget = convert(BitMatrix, testTarget)
        trainTarget = convert(BitMatrix, trainTarget)
        
        # Create validation, test and train tupples 
        valTestTrainTupples = [(valInput,valTarget),(testInput,testTarget),(trainInput,trainTarget)]
        
        # Return validation, test and train tupples 
        return valTestTrainTupples
        
    end
    
end

function trainClassANNacc(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices::Array{Int64,1}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    maxEpochsVal::Int=20)

    # Get the number of partitions
    k = maximum(kFoldIndices)
    
    # Initialize vectors to contain k partitions
    kSetsInputs=[[] for i=1:k]
    kSetsTargets = [[] for i=1:k]
    
    # Fill vectors with K matrices
    for i in 1:length(kFoldIndices)
        push!(kSetsInputs[kFoldIndices[i]],trainingDataset[1][i,:])
        push!(kSetsTargets[kFoldIndices[i]],trainingDataset[2][i,:])
    end
    
    trainMetrics = [] 
    testMetrics = [] 
    
    # Assign training and test sets, and train
    for i in 1:k
        # Assignation (deepcopy kSets to avoid permanent info lost when deleating)
        trainInputs = hcat(vcat(deleteat!(deepcopy(kSetsInputs),i)...)...)'  
        trainTargets = hcat(vcat(deleteat!(deepcopy(kSetsTargets),i)...)...)'
        testInputs = hcat(kSetsInputs[i]...)'
        testTargets = hcat(kSetsTargets[i]...)'
        # Create training and test tupples
        trainingDataset = (trainInputs,trainTargets)
        testDataset = (testInputs,testTargets)
        # Train
        resultsANN = trainClassANN(topology,trainingDataset,testDataset=testDataset,
            transferFunctions=transferFunctions,maxEpochs=maxEpochs,minLoss=minLoss,
            learningRate=learningRate,maxEpochsVal=maxEpochsVal)
        trainPredictions = resultsANN[1](trainInputs')'
        confMtrxTrain = confusionMatrix(trainPredictions,trainTargets,weighted=false)[end]
        testPredictions = resultsANN[1](testInputs')'
        confMtrxTest = confusionMatrix(testPredictions,testTargets,weighted=false)[end]
        diagTrain = []
        diagTest = []
        for i in 1:size(confMtrxTrain)[1]
            append!(diagTrain,confMtrxTrain[i,i])
            append!(diagTest,confMtrxTest[i,i])
        end
        kTrainMetrics = sum(diagTrain)/sum(confMtrxTrain)
        kTestMetrics = sum(diagTest)/sum(confMtrxTest)
        # Save chosen metric (accuracy in this case)
        append!(trainMetrics,kTrainMetrics)
        append!(testMetrics,kTestMetrics)

    end
    
    # Calculate mean metrics and their standard deviation
    meanTrainMetric = mean(trainMetrics)
    stdDevTrainMetric = std(trainMetrics)
    meanTestMetric = mean(testMetrics)
    stdDevTestMetric = std(testMetrics)
        
    # Return mean metrics
    return meanTrainMetric, stdDevTrainMetric, meanTestMetric, stdDevTestMetric
    
end

function multiclassToBinary(targets::AbstractArray{Bool,2},catNumber::Int64,catName::String)

    # Converts multicategory targets into binary targets (class of interest and "other")
    binaryTargets = []
    for i in 1:size(targets)[1]
        if targets[i,catNumber] == true
            push!(binaryTargets,catName)
        else
            push!(binaryTargets,"OTHER")
        end
    end 

    return binaryTargets

end

function multiclassToBinarySeveralCats(targets::AbstractArray{Bool,2},numberOfCats::Int64,catNames::Vector{String})

    # Creates several multicategory arrays for one-vs-all classification
    binaryTargetsArray = []
    for i in 1:numberOfCats
        binaryTarget = multiclassToBinary(targets,i,catNames[i])
        push!(binaryTargetsArray,binaryTarget)
    end

    return binaryTargetsArray

end