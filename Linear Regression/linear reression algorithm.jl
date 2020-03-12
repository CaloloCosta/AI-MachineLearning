using DataFrames
using CSV
using StatsBase # for data tranformation

# standardization of the values
function standardization(x)
    dt = StatsBase.fit(ZScoreTransform,x,dims=2)
    return StatsBase.transform(dt,x)
end

# predict function
function predict(x_test,y_test,theta)
    for i in 1:size(x_test)[1]
        println("Predicted: ",hypothesis(x_test[i,:],theta)," Expected: ",y_test[i])
    end
end

# gradient descent method
function gradientDescent(x,y,theta,m,lr)
    nIterations = 70
    
    println("training...")
    for i in 1:nIterations
        cost = 0
        for j in 1:size(theta)[1]
            gradient = 0
            for l in 1:m
                gradient += (hypothesis(x[l,:],theta) - y[j])*x[l,j]
                cost += (hypothesis(x[l,:],theta) - y[j])^2
            end
            gradient *= 2/m
            theta[j] = theta[j] - (lr*gradient)
        end
        cost *= 1/m
        println(i,"-cost: ",cost)
        # println(theta)
    end

    println("\n\n\nTetha: ",theta)
    println("testing...")
    # predict
    predict(x_test,y_test,theta)
end

# prepare for calculations
function prepare(x, y)
    # append the one vector into the x_text
    oneMatrix = ones(size(x)[1])
    x = hcat(oneMatrix,x)

    # create the theta matrix and initialize them to zeros...
    theta = zeros(size(x)[2])
    gradientDescent(x,y,theta,size(x)[1],0.015) # 0.002 = 415
end



# hypothesis function
function hypothesis(x,theta)
    return transpose(theta) * x
end


dataset = CSV.read("../datasets/real-estate-kaggle.csv")
# describe the dataset
#println(describe(dataset))
#println(first(dataset,50))


# call the features x, get all rows from column 3 to 7 (inclusive)
# features = dataset[:,3:7]

# call the features x, get all rows from column 3 to 5 (inclusive)
features = dataset[:,3:5]


# convert dataframe to a matrix
x = convert(Matrix,features)

# call the outcome y, get all rows for column 8
y = dataset[:,8] # becomes a vector

# take 80% for training
i = trunc(Int,(size(x)[1]) * 0.8)
x_train = standardization(x[1:i,:])
x_test = standardization(x[i+1:414,:])
# adding ones to x_test
x_test = hcat(ones(size(x_test)[1]),x_test)

# remaining 20 % for testing
y_train = y[1:i,:]
y_test = y[i+1:414,:]

prepare(x_train,y_train)