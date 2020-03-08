using DataFrames
using CSV
using StatsBase # for data tranformation

# so all the v
function standardization(x)
    dt = StatsBase.fit(ZScoreTransform,x,dims=2)
    return StatsBase.transform(dt,x)
end


function gradientDescent(x,y,theta,m,lr)
    nIterations = 5
    for i in nIterations
        for j in size(theta)[1]
            gradient = 0
            for l in m
                gradient += (hypothesis(x[l],theta) - y[j])*x[l][j]
            end
            gradient *= 2/m
            theta[j] = theta[j] - (lr*gradient)
        end
    end
    println(theta)
end


# prepare for calculations
function prepare(x, y)
    # append the one vector into the x_text
    oneMatrix = ones(size(x)[1])
    x = hcat(oneMatrix,x)
    # create the theta matrix and initialize them to zeros...
    theta = zeros(size(x)[2])
    gradientDescent(x,y,theta,size(x)[1],0.1)
end



# hypothesis function
function hypothesis(x,theta)
    return x * theta
end



# find the gradient decest
dataset = CSV.read("../datasets/real-estate-kaggle.csv")
# describe the dataset
#println(describe(dataset))
# println("it works...")
# println("Size: ",size(dataset))
#println(dataset)

# call the features x, get all rows from column 3 to 7 
features = dataset[:,3:7]
#println(features)
# convert dataframe to a matrix
x = convert(Matrix,features)
# call the outcome y, get all rows for column 8
y = dataset[:,8] # becomes a vector

# take 80% for training
i = trunc(Int,(size(x)[1]) * 0.8)
x_train = standardization(x[1:i,:])
x_test = standardization(x[i+1:414,:])


#println(x_test)

y_train = y[1:i,:]
y_test = y[i+1:414,:]

prepare(x_train,y_train)

# # gradient descent
# function gradientDescent(x, y, theta, m, lr)
#     iterations = 10
#     for i in iterations
#         for j in size(theta)[1]
#             gradient = 0
#             for i in m
#                 gradient += ((hypothesis(x[i],theta) - y[i]) * x[i][j])
#             end
#         gradient *= 1/m
#         theta[j] = theta[j] - (lr * gradient)
#         println(theta,"\n\n")
#         end
#     end
#     return theta
    
# end


# gradientDescent(x_train,y_train,theta,size(y_train)[1],0.001)
# println("size of x_train", trunc(Int,(size(x)[1]) * 0.8))