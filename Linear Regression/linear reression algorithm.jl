using DataFrames
using CSV


# find the gradient decest
dataset = CSV.read("../datasets/real-estate-kaggle.csv")
println("it works...")
println("Size: ",size(dataset))

# call the features x, get all rows from column 3 to 7 
features = dataset[:,3:7]
# convert dataframe to a matrix
x = convert(Matrix,features)
# call the outcome y, get all rows for column 8
y = dataset[:,8] # becomes a vector

# take 80% for training
i = trunc(Int,(size(x)[1]) * 0.8)
x_train = x[1:i,:]
x_test = x[i+1:414,:]

y_train = y[1:i,:]
y_test = y[i+1:414,:]

# println("size of x_train", trunc(Int,(size(x)[1]) * 0.8))