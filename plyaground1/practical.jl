using DataFrames
using CSV


dataset = CSV.read("../datasets/student-mat.csv")
println(size(dataset))
println(dataset)

# binary classification