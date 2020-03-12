using DataFrames
using CSV
using Statistics


dataset = CSV.read("../datasets/student-mat.csv")
dataset = convert(Matrix,dataset)
println(size(dataset))
# println(first(dataset,10))
# println(dataset.G1[1])


function cleanData(dataset)
    # get the grades and create the pass array
    grades = dataset[:,31:33]
    pass = Array{Int64}(undef,size(dataset)[1],1)

    # loop for cleaning the data
    for i in 1:size(dataset)[1]

        # school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
        if(dataset[i,1] == "GP")
            dataset[i,1] = 1
        elseif(dataset[i,1] == "MS")
            dataset[i,1] = 0
        end

        # sex - student's sex (binary: 'F' - female or 'M' - male)
        if(dataset[i,2] == "M")
            dataset[i,2] = 1
        elseif(dataset[i,2] == "F")
            dataset[i,2] = 0
        end

        # 4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
        if(dataset[i,4] == "U")
            dataset[i,4] = 1
        elseif(dataset[i,4] == "R")
            dataset[i,4] = 0
        end

        
        # get the grade mean
        if(mean(grades[i,:]) >= 10)
            pass[i] = 1
        else
            pass[i] = 0
        end
    end

    #println(head(dataset,10))
end


cleanData(dataset)



