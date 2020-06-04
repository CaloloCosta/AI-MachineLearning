using DataFrames
using CSV
using Statistics
using StatsBase # for data tranformation


df = CSV.read("../datasets/student-mat.csv")
df = convert(Matrix,df)
#println(size(df))
# println(first(dataset,10))
# println(dataset.G1[1])


# standardization of the values
function standardization(x)
    dt = StatsBase.fit(ZScoreTransform,x,dims=2)
    return StatsBase.transform(dt,x)
end


# maybe modify this function in the future...
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

        # 5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
        if(dataset[i,5] == "LE3")
            dataset[i,5] = 1
        elseif(dataset[i,5] == "GT3")
            dataset[i,5] = 0
        end

        # 6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
        if(dataset[i,6] == "T")
            dataset[i,6] = 1
        elseif(dataset[i,6] == "A")
            dataset[i,6] = 0
        end

        # 9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
        dataset[i,9] = job(dataset[i,9])

        # 10 Fjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
        dataset[i,10] = job(dataset[i,10])

        # 11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
        if(dataset[i,11] == "home")
            dataset[i,11] = 1
        elseif(dataset[i,11] == "reputation")
            dataset[i,11] = 2
        elseif(dataset[i,11] == "course")
            dataset[i,11] = 3
        elseif(dataset[i,11] == "other")
            dataset[i,11] = 4
        end
 
        # get the grade mean
        if(mean(grades[i,:]) >= 10)
            pass[i] = 1
        else
            pass[i] = 0
        end

        # 12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
        if(dataset[i,12] == "mother")
            dataset[i,12] = 1
        elseif(dataset[i,12] == "father")
            dataset[i,12] = 2
        else
            dataset[i,12] = 3
        end

        # 16 schoolsup - extra educational support (binary: yes or no)
        if(dataset[i,16] == "yes")
            dataset[i,16] = 1
        else
            dataset[i,16] = 0
        end

        # 17 famsup - family educational support (binary: yes or no)
        if(dataset[i,17] == "yes")
            dataset[i,17] = 1
        else
            dataset[i,17] = 0
        end

        # 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
        if(dataset[i,18] == "yes")
            dataset[i,18] = 1
        else
            dataset[i,18] = 0
        end

        # 19 activities - extra-curricular activities (binary: yes or no)
        if(dataset[i,19] == "yes")
            dataset[i,19] = 1
        else
            dataset[i,19] = 0
        end

        # 20 nursery - attended nursery school (binary: yes or no)
        if(dataset[i,20] == "yes")
            dataset[i,20] = 1
        else
            dataset[i,20] = 0
        end

        #  21 higher - wants to take higher education (binary: yes or no)
        if(dataset[i,21] == "yes")
            dataset[i,21] = 1
        else
            dataset[i,21] = 0
        end
        # 22 nternet - Internet access at home (binary: yes or no)
        if(dataset[i,22] == "yes")
            dataset[i,22] = 1
        else
            dataset[i,22] = 0
        end

        # 23 romantic - with a romantic relationship (binary: yes or no)
        if(dataset[i,23] == "yes")
            dataset[i,23] = 1
        else
            dataset[i,23] = 0
        end
    end

    # convert from array of any to array of Float64
    x = convert(Array{Float64},dataset[:,1:30])
    return (x,pass)
end


# hypothesis function
function hypothesis(x, theta)
    z = theta' * x
    return 1 / (1 + exp(-z))
end

# gradient descent
function gradientDescent(x,y,theta,m,n,rp,lr)
    c = 0
    s = 0
    for i in 1:m
        c = -y[i]*log(hypothesis(x[i,:],theta)) - (1 - y[i])*log(1 - hypothesis(x[i,:],theta))
        for j in 2:n
            s += (theta[j])^2
        end
        s *= rp/(2*m)
        c *= 1/m
        c += s
        theta = theta - lr*(1/m*x[i,:]*(hypothesis(x[i,:],theta)-y[i])+((rp/m)*theta))
        #println("theata: ",theta)
        #println("cost: ",c)
    end

    return theta
end



# utils...
function job(data)
    if(data == "teacher")
        data = 1
    elseif(data == "health")
        data = 2
    elseif(data == "services")
        data = 3
    elseif(data == "at_home")
        data = 4
    elseif(data == "other")
        data = 5
    return data
    end 
end

cd = cleanData(df)

x = cd[1]
x = standardization(x)
y = cd[2]
oneMatrix = ones(size(x)[1])
x = hcat(x,oneMatrix)
theta = zeros(size(x)[2])

# 80% for training 20% for testing
i = trunc(Int,(size(x)[1]) * 0.8)
# training data
x_train = x[1:i,:]
y_train = y[1:i,:]

# testing data
x_test = x[i+1:size(x)[1],:]
y_test = y[i+1:size(x)[1],:]

# println("x size: ",size(x_test),"\nTheta size: ",size(theta))
# println("Features: ",features)
# println("Outcome: ",y)

println("mean output: ",mean(y_train))
# gradient descent args desc: x,y,theta,size of the dataset, size of the features, rp, learning rate
t = gradientDescent(x_train,y_train,theta,size(x_train)[1],size(x_train)[2],0.3,0.001)

function predict(x,theta,y)
    for i in 1:size(x)[1]
        println("Predicted: ",hypothesis(x[i,:],theta)," Expected: ",y[i])
    end
end

predict(x_test,t,y_train)


#println(-y[1]*log(hypothesis(x[1,:],theta)) - (1 - y[1])*log(1 - hypothesis(x[1,:],theta)))

#    println(theta - 0.01*(1/10*x[1,:]*(hypothesis(x[1,:],theta)-y[1])+(0.3/10)*theta))
#    println()






