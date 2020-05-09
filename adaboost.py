from csv import reader
from operator import itemgetter
import math
import sys
from matplotlib import pyplot as plt

def clean_data(data):
    cleaned_data = []
    updated_data = []
    with open(data, 'r') as read_obj:
        csv_reader = reader(read_obj)
        next(csv_reader)
        for i in csv_reader:
            temp = []
            for j in i:
                temp.append(float(j))
            cleaned_data.append(temp)
    for j in range(len(cleaned_data[0])):
        result = []
        for n in range(len(cleaned_data)):
            if cleaned_data[n][-1] == 0:
                result.append([cleaned_data[n][j], -1, 1 / len(cleaned_data)])
            else:
                result.append([cleaned_data[n][j], 1, 1 / len(cleaned_data)])
        updated_data.append(result)
    return updated_data[:-1]


def decision_stump(matrix):
    F_star = float('inf')
    j_star = 0
    theta_star = 0
    j_column = -1

    for j in range(len(matrix)):
        F = 0
        for n in range(len(matrix[j])):
            if matrix[j][n][-2] == 1:
                F+=matrix[j][n][-1]
        sorted_x_val = sorted(matrix[j], key=itemgetter(0))
        if F<F_star:
            F_star=F
            theta_star = sorted_x_val[0][0]-1
            j_star = j
        for i in range(len(sorted_x_val)-1):
            F = F-(sorted_x_val[i][-2]*sorted_x_val[i][-1])
            if F < F_star and sorted_x_val[i][0]!= sorted_x_val[i+1][0]:
                F_star = F
                theta_star = (1/2)*(sorted_x_val[i][0] + sorted_x_val[i+1][0])
                j_star = j

    return j_star, theta_star

def adaboost_train(matrix, iters):
    hypothesis = []
    for t in range(iters):
        z_t = 0.0
        errors = 0.0
        col_num, dec_stump = decision_stump(matrix)
        #print(col_num, dec_stump)

        #finding errors
        for m in range(len(matrix[col_num])):
            if matrix[col_num][m][0] > dec_stump:
                decision = 1
            else:
                decision = -1
            if decision != matrix[col_num][m][-2]:
                errors += matrix[col_num][m][-1]
        weight_t = (1/2)*math.log((1/errors)-1)

        #calculation z_t
        for j in range(len(matrix[col_num])):
            if matrix[col_num][j][0] > dec_stump:
                decision = 1
            else:
                decision = -1
            z_t += (matrix[col_num][j][-1])*(math.exp(-weight_t*matrix[col_num][j][-2]*decision))

        #updating weights
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[col_num][j][0] > dec_stump:
                    decision = 1.0
                    matrix[i][j][-1] = ((matrix[i][j][-1]) * (math.exp(-weight_t * matrix[i][j][-2] * decision))) / z_t
                else:
                    decision = -1.0
                    matrix[i][j][-1] = ((matrix[i][j][-1]) * (math.exp(-weight_t * matrix[i][j][-2] * decision))) / z_t
        hypothesis.append([col_num, weight_t, dec_stump])
    return hypothesis

def adaboost_test(matrix, hyp):
    errors_test = 0
    for i in range(len(matrix[0])):
        prediction = 0.0
        actual = 0
        for j in range(len(matrix)):
            val = matrix[j][i][0]
            actual = matrix[j][i][-2]
            for k in hyp:
                if k[0] == j:
                    if val > k[2]:
                        prediction += k[1]*1
                    else:
                        prediction += k[1]*(-1)
        if prediction>0:
            prediction = 1
        else:
            prediction = -1
        if prediction != actual:
            errors_test+=1

    return ((errors_test / (len(matrix[0]))))

def run_cross_validation(iters):
    mismatch_vec = []
    dataset_train = sys.argv[2]
    data = clean_data(dataset_train)
    for i in range(10):
        training_data_temp = []
        training_data = []
        test_data_temp = []
        test_data = []
        print("\n================Iteration: " , i, "==================")
        for j in range(len(data)):
            count = 0
            for k in range(len(data[0])):
                if count< round(len(data[0])/10):
                    test_data_temp.append(data[j][k])
                    count+=1
                else:
                    training_data_temp.append(data[j][k])
            training_data.append(training_data_temp)
            test_data.append(test_data_temp)
            training_data_temp = []
            test_data_temp = []
        hyp_train = adaboost_train(training_data, iters)
        mismatch = adaboost_test(test_data, hyp_train)
        mismatch_vec.append(mismatch)
        print("Output Hypothesis (Column Number, weight of stump, decision stump):\n", hyp_train)
        data = []

        for j in range(len(test_data)):
            data.append(training_data[j]+test_data[j])
    print("\nPrediction-Error:", mismatch_vec)
    sum = 0
    for i in mismatch_vec:
        sum+=i
    print("\nMean-Error:", sum/len(mismatch_vec))
    return sum/len(mismatch_vec)

def plot_graph():
    mismatch_erm = []
    mismatch_cv = []
    x_axis = []
    for i in range(1,25):
        x_axis.append(i*5)
        dataset_train = sys.argv[2]
        matrix_train = clean_data(dataset_train)
        hyp_train = adaboost_train(matrix_train, iters=i*5)
        mismatch = adaboost_test(matrix_train, hyp_train)
        mismatch_erm.append(mismatch)

        mismatch_mean = run_cross_validation(i*5)
        mismatch_cv.append(mismatch_mean)


    print("Mismatch's for ERM using T = 5, 10, 15 ...\n",mismatch_erm)
    print("Mismatch's for Cross Validation using T = 5, 10, 15 ...\n",mismatch_cv)

    plt.plot(x_axis, mismatch_erm, label='ERM')
    plt.plot(x_axis, mismatch_cv, label='Cross-Validation')
    plt.xlabel("T Value")
    plt.ylabel("Emperical Risk")
    plt.legend()
    plt.show()

def main():
    if sys.argv[1] == '--dataset' and sys.argv[3] == '--mode' and sys.argv[4] == 'erm':
        dataset_train = sys.argv[2]
        matrix_train = clean_data(dataset_train)
        hyp_train = adaboost_train(matrix_train, iters = 100)
        mismatch = adaboost_test(matrix_train, hyp_train)
        print("\nPrediction-Error:",mismatch)
        print("\nOutput Hypothesis (Column Number, weight of stump, decision stump):\n", hyp_train)

    if sys.argv[1] == '--dataset' and sys.argv[3] == '--mode' and sys.argv[4] == '10-fold':
        run_cross_validation(59)

    if sys.argv[1] == '--dataset' and sys.argv[3] == '--mode' and sys.argv[4] == 'plot':
        plot_graph()


if __name__  == '__main__':
    main()