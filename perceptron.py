from csv import reader
import sys
import decimal

def clean_data(data, bias):
    cleaned_data = []
    with open(data, 'r') as read_obj:
        csv_reader = reader(read_obj)
        next(csv_reader)
        for i in csv_reader:
            temp = [bias]
            for j in i:
                temp.append(float(j))
            cleaned_data.append(temp)
    return cleaned_data

def initialise_weights(matrix):
    weight_vector = [0.0] * (len(matrix[1]) - 1)
    return weight_vector

def percetron_train(matrix, lr, bias, max_iter):
    max_iterations = 0
    weight_vector = initialise_weights(matrix)
    valid = 0
    error = len(matrix)
    for x in range(max_iter):
        if error - valid != 0:
            valid = 0
            for i in range(len(matrix)):
                prediction_y = 0.0
                y_val = float(matrix[i][-1])
                if y_val == 0:
                    y_val =-1
                x_val = matrix[i][:-1]
                for j in range(len(x_val)):
                    prediction_y += weight_vector[j] * x_val[j]
                if y_val*(prediction_y)<=0:
                    for j in range(len(x_val)):
                        weight_vector[j] = weight_vector[j] + lr * (y_val) * x_val[j]
                else:
                    valid+=1
        else:
            break
    print("Weight Vector: ", weight_vector)
    return weight_vector

def perceptron_test(matrix, w):
    prediction = list()
    mismatch = 0.0
    match = 0.0

    for i in range(len(matrix)):
        x_val = matrix[i][:-1]
        y_val = 0.0
        for j in range(len(matrix[i]) - 1):
            y_val += w[j] * x_val[j]
        if (y_val)*matrix[i][-1] <= 0:
            prediction.append(0)
        else:
            prediction.append(1)
    for i in range(len(prediction)):
        if prediction[i] != matrix[i][-1]:
            mismatch += 1.0
        else:
            match+=1.0

    return float((mismatch/(match + mismatch))*100.00)


def run_cross_validation():
    sum = 0
    mismatch_vec = []
    dataset_train = sys.argv[2]
    data = clean_data(dataset_train, bias = 1)
    for i in range(10):
        print("================Iteration: " , i, "==================")
        test_dataset = data[0:round(len(data) / 10)]
        train_dataset = data[round(len(data) / 10):]

        weight_vec = percetron_train(train_dataset, lr=1, bias=1, max_iter=15000)
        mismatach = perceptron_test(test_dataset, weight_vec)
        mismatch_vec.append(mismatach)
        data = train_dataset + test_dataset
    for i in mismatch_vec:
        sum+=i
    print("Prediction Error",mismatch_vec)
    print("Mean-error", sum / len(mismatch_vec))

def main():
    if sys.argv[1] == '--dataset' and sys.argv[3] == '--mode' and sys.argv[4] == 'erm':
        dataset_train = sys.argv[2]
        dataset_test = sys.argv[2]
        matrix_train = clean_data(dataset_train, bias = 1)
        matrix_test = clean_data(dataset_test, bias = 1)
        weight_vec = percetron_train(matrix_train, lr=1, bias=1, max_iter=15000)
        mismatch = perceptron_test(matrix_test, weight_vec)
        print("Prediction Error:", mismatch)

    if sys.argv[1] == '--dataset' and sys.argv[3] == '--mode' and sys.argv[4] == '10-fold':
        run_cross_validation()

if __name__ == '__main__':
    main()
