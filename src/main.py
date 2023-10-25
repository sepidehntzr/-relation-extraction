import argparse  # For positional command-line arguments
import pandas as pd
import numpy as np
import csv
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def create_argument_parser():
    """ Create a parser and add 2 arguments to it
    --train for the path to a training corpus
    --test for the path to a test corpus
    --output for the output file
    Returns:
        args: stores all args
    """
    parser = argparse.ArgumentParser(
        "Correctness of grammar of a sentence")
    parser.add_argument("--train", help="Path to the train file")
    parser.add_argument("--test", help="Path to the test file")
    parser.add_argument("--output", help="Path to the output file")

    args = parser.parse_args()
    return args


def remove_stopwords(df):

    stop_words = stopwords.words('english')
    stop_words.append("\"")
    stop_words.append(".")
    stop_words.append(",")
    stop_words.append("-")
    for index, row in df.iterrows():
        tokens = row['tokens']
        new_tokens = []

        for token in tokens:
            if token not in stop_words:
                new_tokens.append(token)
        row['tokens'] = new_tokens
    return df


def naive_bayes(train_data):
    count = {}
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    vocab = []
    classes_prob = {}
    prior = {}
    log_likelihood = {}
    train_data = remove_stopwords(train_data)

    for index, row in train_data.iterrows():
        c = row['relation']
        # if c not in word_class_count.keys():
        #     word_class_count[c][] = 1
        # else:
        #      word_class_count[c] += 1
        if c not in classes_prob.keys():
            classes_prob[c] = 1
        else:
            classes_prob[c] += 1

        tokens = row['tokens'].split()
        for token in tokens:
            if c not in count.keys():
                count[c] = {}
                count[c][token] = 1
                vocab.append(token)
            else:
                if token not in count[c].keys():
                    count[c][token] = 1
                else:
                    count[c][token] += 1
    for c in count.keys():
        for token in count[c].keys():
            if c not in log_likelihood.keys():
                log_likelihood[c] = {}
            log_likelihood[c][token] = np.log(
                count[c][token]+1 / (sum(count[c].values())+len(vocab)))

    for c in classes_prob.keys():
        prior[c] = np.log(classes_prob[c]/sum(classes_prob.values()))
    return prior, log_likelihood


def three_fold_cross_validation(train_data):
    shuffle_data = train_data.sample(frac=1)

    n = len(train_data)
    index = int(n/3)
    fold_1 = shuffle_data.iloc[0:index]
    fold_2 = shuffle_data.iloc[index:2*index]
    fold_3 = shuffle_data.iloc[2*index:]
    folds = [fold_1, fold_2, fold_3]
    return folds


def test_naive_bayes(test_data, priors, log_likelihood):
    output = []
    correct_count = 0
    test_data = remove_stopwords(test_data)
    for index, row in test_data.iterrows():

        original_label = row['relation']
        tokens = row['tokens'].split()
        best_prob = float('-inf')
        output_label = ''

        for c in priors.keys():

            prob = priors[c]
            for token in tokens:
                if token in log_likelihood[c].keys():
                    prob += log_likelihood[c][token]
            if prob > best_prob:
                output_label = c
                best_prob = prob
        if original_label == output_label:
            correct_count += 1
        output.append([original_label, output_label, index])
    acc = correct_count/len(output)
    return acc, output


def write_to_csv(data, path):

    f = open(path, "w")
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
    f.close()


def read_file(path):
    df = pd.read_csv(path, header=0)
    return df


if __name__ == '__main__':
    args = create_argument_parser()
    train_data = read_file(args.train)
    test_data = read_file(args.test)
    folds = three_fold_cross_validation(train_data)
    avg_acc = 0
    best_accuracy = 0
    best_log_likelihood = []
    best_log_prior = []
    for i in range(len(folds)):
        if i == 0:
            dev_data = folds[0]
            frames = [folds[1], folds[2]]
            new_train_data = pd.concat(frames)
        elif i == 1:
            dev_data = folds[1]
            frames = [folds[0], folds[2]]
            new_train_data = pd.concat(frames)
        else:
            dev_data = folds[2]
            frames = [folds[0], folds[1]]
            new_train_data = pd.concat(frames)

        log_priors, log_likelihood = naive_bayes(new_train_data)
        accuracy, output = test_naive_bayes(
            dev_data, log_priors, log_likelihood)
        avg_acc += accuracy
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_log_likelihood = log_likelihood
            best_log_prior = log_priors
    avg_acc = avg_acc/3
    print("Average accuracy in 3 fold cross validation: ", avg_acc)

    accuracy, output = test_naive_bayes(
        test_data, best_log_prior, best_log_likelihood)
    print("Accuracy on test set: ", accuracy)

    write_to_csv(output, args.output)
    #test_data = read_file(args.test)
