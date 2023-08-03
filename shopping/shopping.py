import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month = {}
    for i in range(0, len(months)):
        month[months[i]] = i+1
    print(month)
    with open(filename, "r") as data_file:
        reader = csv.DictReader(data_file)
        for each_row in reader:
            parsed_row = [int(each_row["Administrative"]), float(each_row["Administrative_Duration"]),
                          int(each_row["Informational"]), float(each_row["Informational_Duration"]),
                          int(each_row["ProductRelated"]), float(each_row["ProductRelated_Duration"]),
                          float(each_row["BounceRates"]), float(each_row["ExitRates"]), float(each_row["PageValues"]),
                          float(each_row["SpecialDay"]), int(month[each_row["Month"]]),
                          int(each_row["OperatingSystems"]),
                          int(each_row["Browser"]), int(each_row["Region"]), int(each_row["TrafficType"]),
                          int(each_row["VisitorType"] == 'Returning_Visitor'), int(each_row["Weekend"] == 'TRUE')]
            evidence.append(parsed_row)

            # append the corresponding label
            labels.append(int(each_row["Revenue"] == 'TRUE'))

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # total test set size == total size of rows in the dataset
    size = len(labels)
    # total negative examples or evtns when sale didn't happen
    negatives = 0
    # total positive examples when sale happened
    positives = 0
    # totale num of positives identified as positives
    true_positives = 0
    # total num of negatives identified as negatives
    true_negatives = 0

    for i in range(size):

        if labels[i] == 0:
            negatives += 1
            if labels[i] == predictions[i]:
                true_negatives += 1
        else:
            positives += 1
            if labels[i] == predictions[i]:
                true_positives += 1

    # True Negative Rate
    specificity = true_negatives / negatives

    # True Positive Rate
    sensitivity = true_positives / positives



    return sensitivity, specificity


if __name__ == "__main__":
    main()
