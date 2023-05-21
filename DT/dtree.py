import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if self is LeafNode:
            return self.predict(x_test)
        col = self.col
        split = self.split

        if x_test[col] < split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        return self.prediction


def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    n = len(x)
    (cates, counts) = np.unique(x, return_counts=True)
    sum = 0
    for i in range(len(cates)):
        sum += (counts[i] / n) ** 2
    return 1 - sum


def find_best_split(X, y, loss, min_samples_leaf):
    best = {'feature': -1, 'split': -1, 'loss': loss(y)}  # set the initial value for our best split, will iterate to\
    # update in the following code.
    k = 11

    for feature_i in range(len(X[0])):  # The index of features
        if len(X) > 11:
            candidates = np.random.choice(X[:, feature_i], size=k, replace=False)  # can be improved
        else:
            candidates = X[:, feature_i]
        for split in candidates:  # split is the feature split value in X
            judgel = X[:, feature_i] < split  # for feature i, select all row number that are less than split
            judger = X[:, feature_i] >= split

            # yl = X[judgel, :]
            # yr = X[judger, :]
            yl = y[judgel]
            yr = y[judger]

            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:  # if one of the split set is too small, pass
                # this iteration
                continue

            weighted_loss = (len(yl) * loss(yl) + len(yr) * loss(yr)) / (len(yl) + len(yr))
            if weighted_loss == 0:
                return feature_i, split
            if weighted_loss < best['loss']:
                best = {'feature': feature_i, 'split': split, 'loss': weighted_loss}
    return best['feature'], best['split']


class DecisionTree621:
    """
    This is the core code for this dtree implementation
    fit function invoke fit_ function to recursive creating the node for dtree
    predict function invoke the prediction method of DecisionNode or LeafNode to give a prediction, i.e.
    the self.prediction attribute assigned to LeafNode when created.
    """
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss  # loss function; either np.var for regression or gini for classification

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) < self.min_samples_leaf or len(np.unique(X)) == 1:
            return self.create_leaf(y)
        col, split = find_best_split(X, y, self.loss, self.min_samples_leaf)
        if col == -1:  # meaning the find_best_split can split the decision node anymore
            return self.create_leaf(y)
        judgel = X[:, col] < split  # for col, select all row number that are less than split
        judger = X[:, col] >= split
        lchild = self.fit_(X[judgel], y[judgel])
        rchild = self.fit_(X[judger], y[judger])
        dnode = DecisionNode(col, split, lchild, rchild)
        return dnode

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        pred = []
        for X_i in X_test:
            pred.append(self.root.predict(X_i))
        return np.array(pred)


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.var)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        pred = self.predict(X_test)
        sse = np.sum(np.square(pred-y_test))
        sst = np.sum(np.square(np.mean(y_test)-y_test))
        return 1 - sse / sst

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        pred = self.predict(X_test)
        count = 0
        for i in range(len(pred)):  # count the same value pairwise between test and pred
            if pred[i] == y_test[i]:
                count += 1
        return count/len(pred)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y, stats.mode(y).mode.item())
