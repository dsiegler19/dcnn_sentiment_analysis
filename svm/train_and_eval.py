# Support Vector Machines are a very common machine learning algorithm. An SVM attempts to find the best separating
# hyperplane (since an SVM works in multidimensional, it is a hyperplane, in 2d space this would be just a line).
# An SVM is binary, meaning that it can only classify data into 2 sets at a time. This doesn't mean that an SVM can't
# classify data into more than 2 categories, it just means it only categorizes into 2 categories at a time.
# ^
# |
# |                +
# |              +
# |                +
# |
# |
# |
# |    -
# | -
# |   -
# |------------------>
# An SVM would look to find the separating hyperplane that maximizes the distance from that hyperplane to each point
# ^
# |    \
# |     \          +
# |      \       +/
# |      /\     //  +
# |     / /\   //  /
# |    / // \ //  /
# |   / //   \/  /
# |  / -/     \ /
# | -  /       \
# |   -         \
# |------------------>
# Imagine each of the lines were perpendicular to the separating hyperplane. Although there are other separating
# hyperplanes, this one maximizes the distance to each of the points. This concept can be applied in higher dimensions.
# ^
# |    \
# |     \   p1       +
# |      \       +
# |  p2   \           +
# |        \
# |         \
# |          \
# |    -      \
# | -          \
# |   -         \
# |------------------>
# Since p1 is to the right of the separating hyperplane, p1 is +. Since p2 is to the left of the separating hyperplane,
# it is -.
# Vectors have both magnitude and direction. Vectors are represented like this:
# ->
# A  = [3, 4] represents a vector going from [0, 0] to [3, 4]. Its magnitude is calculated like its length (or norm as
# it is called in linear algebra):
# ||A|| = sqrt((3 ^ 2) + (4 ^ 2))
# Multiplying vectors is done with the dot (·) operator and it results in a scalar (regular number), or a vector with
# just magnitude
# ->           ->
# B = [1, 3]   C = [4, 2]
# ->  ->
# A · B = (1 * 4) + (3 * 2) = 10
# Once the SVM has been trained, how does it classify something? This is done through support vector assertion (see
# Support Vector Assertion.png). A vector W is a vector perpendicular to the separating hyperplane. When the SVM goes to
# classify a point, it makes a vector U to that point and then projects U onto W and calculates, accounting for the
# bias, whether U "goes past" the separating hyperplane (+) or "stays behind" it (-). This is calculated by this
# formula:
# U · W + b
# If this is > 0 then it is positive, if it = 0 then it on the decision boundary, and if it is < 0 then it is negative.
# However, the SVM still has to calculate W and b. Math tells us that if point X is in the positive SV domain then
# the equation = 1 and if X is in the negative SV domain then the equation = -1. This means:
# X(+) · W + b = 1
# X(-) · W + b = -1
# The SVM can introduce variable y sub i (denoted yi), which is dependant on the class of X. This means when X is in
# the positive SV domain yi = 1 and when X is in the negative SV domain, yi = -1
# Going back to the previous equations, the SVM can now use yi to manipulate the equations:
# X(+) · W + b = 1      yi = 1
# X(-) · W + b = -1     yi = -1
# Multiplying both sides by yi:
# yi(X(+) · W + b) = 1 * 1 = 1
# yi(X(-) · W + b) = -1 * -1 = 1
# Subtracting 1 from both sides to make the right hand = 0:
# yi(X · W + b) - 1 = 0
# And this is the equation that the SVM uses to classify Xs. However, this equation doesn't define a support vector,
# but for a SVM to be good its support vectors must satisfy this equation. A support vector is simply a vector that,
# if moved, would change the best separating hyperplane.
# ^
# |               \
# |                +
# |                 \
# |\                 +
# | \                 \
# |  \                 \
# |   \                 \
# |    -
# |     \
# |      \
# |---------------------->
# These are 2 support vectors.
# To get the best separating hyperplane the SVM simply takes the width between the support vectors, divides it by 2,
# and adds that scalar to the smaller of the to support vectors.
# ^
# |              /\
# |             /  +
# |            /    \
# |\          /      +
# | \        /        \
# |  \      /width     \      The width line should be perpendicular to the 2 support vectors
# |   \    /            \
# |    -  /
# |     \/
# |      \
# |---------------------->
# However, to make the best separating hyperplane the SVM must maximize the width.
import time

import os

import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

docs_new = []

f = open("../rotten_tomatoes/rt-polarity.neg", "r", encoding="ISO-8859-1")

text = f.read()

f.close()

text = text.split("\n")

i = 0

for s in text:
    i += 1

    out = open("data/train/neg/" + str(i), "w")
    out.write(s)

f = open("../rotten_tomatoes/rt-polarity.pos", "r", encoding="ISO-8859-1")

text = f.read()

f.close()

text = text.split("\n")

i = 0

for s in text:
    i += 1

    out = open("data/train/pos/" + str(i), "w")
    out.write(s)

f = open("../rotten_tomatoes/imdb_labelled.txt")

text = f.read()

f.close()

text = text.split("\n")

for s in text:
    i += 1

    try:
        score = int(s[-1])

        if score == 1:
            out = open("data/test/pos/" + str(i), "w")
            out.write(s[:-1] + "\n")

        else:
            out = open("data/test/neg/" + str(i), "w")
            out.write(s[:-1] + "\n")

    except Exception as e:
        pass

data = load_files("data/train")
test = load_files("data/test")

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC())])

text_clf = text_clf.fit(data.data, data.target)

predicted = text_clf.predict(test.data)

print(metrics.classification_report(test.target, predicted, target_names=test.target_names))
