import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification,make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics  import accuracy_score
from sklearn.metrics import classification_report

def load_initial_graph(dataset, ax, loaded_file=None):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=2)

    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)

    elif dataset == "load_csv":
        filepath = "/home/blade/Downloads/winequalityN.csv"
        df = pd.read_csv(filepath)

        # Fix missing values
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Convert quality → binary target
        df["wine_quality"] = (df["quality"] > 7).astype(int)

        # Select ONLY 2 numeric features for plotting
        feature1 = "alcohol"
        feature2 = "volatile acidity"

        X = df[[feature1, feature2]].values
        y = df["wine_quality"].values

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)

    # Plot
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="winter")
    return X, y
 
def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array



plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Logistic Regression Classifier")

dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Binary','Multiclass','load_csv')
)

penalty = st.sidebar.selectbox(
    'Regularization',
    ('l2', 'l1','elasticnet','none')
)

c_input = float(st.sidebar.number_input('C',value=1.0))

solver = st.sidebar.selectbox(
    'Solver',
    ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
)

max_iter = int(st.sidebar.number_input('Max Iterations',value=100))

multi_class = st.sidebar.selectbox(
    'Multi Class',
    ('auto', 'ovr', 'multinomial')
)

l1_ratio = int(st.sidebar.number_input('l1 Ratio'))

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
X,y = load_initial_graph(dataset,ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = LogisticRegression(penalty=penalty,C=c_input,solver=solver,max_iter=max_iter,multi_class=multi_class,l1_ratio=l1_ratio,class_weight ='balanced')
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    st.subheader("Accuracy for Decision Tree  " + str(round(accuracy_score(y_test, y_pred), 2)))
    st.text(classification_report(y_test,y_pred))