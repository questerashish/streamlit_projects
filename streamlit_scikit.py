from random import Random
from sqlite3 import paramstyle
import streamlit as st
from sklearn import metrics,preprocessing,model_selection,linear_model,svm,datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)



st.set_page_config(page_title="Machine Learning Basics with Streamlit and ScikitLearn")
st.title("Machine Learning Basics with Streamlit and ScikitLearn")

dataset_name = st.sidebar.selectbox("Select Dataset",("Iris Dataset","Breast Cancer","Wine Dataset"))
clf_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name=="Iris Dataset":
        dataset = datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        dataset = datasets.load_breast_cancer()
    else:
        dataset = datasets.load_wine()
    return dataset

dataset = get_dataset(dataset_name)
X = dataset.data
y = dataset.target

st.write("Shape of Dataset :",X.shape)
st.write("No of Classes :", (len(np.unique(y))))

def param_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
        with st.container():
            st.header("KNN Classifier Algorithm")
            st.text("""
1.K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on
  Supervised Learning technique.
2.K-NN algorithm assumes the similarity between the new case/data and available 
  cases and put the new case into the category that is most similar to the available
  categories.
3.K-NN algorithm stores all the available data and classifies a new data point based 
  on the similarity.
4.This means when new data appears then it can be easily classified into a well suite
  category by using K- NN algorithm.
5.K-NN algorithm can be used for Regression as well as for Classification but mostly 
  it is used for the Classification problems.
6.K-NN is a non-parametric algorithm, which means it does not make any assumption on 
  underlying data.
7.It is also called a lazy learner algorithm because it does not learn from the training 
  set immediately instead it stores the dataset and at the time of classification, it 
  performs an action on the dataset.
8.KNN algorithm at the training phase just stores the dataset and when it gets new data,
  then it classifies that data into a category that is much similar to the new data.
            """)
        return params

    elif clf_name == "SVM":
        params["C"] = st.sidebar.slider("C",0.01,10.0)
        with st.container():
            with st.container():
                st.header("SVM Classifier Algorithm")
                st.text("""
Support Vector Machine or SVM is one of the most popular Supervised Learning 
algorithms, which is used for Classification as well as Regression problems. 
However, primarily, it is used for Classification problems in Machine Learning.

The goal of the SVM algorithm is to create the best line or decision boundary 
that can segregate n-dimensional space into classes so that we can easily put 
the new data point in the correct category in the future. This best decision
boundary is called a hyperplane.

SVM chooses the extreme points/vectors that help in creating the hyperplane. 
These extreme cases are called as support vectors, and hence algorithm is termed
as Support Vector Machine. Consider the below diagram in which there are two
 different categories that are classified using a decision boundary or hyperplane
 """)
        return params
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        st.header("Random Forest Classifier Algorithm")
        st.text(
            """
Random Forest is a popular machine learning algorithm that belongs to the supervised
learning technique. It can be used for both Classification and Regression problems in ML. 
It is based on the concept of ensemble learning, which is a process of combining multiple 
classifiers to solve a complex problem and to improve the performance of the model.

As the name suggests, "Random Forest is a classifier that contains a number of decision
trees on various subsets of the given dataset and takes the average to improve the
predictive accuracy of that dataset." Instead of relying on one decision tree, the
random forest takes the prediction from each tree and based on the majority votes
of predictions, and it predicts the final output.

The greater number of trees in the forest leads to higher accuracy and prevents the problem
of overfitting.
            """
        )
        
        return params
    
params = param_ui(clf_name)

def get_classifier(clf_name,params):
    if clf_name=="KNN":
        model = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name =="SVM":
        model = SVC(C=params["C"])
    else:
        model = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=0)
    return model

model = get_classifier(clf_name,params)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model.fit(X_train,y_train)
prediction = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test,prediction)

st.write("Accuracy of the ",clf_name," model is :" , accuracy)


###### Plotting the data #########

#### Using PCA to reduce the dimensions of the data to 2d

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()

plt.scatter(x1,x2,c=y,alpha = 0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

df = pd.read_csv("summary_result.csv")



fdf = filter_dataframe(df)
st.write(fdf)



