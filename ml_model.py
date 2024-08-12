import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay
)


# from logistic_regression import numerical_feature_selection, categorical_feature_selection

@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df


# Define features and target
def split_dataset(X, Y, test_size=0.30, random_state=7):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def numerical_feature_selection(df):
    numerical_columns = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    # Let the user select from these categorical features
    selected_features = st.multiselect(
        "Select numerical features for training the ML model:",
        options=numerical_columns
    )
    return selected_features


def categorical_feature_selection(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Let the user select from these categorical features
    selected_features = st.multiselect(
        "Select categorical features for training the ML model:",
        options=categorical_columns
    )
    return selected_features


def ml_model(uploaded_file):
    st.info("Model Selection + Training")
    df = load_data(uploaded_file)
    submenu = st.sidebar.selectbox("ML Models", ["Logistic Regression", "Decision Tree", "KNN"])
    if submenu == "Logistic Regression":
        st.title("Logistic Regression")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info("Choose numerical feature ")
            X = df[numerical_feature_selection(df)]
            st.dataframe(X)
        with col2:
            st.info("Choose categorical feature ")
            Y = df[categorical_feature_selection(df)]
            st.dataframe(Y)

        if "x_train" not in st.session_state:
            st.session_state.x_train = None
            st.session_state.x_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.lr_model = None
            st.session_state.predictions = None

        with st.expander("Split the dataset into training & testing"):
            # test_size = st.number_input("Insert test size", value=None, placeholder="Type a decimal number...")
            # random_state = st.number_input("Insert random state", value=None, placeholder="Type int number...")
            if st.button("Split"):
                st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test = split_dataset(
                    X, Y, test_size=0.3, random_state=10)
                st.write("The shape of original data", df.shape)
                st.write("The shape of training data", st.session_state.x_train.shape)
                st.write("The shape of testing data", st.session_state.y_test.shape)

        with st.expander("Training with split dataset"):
            if st.session_state.x_train is not None and st.session_state.y_train is not None:
                if st.button("Train"):
                    # Using LogisticRegression
                    lr_model = LogisticRegression()
                    lr_model.fit(st.session_state.x_train, st.session_state.y_train)
                    # saving to st.sesison_state
                    st.session_state.lr_model = lr_model
                    # saving to to a file pkl
                    model_file = open("logistic_regression_model_08112024.pkl", "wb")
                    joblib.dump(lr_model, model_file)
                    model_file.close()
                    st.write("Training complete!!! Please check the trained model file ")
                else:
                    st.info("Please click on the button to train the model")
            else:
                st.warning("Please split the data before training the model.")
        with st.expander("Prediction with test dataset"):
            if st.button("Predict"):
                st.session_state.predictions = st.session_state.lr_model.predict(st.session_state.x_test)
                st.write("Predictions on test data:", st.session_state.predictions)
                actual_values = st.session_state.y_test.iloc[:, 0].values
                results_df = pd.DataFrame({
                    "Actual": actual_values,
                    "Predicted": st.session_state.predictions
                })
            #
            # st.write("Predictions on test data:")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='logistic_regression_predictions.csv',
                    mime='text/csv'
                )
        with st.expander("Evaluate model with Metrics: Accuracy, Confusion Matrix, Classification Report, ROC curve"):
            if st.button("Evaluate"):
                st.info("Accuracy")
                accuracy_score = st.session_state.lr_model.score(st.session_state.x_test, st.session_state.y_test)
                st.write("Model Accuracy score:", accuracy_score)
                # if st.button("Plot Confusion Matrix"):

                st.info("Confusion Matrix Table and Plot")
                st.write(confusion_matrix(st.session_state.y_test, st.session_state.predictions))
                # Confusion Matrix
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay.from_estimator(
                    st.session_state.lr_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax,
                    cmap=plt.cm.Blues,
                    normalize='true'
                )
                st.pyplot(fig)
                # if st.button("Classification Report"):
                target_names = ["Negative(0)", "Positive(1)"]
                st.info("Classification Report")
                st.text(classification_report(st.session_state.y_test, st.session_state.predictions))

                # Plot and display precision-recall curve
                st.info("Precision-Recall Curve:")
                fig, ax = plt.subplots()
                disp = PrecisionRecallDisplay.from_estimator(
                    st.session_state.lr_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax
                )
                st.pyplot(fig)
                # Plot and display ROC curve
                st.info("ROC Curve:")
                fig, ax = plt.subplots()
                disp = RocCurveDisplay.from_estimator(
                    st.session_state.lr_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax
                )
                st.pyplot(fig)
    elif submenu == "Decision Tree":
        st.title("DecisionTree")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info("Choose numerical feature ")
            X = df[numerical_feature_selection(df)]
            st.dataframe(X)
        with col2:
            st.info("Choose categorical feature ")
            Y = df[categorical_feature_selection(df)]
            st.dataframe(Y)

        if "x_train" not in st.session_state:
            st.session_state.x_train = None
            st.session_state.x_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.dt_model = None
            st.session_state.predictions = None

        with st.expander("Split the dataset into training & testing"):
            # test_size = st.number_input("Insert test size", value=None, placeholder="Type a decimal number...")
            # random_state = st.number_input("Insert random state", value=None, placeholder="Type int number...")
            if st.button("Split"):
                st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test = split_dataset(
                    X, Y, test_size=0.3, random_state=10)
                st.write("The shape of original data", df.shape)
                st.write("The shape of training data", st.session_state.x_train.shape)
                st.write("The shape of testing data", st.session_state.y_test.shape)

        with st.expander("Training with split dataset"):
            if st.session_state.x_train is not None and st.session_state.y_train is not None:
                if st.button("Train"):
                    # Using Decision Tree
                    dt_model = DecisionTreeClassifier()
                    dt_model.fit(st.session_state.x_train, st.session_state.y_train)
                    # saving to st.sesison_state
                    st.session_state.dt_model = dt_model
                    # saving to to a file pkl
                    dt_model_file = open("decision_tree_model_08112024.pkl", "wb")
                    joblib.dump(dt_model, dt_model_file)
                    dt_model_file.close()
                    st.write("Training complete!!! Please check the trained model file ")
                else:
                    st.info("Please click on the button to train the model")
            else:
                st.warning("Please split the data before training the model.")
        with st.expander("Prediction with test dataset"):
            if st.button("Predict"):
                st.session_state.predictions = st.session_state.dt_model.predict(st.session_state.x_test)
                st.write("Predictions on test data:", st.session_state.predictions)

                actual_values = st.session_state.y_test.iloc[:, 0].values
                results_df = pd.DataFrame({
                    "Actual": actual_values,
                    "Predicted": st.session_state.predictions
                })
                #
                # st.write("Predictions on test data:")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='decisiontree_predictions.csv',
                    mime='text/csv'
                )
        with st.expander("Evaluate model with Metrics: Accuracy, Confusion Matrix, Classification Report, ROC curve"):
            if st.button("Evaluate"):
                st.info("Accuracy")
                accuracy_score = st.session_state.dt_model.score(st.session_state.x_test, st.session_state.predictions)
                st.write("Model Accuracy score:", accuracy_score)

                st.info("Confusion Matrix Table and Plot")
                st.write(confusion_matrix(st.session_state.y_test, st.session_state.predictions))
                # Confusion Matrix
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay.from_estimator(
                    st.session_state.dt_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax,
                    cmap=plt.cm.Blues,
                    normalize='true'
                )
                st.pyplot(fig)
                # if st.button("Classification Report"):
                target_names = ["Negative(0)", "Positive(1)"]
                st.info("Classification Report")
                st.text(classification_report(st.session_state.y_test, st.session_state.predictions))

                # Plot and display precision-recall curve
                st.info("Precision-Recall Curve:")
                fig, ax = plt.subplots()
                disp = PrecisionRecallDisplay.from_estimator(
                    st.session_state.dt_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax
                )
                st.pyplot(fig)
                # Plot and display ROC curve
                st.info("ROC Curve:")
                fig, ax = plt.subplots()
                disp = RocCurveDisplay.from_estimator(
                    st.session_state.dt_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax
                )
                st.pyplot(fig)
    else:
        st.title("K-Nearest Neighbors")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info("Choose numerical feature ")
            X = df[numerical_feature_selection(df)]
            st.dataframe(X)
        with col2:
            st.info("Choose categorical feature ")
            Y = df[categorical_feature_selection(df)]
            st.dataframe(Y)

        if "x_train" not in st.session_state:
            st.session_state.x_train = None
            st.session_state.x_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.knn_model = None
            st.session_state.predictions = None
            st.session_state.accuracy = None
            st.session_state.n_neighbors = 5
            st.session_state.results_df = None

        with st.expander("Split the dataset into training & testing"):
            test_size = st.number_input("Insert test size", value=0.3, min_value=0.1, max_value=0.9, step=0.1)
            random_state = st.number_input("Insert random state", value=7)

            if st.button("Split"):
                # Split the data and store it in session state
                st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test = split_dataset(
                    X, Y, test_size=test_size, random_state=random_state)

                st.write("The shape of original data", df.shape)
                st.write("The shape of training data", st.session_state.x_train.shape)
                st.write("The shape of testing data", st.session_state.y_test.shape)

        with st.expander("Training with split dataset"):
            if st.session_state.x_train is not None and st.session_state.y_train is not None:
                st.session_state.n_neighbors = st.number_input("Select number of neighbors", value=5, min_value=1,
                                                               step=1)
                if st.button("Train"):

                    knn_model = KNeighborsClassifier(n_neighbors=st.session_state.n_neighbors)
                    knn_model.fit(st.session_state.x_train, st.session_state.y_train)

                    # Save the trained model in session state
                    st.session_state.knn_model = knn_model

                    # saving to to a file pkl
                    knn_model_file = open("knn_model_08112024.pkl", "wb")
                    joblib.dump(knn_model, knn_model_file)
                    knn_model_file.close()
                    st.write("Training complete!!! Please check the trained model file ")
                else:
                    st.info("Please click on the button to train the model")
            else:
                st.warning("Please split the data before training the model.")
        with st.expander("Prediction with test dataset"):
            if st.button("Predict"):
                st.session_state.predictions = st.session_state.knn_model.predict(st.session_state.x_test)
                st.write("Predictions on test data:", st.session_state.predictions)

                actual_values = st.session_state.y_test.iloc[:, 0].values
                results_df = pd.DataFrame({
                    "Actual": actual_values,
                    "Predicted": st.session_state.predictions
                })
                #
                # st.write("Predictions on test data:")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='knn_predictions.csv',
                    mime='text/csv'
                )
        with st.expander("Evaluate model with Metrics: Accuracy, Confusion Matrix, Classification Report, ROC curve"):
            if st.button("Evaluate"):
                st.info("Accuracy")
                accuracy_score = st.session_state.knn_model.score(st.session_state.x_test, st.session_state.predictions)
                st.write("Model Accuracy score:", accuracy_score)

                st.info("Confusion Matrix Table and Plot")
                st.write(confusion_matrix(st.session_state.y_test, st.session_state.predictions))
                # Confusion Matrix
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay.from_estimator(
                    st.session_state.knn_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax,
                    cmap=plt.cm.Blues,
                    normalize='true'
                )
                st.pyplot(fig)
                # if st.button("Classification Report"):
                target_names = ["Negative(0)", "Positive(1)"]
                st.info("Classification Report")
                st.text(classification_report(st.session_state.y_test, st.session_state.predictions))

                # Plot and display precision-recall curve
                st.info("Precision-Recall Curve:")
                fig, ax = plt.subplots()
                disp = PrecisionRecallDisplay.from_estimator(
                    st.session_state.knn_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax
                )
                st.pyplot(fig)
                # Plot and display ROC curve
                st.info("ROC Curve:")
                fig, ax = plt.subplots()
                disp = RocCurveDisplay.from_estimator(
                    st.session_state.knn_model,
                    st.session_state.x_test,
                    st.session_state.y_test,
                    ax=ax
                )
                st.pyplot(fig)
