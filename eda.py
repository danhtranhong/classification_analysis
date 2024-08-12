import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# # Increase the image size limit
# Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely (use with caution)
#
# # or set a specific higher limit
# Image.MAX_IMAGE_PIXELS = 500000000  # Set a higher limit (e.g., 500 million pixels)

@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df

def check_missing(df):
    missing_values = df.isnull()
    missing_counts = missing_values.sum()
    total_rows = df.shape[0]
    missing_percentages = (missing_counts / total_rows) * 100

    # Combine the counts and percentages into a summary DataFrame:
    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages
    })
    return missing_summary

def plot_histograms(df, bins=30, alpha=0.7, color='blue'):
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df[column], bins=bins, alpha=alpha, color=color)
        #plt.hist(df['radius_mean'], bins=bins, alpha=alpha, color=color)
        plt.title(f'Histogram of {column}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        st.pyplot(plt.gcf())
        plt.close()


def plot_pairplot(df, selected_features):
    if len(selected_features) == 0:
        st.warning("Please select at least one feature to plot.")
        return

    plt.figure(figsize=(10, 8))
    sns.pairplot(df[selected_features])
    st.pyplot(plt.gcf())
    plt.close()

def plot_correlation_heatmap(df,selected_features):
    if len(selected_features) == 0:
        st.warning("Please select at least one feature to plot.")
        return
    df_numeric = df.select_dtypes(exclude=['object'])
    corr = df_numeric[selected_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(plt.gcf())
    plt.close()

def eda_app(uploaded_file):
    st.subheader("Data Integrity Checks and EDA")
    df = load_data(uploaded_file)
    if df is None:
        st.error("No dataframe to process, please go back Home menu to upload")
    else:
        submenu = st.sidebar.selectbox("SubMenu", ["Descriptive", "Plots"])
        if submenu == "Descriptive":
            st.dataframe(df.head(10))
            with st.expander("Data Shape"):
                st.dataframe(df.shape)
                num_rows, num_columns = df.shape
                st.write(f"Number of rows: {num_rows}")
                st.write(f"Number of columns: {num_columns}")
            with st.expander("Data Describe"):
                st.dataframe(df.describe())
            with st.expander("Missing Value"):
                st.info("checking missing")
                st.write(check_missing(df))
            with st.expander("Data Types Summary"):
                st.dataframe(df.dtypes)
            with st.expander("Count Variable"):
                variable = st.text_input("Input variable to count")
                st.write("Input variable:", variable)
                if variable:
                    st.dataframe(df[variable].value_counts())
                else:
                    st.info("Please input variable name for counting!!!")
        else:
            st.subheader("Plots")
            # Call the function with your DataFrame
            st.dataframe(df.head(10))
            with st.expander("Histogram for all features"):
                if st.button("Show Histograms"):
                    plot_histograms(df)
                else:
                    st.info("Please click on the button to show charts")

            with st.expander("Pair plots for visualizing feature relationships"):
                selected_features = st.multiselect(
                    "Select maximum 5 features for the pair plot:",
                    options=df.columns.tolist(),
                    default=df.columns[:5].tolist()  # Default to first 5 features
                )
                if st.button("Show Pair Plots"):
                    if len(selected_features) > 5:
                        st.error("Only 5 features !!!")
                    else:
                        plot_pairplot(df, selected_features)
                else:
                    st.info("Please click on the button to show the Pair Plots")
            with st.expander("Correlation Heatmap of Dataset Features"):
                selected_features_heatmap = st.multiselect(
                    "Select maximum 5 features for heatmap:",
                    options=df.columns.tolist(),
                    default=df.columns[:5].tolist()  # Default to first 5 features
                )
                if st.button("Show Correlation Heatmap"):
                    plot_correlation_heatmap(df, selected_features_heatmap)
                else:
                    st.info("Please click on the button to show the heatmap")


