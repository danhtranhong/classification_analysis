import streamlit as st
import streamlit.components.v1 as stc
from eda import eda_app,load_data
from ml_model import ml_model
import pandas as pd

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Classification Analysis </h1>
		</div>
		"""
def main():
	# st.title("ML Web App with Streamlit")
	stc.html(html_temp)

	menu = ["Home", "EDA", "Model Selection", "About"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Home":
		#st.subheader("Home")
		st.write("""
			#### Download the Dataset:
				- https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
				- https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
			#### App Structure
				- EDA Section: Exploratory Data Analysis of Data
				- Model Section and Training: select ML model to train 
				- ML Predcition and Evaluation: Run predictor app with test dataset and evaluate result
				- Using Logistic Regression /Decision Tree/ K-Nearest Neighbors
			""")
		uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
		if uploaded_file is not None:
			st.session_state['uploaded_file'] = uploaded_file
			st.write("File uploaded successfully!")
		else:
			st.write("Please upload a CSV file.")
	elif choice == "EDA":
		eda_app(st.session_state['uploaded_file'])
	elif choice == "Model Selection":
		ml_model(st.session_state['uploaded_file'])
	else:
		st.subheader("About")
		st.write("Testing")


if __name__ == '__main__':
	main()