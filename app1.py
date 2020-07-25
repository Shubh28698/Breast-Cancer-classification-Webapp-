import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
	st.title("Breast Cancer Classification Web app 🎀")
	st.sidebar.title("Breast Cancer Classification web app 🎀")
	st.markdown("Maligne or Benigne??")
	st.sidebar.markdown("Maligne or Benigne")

	@st.cache(persist=True)
	def load_data():
		data=pd.read_csv("C:/Users/Admin/Downloads/datasets/data.csv")
		label=LabelEncoder()
		for col in data.columns:
			data[col]=label.fit_transform(data[col])
		return data

	@st.cache(persist=True)
	def split(df):
		y=df.diagnosis
		x=df.drop(columns=['diagnosis'])
		x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
		return x_train,x_test,y_train,y_test

	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
			st.pyplot()

		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			plot_roc_curve(model,x_test,y_test)
			st.pyplot()

		if 'Precision-Recall Curve' in metrics_list:
			st.subheader("Precision-Recall Curve")
			plot_precision_recall_curve(model,x_test,y_test)
			st.pyplot()



	df= load_data()
	x_train,x_test,y_train,y_test=split(df)
	class_names=['maligne','Benigne']
	st.sidebar.subheader("choose classifier")
	classifier=st.sidebar.selectbox("Classifier",("support vector machine(SVM)","logistic regression"))


	if classifier == "support vector machine(SVM)":
		st.sidebar.subheader("Model Hyperparameters")
		C=st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C')
		kernel=st.sidebar.radio("kernel",("rbf","linear"),key='kernel')
		gamma =st.sidebar.radio("Gamma (kernel Coefficient)",("scale","auto"),key='gamma')


		metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

		if st.sidebar.button("Classify",key='classify'):
			st.subheader("support vector machine(SVM) results")
			model=SVC(C=C,kernel=kernel,gamma=gamma)
			model.fit(x_train,y_train)
			accuracy=model.score(x_test,y_test)
			y_pred=model.predict(x_test)
			st.write("Accuracy:",accuracy.round(2))
			st.write("Precision:",precision_score(y_test,y_pred,labels=class_names).round(2))
			st.write("Recall:",recall_score(y_test,y_pred,labels=class_names).round(2))
			plot_metrics(metrics)


	if classifier == "logistic regression":
		st.sidebar.subheader("Model Hyperparameters")
		C=st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C')
		max_iter=st.sidebar.slider("Maximum number of iterations",100,500,key='max_iter')
		

		metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

		if st.sidebar.button("Classify",key='classify'):
			st.subheader("logistic regression results")
			model=LogisticRegression(C=C,max_iter=max_iter)
			model.fit(x_train,y_train)
			accuracy=model.score(x_test,y_test)
			y_pred=model.predict(x_test)
			st.write("Accuracy:",accuracy.round(2))
			st.write("Precision:",precision_score(y_test,y_pred,labels=class_names).round(2))
			st.write("Recall:",recall_score(y_test,y_pred,labels=class_names).round(2))
			plot_metrics(metrics)


	if st.sidebar.checkbox("Show Raw data",False):
		st.subheader("Breast Cancer Dataset (classification)")
		st.write(df)

	


if __name__ == '__main__':
	main()
