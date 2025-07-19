import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Step 1: Load Data
st.title("Simple Linear Regression - Salary Prediction")

st.header("Step 1: Load and Display Data")
uploaded_file = st.file_uploader("Upload CSV file",type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)

    #Step 2: Visualize the Data
    st.header("Step 2: Visualize Salary vs Experience")
    fig, ax = plt.subplots()
    ax.scatter(data["YearsExperience"], data["Salary"], color='blue')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary')
    ax.set_title("Scatter Plot")
    st.pyplot(fig)

#train the model
    st.header("Step 3: Train the Model")
    x = data[["YearsExperience"]]
    y = data["Salary"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)

    st.success("Model trained successfully!")

#step 4: Visualize the Regression Line
    st.header("Step 4: Visualize the Regression Line")
    fig2, ax2 = plt.subplots()
    ax2.scatter(x,y,color = 'blue',label='Actual Data')
    ax2.plot(x_test, model.predict(x_test), color='red',label='Regression Line')
    ax2.set_xlabel('Years of Experience')
    ax2.set_ylabel('Salary')
    ax2.legend()
    st.pyplot(fig2)
    st.success("Regression line plotted successfully!") 


    # Step 5: Predict Salary
    st.header("Step 5: Predict Salary")
    years_experience = st.number_input("Enter Years of Experience", min_value=0.0, step=0.1, key="experience_input")
    if st.button("Predict Salary"):
        prediction = model.predict([[years_experience]])
        prediction_inr = prediction[0] * 83
        formatted_salary = f"₹{prediction_inr:,.2f}"
        st.success(f"Predicted Salary for {years_experience} years of experience is {formatted_salary}")
