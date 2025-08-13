import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

def home_page():
    st.markdown(
        """
        <h1 style='text-align: center; color: #2E86C1;'>
            Business Customer Churn Prediction
        </h1>
        <h3 style='text-align: center; color: #566573;'>
            Machine Learning for Business Insights
        </h3>
        """,
        unsafe_allow_html=True
    )
    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/1*CwJuemj1YV1oMbHIANWz4Q.png",
        use_container_width=True
    )
    st.subheader("Column Descriptions")

    st.write('"CustomerID": Unique float ID (e.g., 2.0 to 64374.0) for customer identification.')
    st.write('"Age": Customer age in years (18-65), for demographic segmentation.')
    st.write('"Gender": Binary string (\'Female\', \'Male\') for diversity analysis.')
    st.write('"Tenure": Subscription duration in months (1-60), indicates loyalty.')
    st.write('"Usage Frequency": Sessions per period (1-30), measures engagement.')
    st.write('"Support Calls": Number of support contacts (0-10), signals issues.')
    st.write('"Payment Delay": Days payments are delayed (0-30), reflects financial behavior.')
    st.write('"Subscription Type": Plan tier (\'Basic\', \'Standard\', \'Premium\'), shows value perception.')
    st.write('"Contract Length": Billing period (\'Monthly\', \'Quarterly\', \'Annual\'), affects churn barriers.')
    st.write('"Total Spend": Cumulative spend (100-999), indicates customer value.')
    st.write('"Last Interaction": Days since last activity (1-30), tracks dormancy.')
    st.write('"Churn": Binary target (0.0 = retained, 1.0 = churned) for prediction models.')
    df = pd.read_csv("df_cleaned_for_preprocessing.csv")
    st.write("Sample of DataSet")
    st.write(df.head(20))
    st.write("Summary Statistics of Data")
    st.write('For Numerical')
    st.write(df.describe())
    st.write('For Categorical')
    st.write(df.describe(include='object'))

def univariate_analysis():
    st.title("Univariate Analysis")
    df = pd.read_csv("df_for_analysis.csv")
    st.write("What is the Distribuation of Age?")
    st.write(px.histogram(data_frame=df, x='Age'))
    st.write("What is the Distribuation of Total Spend?")
    st.write(px.histogram(data_frame=df, x='Total Spend'))
    st.write("What is the Distribuation of Support Calls?")
    st.write(px.histogram(data_frame=df, x='Support Calls'))
    st.write("What is the Distribuation of Payment Delay?")
    st.write(px.histogram(data_frame=df, x='Payment Delay'))
    st.write("What is the Distribuation of Usage Frequency?")
    st.write(px.histogram(data_frame=df, x='Usage Frequency'))
    st.write("What is the Distribuation of Tenure?")
    st.write(px.histogram(data_frame=df, x='Tenure'))
    st.write("What is the Distribuation of Gender?")
    st.write(px.pie(data_frame=df, names='Gender'))
    st.write("What is the Distribuation of Subscription Type?")
    st.write(px.pie(data_frame=df, names='Subscription Type'))
    st.write("What is the Distribuation of Contract Length?")
    st.write(px.pie(data_frame=df, names='Contract Length'))
    st.write("What is the Distribuation of Age Group?")
    age_counts = df['Age Group'].value_counts().reset_index()
    age_counts.columns = ['Age Group', 'count']
    st.write(px.bar(age_counts, x='Age Group', y='count', text_auto=True))
    age_counts = df['Loyalty_Level'].value_counts().reset_index()
    age_counts.columns = ['Loyalty_Level', 'count']
    st.write(px.bar(age_counts, x='Loyalty_Level', y='count', text_auto=True))

    st.plotly_chart(px.box(df, y="Age", title="Box Plot for Age "))
    st.plotly_chart(px.box(df, y="Tenure", title="Box Plot for Tenure "))
    st.plotly_chart(px.box(df, y="Usage Frequency", title="Box Plot for Usage Frequency "))
    st.plotly_chart(px.box(df, y="Support Calls", title="Box Plot for Support Calls "))
    st.plotly_chart(px.box(df, y="Payment Delay", title="Box Plot for Payment Delay "))
    st.plotly_chart(px.box(df, y="Total Spend", title="Box Plot for Total Spend "))
    st.plotly_chart(px.box(df, y="Last Interaction", title="Box Plot for Last Interaction"))


def bivariate_analysis():
    st.title("Bivariate Analysis")
    df = pd.read_csv("df_for_analysis.csv")
    d1 = df.groupby('Subscription Type')['Avg_Spend_per_Month'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(d1, x='Subscription Type', y='Avg_Spend_per_Month', text_auto=True,title="The Summation of Avg Spend per Month by Subscription Type"))
    d2 = df.groupby('Contract Length')['Engagement_Score'].mean().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(d2, x='Contract Length', y='Engagement_Score', text_auto=True,title="The average engagement score of customers with different contract lengths"))
    d3 = df.groupby('Gender')['Support_Load'].max().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(d3, x='Gender', y='Support_Load', text_auto=True,title="The Maximum Support Load by Gender"))
    d4 = df.groupby('Age Group')['Last Interaction'].mean().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(d4, x='Age Group', y='Last Interaction', text_auto=True,title="The Average Last Interaction by Age Group"))
    d5 = df.groupby('Age Group')['Churn'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(d5, x='Age Group', y='Churn', text_auto=True,title="The Summation churn by Age Group"))
    d6 = df.groupby('Loyalty_Level')['Churn'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(d6, x='Loyalty_Level', y='Churn', text_auto=True,title="The Summation churn by Loyalty_Level"))
    d7=df.groupby('Contract Length')['Churn'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.pie(d7,values='Churn',names='Contract Length',title='Summation Churn for each contract length'))
    d8=df.groupby('Subscription Type')['Churn'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.pie(d8,names="Subscription Type",values="Churn",title="Summation Churn for each subscription type"))
    d9=df.groupby('Gender')['Churn'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.pie(d9,names="Gender",values="Churn",title="Summation Churn for each gender"))
    st.plotly_chart(px.scatter(data_frame=df, x="Total Spend", y="Avg_Spend_per_Month", trendline="ols").update_traces(
          mode='lines+markers', line=dict(color='red'), selector=dict(type='scatter', mode='markers')
          ).update_traces(
                line=dict(color='red'), selector=dict(type='scattergl', mode='lines')
                ))
    

def multivariate_analysis():
    st.title("Multivariate Analysis")
    df = pd.read_csv("df_for_analysis.csv")
    st.write("Correlation Matrix")
    df_corr = (df.drop(df.select_dtypes(include='object').columns, axis=1)).corr()
    st.plotly_chart(
    px.imshow(
        df_corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        width=1000,   
        height=800   
    )
 )
    st.plotly_chart(px.bar(df,x='Subscription Type',y='Tenure',color='Gender',title='Relation between Subscription Type and Tenure for each Gender',text_auto=True))
    #st.plotly_chart(px.bar(df,x='Loyalty_Level',y='Tenure',color='Gender',title='Relation between Loyalty_Level and Tenure for each Gender',text_auto=True))
    #st.plotly_chart(px.bar(df,x='Subscription Type',y='Last Interaction',color='Loyalty_Level',title='Relation between Subscription Type and Last Interaction for each Loyalty_Level',text_auto=True))
def custom_model_mode():
    st.subheader("📌 Custom Model Mode")

    uploaded_file = st.file_uploader("📂 Upload preprocessed CSV file", type=["csv"], key="custom_file_uploader")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Data Loaded Successfully")
        st.dataframe(df.head(20))

        target_column = st.selectbox(" Choose target column", df.columns, key="custom_target_col")
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]

        
            model_choice = st.selectbox(
                " Choose Model",
                ["Random Forest", "Logistic Regression", "KNN Classifier", "Decision Tree", "SVC"],
                key="model_choice"
            )

            if model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_choice == "KNN Classifier":
                model = KNeighborsClassifier()
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_choice == "SVC":
                model = SVC(probability=True, random_state=42)

        
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        
            cat_cols = X.select_dtypes(include=['object']).columns.tolist()
            num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

    
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
                ]
            )

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            if st.button(" Train Model", key="custom_train_btn"):
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                st.success(f" Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.text(classification_report(y_test, y_pred))
                joblib.dump(pipeline, "trained_model.pkl")
                st.success(" Model Saved Successfully")

            st.markdown("---")
            st.subheader(" Prediction on New Data")
            prediction_option = st.radio(
                "Choose input method",
                ["Upload CSV", "Manual Entry"],
                key="prediction_input"
            )

            if prediction_option == "Upload CSV":
                new_file = st.file_uploader("📂 Upload CSV for Prediction", type=["csv"], key="predict_file")
                if new_file is not None:
                    new_df = pd.read_csv(new_file)
                
                    missing_cols = set(X.columns) - set(new_df.columns)
                    for c in missing_cols:
                        new_df[c] = 0 
                    new_df = new_df[X.columns]  
                    st.write(" New Data for Prediction (first 1000 rows):")
                    st.dataframe(new_df.head(1000).reset_index(drop=True))
                    
                    loaded_pipeline = joblib.load("trained_model.pkl")
                    preds = loaded_pipeline.predict(new_df)
                    try:
                        probs = loaded_pipeline.predict_proba(new_df)
                        st.write(" Predictions with Probabilities:")
                        st.dataframe(
                            pd.DataFrame({"Prediction": preds, "Probability": probs.max(axis=1)}).head(1000).reset_index(drop=True)
                        )
                    except:
                        st.write(" Predictions:")
                        st.dataframe(pd.DataFrame({"Prediction": preds}).head(1000).reset_index(drop=True))

            elif prediction_option == "Manual Entry":
                input_data = {}
                for col in X.columns:
                    if df[col].dtype == 'object' or len(df[col].unique()) < 15:
                        options = df[col].unique().tolist()
                        val = st.selectbox(f"Select value for {col}", options)
                    else:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        val = st.number_input(f"Enter value for {col}", min_value=min_val, max_value=max_val, value=min_val)
                    input_data[col] = [val]

                if st.button("Predict Manual Data"):
                    new_df = pd.DataFrame(input_data)
                    st.dataframe(new_df.reset_index(drop=True))
                    loaded_pipeline = joblib.load("trained_model.pkl")
                    pred = loaded_pipeline.predict(new_df)
                    try:
                        prob = loaded_pipeline.predict_proba(new_df)
                        st.write(f" Prediction: {pred[0]} (Prob: {prob.max():.2f})")
                    except:
                        st.write(f" Prediction: {pred[0]}")


def machine_learning_page():
    custom_model_mode()

def filtering_data():
    df = pd.read_csv("df_cleaned_for_preprocessing.csv")
    st.title("Report")
    filtered_df = df.copy()
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        
        if pd.api.types.is_object_dtype(df[col]) or len(unique_vals) <= 10:
            selected_options = st.multiselect(
                f"Filter {col}",
                options=sorted(unique_vals),
                default=sorted(unique_vals)
            )
            filtered_df = filtered_df[filtered_df[col].isin(selected_options)]
        
        
        elif pd.api.types.is_numeric_dtype(df[col]):
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            
            if len(unique_vals) <= 10:
                selected_options = st.multiselect(
                    f"Filter {col}",
                    options=sorted(unique_vals),
                    default=sorted(unique_vals)
                )
                filtered_df = filtered_df[filtered_df[col].isin(selected_options)]
            else:
            
                selected_range = st.slider(
                    f"Filter {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= selected_range[0]) &
                    (filtered_df[col] <= selected_range[1])
                ]

    st.subheader("Filtered Data")
    st.dataframe(filtered_df)
def searching_page():
    st.title("Search")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("🔍 Search ")
        name = st.text_input("")
        search = st.button("Search")

    if search:
        if name.lower() == 'nti':  
            st.image("https://www.nti.sci.eg/b5g2024/images/smart-02.jpg")
            st.success(''' 
                NTI can stand for various terms depending on the context. Here are some of the most common meanings:

                Nuclear Threat Initiative: A nonprofit organization focused on reducing global nuclear, biological, and chemical threats through policy and action.
                Narrow Therapeutic Index: A medical term for drugs with a small margin between effective and toxic doses, requiring careful monitoring.
                National Transit Institute: A U.S.-based organization providing training and education for the public transportation industry.
                NewTech Infosystems: A software company known for developing backup and data recovery solutions.
                National Teachers Institute: An educational body, often associated with teacher training and certification, particularly in Nigeria.

                The specific meaning of NTI depends on the field or industry in question, such as technology, healthcare, education, or security.
            ''')
            st.balloons()
            st.snow()
            
        elif name.lower() == 'esraa abdullah':  
            st.success('''
                Esraa Abdullah is an exceptional and highly dedicated instructor at the National Teachers Institute (NTI),
                where she served as our mentor during an intensive and transformative training program. 
                Throughout our Machine Learning track, she invested extraordinary effort,
                showcasing her profound expertise and genuine passion for teaching. 
                Esraa’s remarkable patience, kindness, and approachable demeanor fostered an incredibly supportive 
                and inspiring learning environment, making even the most complex concepts accessible and engaging.
                Her unwavering commitment to our growth, coupled with her ability to provide clear 
                and personalized guidance, significantly enhanced our understanding, skills, 
                and confidence in Machine Learning, leaving a lasting impact on our educational journey.
            ''')
            st.balloons()
            st.snow()
            
        elif name.lower() == 'team':  
            st.success('''
                The Customer Churn project team, comprising
                Eng: Ahmed Rafaat,
                Eng: Osama Ashraf, 
                Eng: Ahmed Ashraf, and 
                Eng: Mostafa Yassin,
                worked collaboratively to deliver an outstanding and complete product.
                We performed meticulous data preprocessing, cleaning, and detailed analysis to ensure top-notch results.
                The team experimented with and implemented various classification models to predict customer churn,
                rigorously evaluating their performance to achieve optimal outcomes. To highlight our work,
                we deployed the final product using Streamlit, creating an engaging 
                and user-friendly interface that effectively showcased the project’s insights and functionality.
            ''')
            st.balloons()
            st.snow()
        else:
            st.info("Let's go 🚀 (No special match found)")

    
    
def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    st.sidebar.title("📂 Pages")
    page = st.sidebar.radio("Go to", ["Home", "Analysis","Reporting","Machine Learning","Searching"])
    if page == "Home":
        home_page()
    elif page == "Analysis":
        subpage = st.sidebar.radio("Analysis Pages", ["UniVariate Analysis", "BiVariate Analysis", "MultiVariate Analysis"])
        if subpage == "UniVariate Analysis":
            univariate_analysis()
        elif subpage == "BiVariate Analysis":
            bivariate_analysis()
        elif subpage == "MultiVariate Analysis":
            multivariate_analysis()
    elif page == "Machine Learning":
        machine_learning_page()
    elif page == "Reporting":
        filtering_data()
    elif page== "Searching":
        searching_page()

if __name__ == "__main__":
    main()
