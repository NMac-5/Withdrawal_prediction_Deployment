import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import pickle

# Authentication
usernames_passwords = {'username1': 'password1', 'username2': 'password2'}


# Load the model 
model = pickle.load(open('best_xgb_model.pkl', 'rb'))  #The pickle file
scaler = pickle.load(open('clean_df_scaler.pkl', 'rb'))  #The pickle file


sponsor_codes = ['SP - 007B', 'SP - 005B', 'SP - 016B', 'SP - 008B', 'SP - 009B', 'SP - 007A', 'SP - 008A', 'SP - 003A', 'SP - 004B', 
                 'SP - 004A','SP - 1060', 'SP - 005A', 'SP - 001B', 'SP - 001A', 'SP - 1058','SP - 1074', 'SP - 0180', 'SP - 047A', 
                 'SP - 011A', 'SP - 002A','SP - 010A', 'SP - 1096', 'SP - 1200', 'SP - 036AG', 'SP - 036B','SP - 030B', 'SP - 030A', 
                 'SP - 030BG', 'SP - 1068', 'SP - 1048','SP - 1082', 'SP - 1087', 'SP - 1093', 'SP - 1050', 'SP - 1083','SP - 1071', 
                 'SP - 1073', 'SP - 1052', 'SP - 1097', 'SP - 1075','SP - 1081', 'SP - 039A', 'SP - 040B', 'SP - 040BG', 'SP - 040A',
                 'SP - 014B', 'SP - 011B', 'SP - 034B', 'SP - 037B', 'SP - 1051','SP - 022B', 'SP - 022A', 'SP - 045B', 'SP - 015B', 
                 'SP - 015A','SP - 020B', 'SP - 042B', 'SP - 017A', 'SP - 017B', 'SP - 029B','SP - 029A', 'SP - 021A', 'SP - 021B', 
                 'SP - 044B', 'SP - 035B','SP - 035A', 'SP - 047B', 'SP - 018B', 'SP - 019B', 'SP - 013B','SP - 027B', 'SP - 027C', 
                 'SP - 041B', 'SP - 041A', 'SP - 032A','SP - 032B', 'SP - 046A', 'SP - 028A', 'SP - 028B', 'SP - 039B','SP - 043B', 
                 'SP - 023B', 'SP - 026B', 'SP - 026A', 'SP - 038B','SP - 024B', 'SP - 024D', 'SP - 019AG', 'SP - 1066', 'SP - 1202',
                 'SP - 010B', 'SP - 012B', 'SP - 003B', 'SP - 031B', 'SP - 031A','SP - 006B', 'SP - 006A', 'SP - 033B', 'SP - 038AG', 
                 'SP - 038A','SP - 024AG', 'SP - 024A', 'SP - 028AG', 'SP - 042AG', 'SP - 002B','SP - 044AG', 'SP - 033A', 'SP - 032AG',
                   'SP - 046B', 'SP - 014A','SP - 019A', 'SP - 018A', 'SP - 042A', 'SP - 027A', 'SP - 034A','SP - 012A', 'SP - 036A', 
                   'SP - 037A', 'SP - 020A', 'SP - 025A','SP - 009A', 'SP - 1057', 'SP - 1063', 'SP - 1064', 'SP - 1062','SP - 045A', 
                   'SP - 045C', 'SP - 013A', 'SP - 1059', 'SP - 024BG','SP - 016A', 'SP - 023A', 'SP - 044A', 'SP - 1090', 'SP - 040AG',
                   'SP - 1077', 'SP - 1054', 'SP - 1067', 'SP - 031AG', 'SP - 1094','INFORMAL', 'SP - 006AG', 'SP - 043A', 'SP - 1088', 
                   'SP - 002C','SP - 1084', 'SP - 1091', 'SP - 1203', 'SP - 033AG', 'SP - 025B','SP - 1086', 'SP - 040C', 'SP - 015C',
                     'SP - 016C', 'SP - 001C','SP - 041C', 'SP - 038C', 'SP - 042BG', 'SP - 1201', 'SP - 038BG','SP - 1204', 'SP - 0182', 
                     'SP - 1210', 'SP - 1089', 'SP - 1098','SP - 0198', 'SP - 0185', 'SP - 0178', 'SP - 1199', 'SP - 1207','SP - 022BG', 
                     'SP - 0195', 'IMARISHA', 'SP - 0183', 'SP - 011AG','SP - 1085', 'SP - 010BG', 'SP - 1072']  # Your full list
encoded_sponsor_codes = [18,  13,  39,  20,  22,  17,  19,   8,  11,  10, 141,  12,   3,  2, 139, 151,  44, 131,  26,   5,  23, 167, 
                         171,  98,  99,  82,  81,  83, 147, 133, 155, 160, 165, 134, 156, 148, 150, 136, 168,  152, 154, 107, 111, 112, 
                         109,  34,  28,  94, 101, 135,  60,  59,  127,  36,  35,  56, 119,  42,  43,  80,  79,  57,  58, 125,  96,  95, 
                         132,  49,  54,  32,  74,  75, 115, 114,  87,  89, 129,  76,  78, 108, 122,  63,  72,  71, 104,  66,  68,  53, 
                         145, 173,  24,  30,   9,  86,  84,  16,  14,  92, 103, 102,  65,  64,  77, 118,  6, 124,  90,  88, 130,  33, 
                           52,  48, 117,  73,  93,  29,  97,  100,  55,  69,  21, 138, 143, 144, 142, 126, 128,  31, 140,  67,  38,  62,
                             123, 163, 110, 153, 137, 146,  85, 166,   1,  15, 121,  161,   7, 157, 164, 174,  91,  70, 159, 113,  37,
                                 40,   4, 116,  106, 120, 172, 105, 175,  45, 177, 162, 169,  51,  47,  41, 170,  176,  61,  50,   0,
                                     46,  27, 158,  25, 149]  # encoded labels

if 'username' not in st.session_state:
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    login_button = st.button('Login')
    if login_button:
        if username in usernames_passwords and usernames_passwords[username] == password:
            st.session_state['username'] = username
            st.success('Login successful!')
        else:
            st.error('Invalid username or password!')
else:
    st.title('Withdrawal Prediction')
    page = st.radio("Navigation", ["Prediction", "Upload Excel"])

    if page == "Prediction":
        # Input fields for user
        st.header("Please Input the following values")

        # User input fields
        gender_str = st.selectbox('Gender', ['Male', 'Female'])  # Radio buttons could also work
        dob_str = st.date_input('Date of Birth', min_value=date(year=1940, month=1, day=1), max_value = date.today() - timedelta(days=365 * 20)
                                , value=date(year=1980, month=1, day=1))
        sponsor_code_input = st.text_input('Sponsor Code')  # Assuming user knows their code
        join_date_str = st.date_input('Join Scheme Date', min_value=date(year=1940, month=1, day=1), max_value=date.today())
        last_contrib_str = st.date_input('Last Contribution Date', min_value=date.today() - timedelta(days=365*100), max_value=date.today())  # Adjust date range for reasonable contributions
        member_status_str = st.selectbox('Member Status', ['Active', 'Inactive', 'Deferred', 'Pending'])
        ee = st.number_input('Employee Contribution', min_value=0.00, step=0.01)
        er = st.number_input('Employer Contribution', min_value=0.00, step=0.01)
        eeavc = st.number_input('Employee Additional Voluntary Contribution', min_value=0.00, step=0.01)
        eravc = st.number_input('Employer Additional Voluntary Contribution', min_value=0.00, step=0.01)
        balance = st.number_input('Balance', min_value=0.00, step=0.01)
        has_kin = st.checkbox('Has Next of Kin')
       # Get user input and preprocess data
            
        # Button to trigger prediction
        predict_button = st.button('Predict')
        if predict_button:

          # Calculate derived features
          today = date.today()
          age = (today.year - dob_str.year )
          yrs_of_membership = (today.year - join_date_str.year)
          yrs_to_ret = 60 - age
          months_since_last_contrib = (today.year - last_contrib_str.year) * 12 + today.month - last_contrib_str.month
          # Find the encoded value for the sponsor code
          if sponsor_code_input in sponsor_codes:
            encoded_sponsor_code = encoded_sponsor_codes[sponsor_codes.index(sponsor_code_input)]
          else:
            # Handle cases where the code isn't found (optional: display error message)
            encoded_sponsor_code = -1  # Placeholder or error handling

          
          # Map categorical features
          Gender = 1 if gender_str == 'Male' else 0
          has_kin = int(has_kin)  # Convert checkbox to integer (1 or 0)
          member_status = {'Active': 0, 'Inactive': 2, 'Deferred': 1, 'Pending': 3}[member_status_str]  # Map member status using dictionary

          # Prepare data for prediction (assuming your model expects a 2D array)
          data = [[Gender, encoded_sponsor_code, member_status, ee, er, eeavc, eravc, balance, has_kin, age, yrs_of_membership, yrs_to_ret, months_since_last_contrib  ]]
          # '''Index(['Gender', 'Sponsor code', 'MemberStatus ', 'EE', 'ER', 'EEAVC', 'ERAVC','Balance', 'Has Next of Kin', 'Age', 'Yrs_of_Membership', 'Yrs_to_Ret',
          #   'Months_Since_last_Contribution'],
          #   dtype='object') 
          #   '''
          
          st.write(data)
          scaled_data = scaler.fit_transform(data)

          # Make prediction using the model
          prediction = model.predict(scaled_data)

          # Display prediction output
          prediction_text = 'Active' if prediction == 0 else 'Potential Withdrawal'
          st.write(f"Prediction: {prediction} Therefore the member is {prediction_text}")

    

    elif page == "Upload Excel":
       # Assuming you have your model and scaler loaded elsewhere

      uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"])

      if uploaded_file is not None:
          try:
              if uploaded_file.type == "text/csv":
                  df = pd.read_csv(uploaded_file)
              elif uploaded_file.type in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                          "application/vnd.ms-excel"):  # Covers both xlsx and xls
                  df = pd.read_excel(uploaded_file)
              else:
                  st.error("Unsupported file format. Please upload a CSV or Excel file.")
              # return

              # Your existing data preprocessing and prediction logic here (assuming 'data' is not used)
              data = df[['Gender', 'Sponsor code', 'MemberStatus ', 'EE', 'ER', 'EEAVC', 'ERAVC','Balance', 'Has Next of Kin', 'Age', 'Yrs_of_Membership', 'Yrs_to_Ret',
             'Months_Since_last_Contribution']]

             # Loop through each row and make predictions
              predictions = []
              for index, row in df.iterrows():
                  # Preprocess data for prediction
                  # Make prediction using the model
                  prediction = model.predict(data)
                  predictions.append(prediction)

              # Display predictions as a DataFrame
              df['Prediction'] = predictions
              st.write(df)

              # Optionally, add a button to download the predictions
              if st.button("Download Predictions"):
                  df.to_csv("predictions.csv", index=False)
                  st.success("Predictions downloaded successfully!")

          except Exception as e:
              st.error(f"Error reading file: {e}")


      

        # if uploaded_file is not None:
        #     # Read the Excel file
        #     df = pd.read_excel(uploaded_file)
        #     data = df[['Gender', 'Sponsor code', 'MemberStatus ', 'EE', 'ER', 'EEAVC', 'ERAVC','Balance', 'Has Next of Kin', 'Age', 'Yrs_of_Membership', 'Yrs_to_Ret',
        #     'Months_Since_last_Contribution']]
        #     # Loop through each row and make predictions
        #     predictions = []
        #     for index, row in df.iterrows():
        #         # Preprocess data for prediction
        #         # Make prediction using the model
        #         prediction = model.predict(data)
        #         predictions.append(prediction)

        #     # Display predictions as a DataFrame
        #     df['Prediction'] = predictions
        #     st.write(df)

        #     # Optionally, add a button to allow users to download the predictions
        #     if st.button("Download Predictions"):
        #         df.to_csv("predictions.csv", index=False)
        #         st.success("Predictions downloaded successfully!")
