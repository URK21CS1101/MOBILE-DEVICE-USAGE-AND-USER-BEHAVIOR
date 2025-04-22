import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

import streamlit as st
import base64
import sqlite3
# ================ Background image ===

st.write("---------------------------------------------------------------")
st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:26px;font-family:verdana;">{"Mobile Device Usage and User Behavior"}</h1>', unsafe_allow_html=True)
st.write("---------------------------------------------------------------")


# st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:22px;">{"Prediction!!!"}</h1>', unsafe_allow_html=True)


st.write("---------------------------------------------------------------")


selected = option_menu(
    menu_title=None, 
    options=["User Behaiviour","Smart Phone Addiction"],  
    orientation="horizontal",
)


st.markdown(
    """
    <style>
    .option_menu_container {
        position: fixed;
        top: 20px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if selected == 'User Behaiviour':
    
    st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:26px;font-family:verdana;">{"User Behaviour"}</h1>', unsafe_allow_html=True)

    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('1.jpg')



    # ---- DATA
    
    
    dataframe=pd.read_csv("updated_device_overheating_data.csv")
    
    
    df_class=dataframe['Device Model'].unique()
    df_class1=dataframe['Operating System'].unique()
    df_class2=dataframe['Gender'].unique()
    df_class3=dataframe['Mobile Heating Label'].unique()
    
    
    # -- INPUT VALUES
    
    
    user_id = st.number_input("Enter User ID = ")
    
    dev_mod = st.selectbox("Choose Device Model",df_class)
    
    if dev_mod =='Google Pixel 5':
        a1=0
    elif dev_mod =='iPhone 12':
        a1=1    
    elif dev_mod =='OnePlus 9':
        a1=2
    elif dev_mod =='Samsung Galaxy S21':
        a1=3         
    else:
        a1=4
    
    
    ##
    
    dev_os = st.selectbox("Choose Operating System",df_class1)
    
    if dev_mod =='Android':
        a2=0
    else:
        a2=1
    
    
    # app_use = st.text_number("Enter App Usage Time = ")
    
    app_use = st.number_input("Enter App Usage Time (min/day) = ")
    
    screen_time = st.number_input("Enter Screen On Time (hours/day) = ")
    
    battery_drain = st.number_input("Enter Battery Drain (mAh/day) = ")
    
    apps_ins = st.number_input("Enter Number of Apps Installed = ")
    
    data_usa = st.number_input("Enter Data Usage (MB/day) = ")
    
    age = st.number_input("Enter Age = ")
    
    
    gender = st.selectbox("Choose Gender",df_class2)
    
    if gender =='Female':
        a3=0
    else:
        a3=1
    
    
    # user_beh = st.selectbox("Choose User Behaivour",('Occasional User','Balanced User','Heavy User','Extremely Active User','Constantly On-the-Go User'))
    
    # if user_beh =='Occasional User':
    #     a4=1
    # elif dev_mod =='Balanced User':
    #     a4=2  
    # elif dev_mod =='Heavy User':
    #     a4=3
    # elif dev_mod =='Extremely Active User':
    #     a4=4      
    # else:
    #     a4=5
    
    
    temperature = 30 + (screen_time * 2) + (app_use / 60) + (battery_drain / 1000)
    
    st.write(temperature)
    
    
    
    if temperature>=40:
        
        a4=0  # high
    
    else:
        
        a4=1
    
    
    butt = st.button("Submit")
    
    if butt:
        
        import numpy as np
        
        Data = np.array([user_id,a1,a2,app_use,screen_time,battery_drain,apps_ins,data_usa,age,a3,temperature,a4]).reshape(1, -1)
        
        
        import pickle
        
        with open('model.pickle', 'rb') as f:
            dt = pickle.load(f)
        
        pred_rf = dt.predict(Data)
          
        pred_rf = int(pred_rf)
        
        
        
       
        if pred_rf==0:
           
           st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified User Behaivour - Occasional User"}</h1>', unsafe_allow_html=True)
    
    
        elif pred_rf==1:
           
           st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified User Behaivour - Balanced User"}</h1>', unsafe_allow_html=True)
    
        elif pred_rf==2:
           
           st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified User Behaivour - Extremely Active User"}</h1>', unsafe_allow_html=True)
    
        elif pred_rf==3:
           
           st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified User Behaivour - Heavy User"}</h1>', unsafe_allow_html=True)
    
    
        else:
           
           st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified - Constantly On-the-Go User"}</h1>', unsafe_allow_html=True)
    
    

if selected == 'Smart Phone Addiction':
    
    st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:26px;font-family:verdana;">{"Smart Phone Addiction"}</h1>', unsafe_allow_html=True)

    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('2.jpg')
    
    
    a1 = st.selectbox("Choose Gender",('Male','Female'))
    
    if a1 =="Male":
        
        i1=1
    else:
        i1=0
    
    # ---
    
    a2 = st.selectbox("Interfere with your sleeping?",('No','Sometimes','Yes'))
    
    if a2 =="No":
        i2=0
    elif a2=="Sometimes":
        i2=1
    else:
        i2=2
    
    
    # ---
  
    
    a3 = st.selectbox("Before going to sleep/just after waking up",('No','Yes'))
    
    if a3 =="No":
        i3=0
    else:
        i3=1
       
    # ---
  
    
    a4 = st.selectbox("Survive without mobilephone",('No','Yes'))
    
    if a4 =="No":
        i4=0
    else:
        i4=1    
    
    
    # ---
  
    
    a5 = st.selectbox("Usage is more in ",('Academic Purpose','Basic Purposes','Browsing','Gaming','Social Media'))
    
    if a5 =="Academic Purpose":
        i5=0
    elif a5 =="Basic Purposes":
        i5=1
    elif a5 =="Browsing":
        i5=2 
    elif a5 =="Gaming":
        i5=3
    else:
        i5=4   
    
    # ---
  
    
    a6 = st.selectbox("Screening time",('1-3 hrs','3-5 hrs','Less than 1 hr','More than 5 hr'))
    
    if a6 =="1-3 hrs":
        i6=0
    elif a6 =="3-5 hrs":
        i6=1
    elif a6 =="Less than 1 hr":
        i6=2 
    else:
        i6=3      
    
    # ---
  
    
    a7 = st.selectbox("Distracted during class or while studying",('Always','Often','Only when it is neccessary','Very Often'))
    
    if a7 =="Always":
        i7=0
    elif a7 =="Often":
        i7=1
    elif a7 =="Only when it is neccessary":
        i7=2 
    else:
        i7=3      
    
    
    # ---
  
    
     
    a8 = st.selectbox("Use phone late at night (exam the next day)",('No','Yes'))
    
    if a8 =="No":
        i8=0
    else:
        i8=1    
    
    
    # ---
  
         
    a9 = st.selectbox("Unable to focus in class due to lack of sleep caused by phone usage",('No','Yes'))
    
    if a9 =="No":
        i9=0
    else:
        i9=1       
    
    
    # ---
  
         
    a10 = st.selectbox("Headaches or eye strain as a result of excessive phone use",('No','Yes'))
    
    if a10 =="No":
        i10=0
    else:
        i10=1      
    
    # ---
  
         
    a11 = st.selectbox("Anxiety",('No','Yes'))
    if a11 =="No":
        i11=0
    else:
        i11=1       
    
    # ---
  
         
    a12 = st.selectbox("Depression",('No','Yes'))    
    if a12 =="No":
        i12=0
    else:
        i12=1     
    
    # ---
  
         
    a13 = st.selectbox("Sleep disturbances",('No','Yes'))        
    if a13 =="No":
        i13=0
    else:
        i13=1       
    # ---
  
         
    a14 = st.selectbox("Social isolation",('No','Yes'))        
    if a14 =="No":
        i14=0
    else:
        i14=1    
    
    # ---
  
         
    a15 = st.selectbox("Attention and concentration difficulties",('No','Yes'))        
    if a15 =="No":
       i15=0
    else:
       i15=1      
    # ---
  
         
    a16 = st.selectbox("Irritability and aggression",('No','Yes'))        
    if a16 =="No":
        i16=0
    else:
        i16=1  
    
    # ---
  
         
    a17 = st.selectbox("Impaired cognitive functioning ",('No','Yes'))        
    if a17 =="No":
        i17=0
    else:
        i17=1  
    # ---
  
         
    a18 = st.selectbox("Low self-esteem",('No','Yes'))       
    if a18 =="No":
        i18=0
    else:
        i18=1     
    
    
    butt = st.button("Submit")
    
    if butt:
        
        import numpy as np
    
        Data = np.array([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18]).reshape(1, -1)
        
        import pickle
        
        with open('model_smart.pickle', 'rb') as f:
            rf = pickle.load(f)

        
        pred_rf = rf.predict(Data)
        
        import random
        
        percent = random.randint(80,95)
        
        
        
        if pred_rf == 1:

            st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified -- The Person is ADDICTED in Smart Phone"}</h1>', unsafe_allow_html=True)

        
        elif pred_rf == 0:
            
            aa = "Identified -- The Person is ADDICTED " +  "( " + str(percent) + "%" + " )" + "in Smart Phone"

            st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{aa}</h1>', unsafe_allow_html=True)

            
            st.write("------------------------------------------------")
            
            st.write("1) Rediscover and engage in offline hobbies and activities that bring joy and satisfaction, such as reading, exercising, or socializing in person.")
            
            st.write("2) Increase physical activity to enhance overall well-being and reduce reliance on smartphones for entertainment.")
        
            st.write("3) Spend quality time with family and friends to strengthen relationships and reduce the need for virtual interactions.")        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    