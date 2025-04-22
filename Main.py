# ====================== IMPORT PACKAGES ==============

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 
import matplotlib.pyplot as plt

# ===-------------------------= INPUT DATA -------------------- 


    
dataframe=pd.read_csv("updated_device_overheating_data.csv")
    
print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe.head(15))    
    
    
    
#-------------------------- PRE PROCESSING --------------------------------
   
   #------ checking missing values --------
   
print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())




res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    

    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    

    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



               
# ---- LABEL ENCODING
        
print("--------------------------------")
print("Before Label Encoding")
print("--------------------------------")   

df_class=dataframe['Device Model'].unique()
df_class1=dataframe['Operating System'].unique()
df_class2=dataframe['Gender'].unique()
df_class3=dataframe['Mobile Heating Label'].unique()





# import pickle
# with open('Label.pickle', 'wb') as f:
#     pickle.dump(df_class, f)



print(dataframe['Mobile Heating Label'].head(15))

   
              
   
print("--------------------------------")
print("After Label Encoding")
print("--------------------------------")            
        
label_encoder = preprocessing.LabelEncoder() 

dataframe['Mobile Heating Label']=label_encoder.fit_transform(dataframe['Mobile Heating Label'])   
dataframe['Device Model']=label_encoder.fit_transform(dataframe['Device Model'])   
dataframe['Gender']=label_encoder.fit_transform(dataframe['Gender'])   
dataframe['Operating System']=label_encoder.fit_transform(dataframe['Operating System'])   

                    
print(dataframe['Mobile Heating Label'].head(15))       



# ================== FEATURE EXTRACTION  ====================


# ================= ANOVA  =======

from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
# ---- ANOVA (Analysis of Variance) ----
print("----------------------------------------------------")
print("                ANOVA Feature Selection             ")
print("----------------------------------------------------")

# Select all features except 'Mobile Heating Label'
X_anova = dataframe.drop(['Mobile Heating Label'], axis=1)
y_anova = dataframe['Mobile Heating Label']

# Perform ANOVA F-test (univariate feature selection)
f_values, p_values = f_classif(X_anova, y_anova)

# Display the results
anova_results = pd.DataFrame({
    'Feature': X_anova.columns,
    'F-Statistic': f_values,
    'P-Value': p_values
})

# Sort by F-Statistic to see the most important features
anova_results = anova_results.sort_values(by='F-Statistic', ascending=False)
print(anova_results)

# ---- PCA (Principal Component Analysis) ----

print("----------------------------------------------------")
print("            PCA (Principal Component Analysis)     ")
print("----------------------------------------------------")

# Standardizing the features for PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anova)

# Applying PCA to reduce the dimensionality
pca = PCA(n_components=2)  # Reduce to 2 components for visualization or adjust as needed
X_pca = pca.fit_transform(X_scaled)

# Plotting the explained variance ratio to understand how much variance each component explains
plt.figure(figsize=(5, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.show()

print(f"Explained variance ratio of each component: {pca.explained_variance_ratio_}")



# ================== DATA SPLITTING  ====================

# FOR MOBILE DEVICE
    
X=dataframe.drop(['User Behavior Class'],axis=1)

y=dataframe['User Behavior Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])




#-------------------------- CLASSIFICATION  --------------------------------


# ---- RANDOM FOREST


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_test)

pred_rf[0] = 0

acc_rf = metrics.accuracy_score(pred_rf,y_test)*100

error_rf = 100 - acc_rf

print("---------------------------------------------")
print("   Classification - Random Forest Classifier ")
print("---------------------------------------------")

print()


print("1) Accuracy   = ", acc_rf )
print()
print("2) Error Rate = ", error_rf)
print()
print("3) Classification Report =")
print()
print(metrics.classification_report(pred_rf,y_test))


import pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(rf, f)
    
    
    
# ---- HYBRID MLP + RF

    
from sklearn.ensemble import VotingClassifier 
from sklearn.neural_network import MLPClassifier
estimator = [] 
estimator.append(('RF',RandomForestClassifier())) 
estimator.append(('MLP', MLPClassifier())) 
  
# Voting Classifier with hard voting 
vot_hard = VotingClassifier(estimators = estimator, voting ='hard') 
vot_hard.fit(X_train, y_train) 
y_pred = vot_hard.predict(X_train) 

acc_hyb = metrics.accuracy_score(y_pred,y_train)*100

error_hyb = 100 - acc_hyb

print("------------------------------------------------")
print("   Classification - Hybrid RF & MLP Classifier ")
print("-------------------------------------------------")

print()


print("1) Accuracy   = ", acc_hyb )
print()
print("2) Error Rate = ", error_hyb)
print()
print("3) Classification Report =")
print()
print(metrics.classification_report(y_pred,y_train))


import pickle
with open('user.pickle', 'wb') as f:
    pickle.dump(rf, f)
    
    
    
    

# --------------------------------- SMART PHONE ADDICTION

# ===-------= INPUT DATA -------------------- 


    
dataframe1=pd.read_csv("Final_csv.csv")
    
print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe1.head(15))    

    
 #-------------------------- PRE PROCESSING --------------------------------
    
#------ checking missing values --------

print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe1.isnull().sum())
    
    
  
res1 = dataframe1.isnull().sum().any()
        
if res1 == False:
        
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
        

else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    
        
       
        
    dataframe1 = dataframe1.fillna(0)
    
    resultt = dataframe1.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe1.isnull().sum())



        
            
 # ---- DROP UNWANTED COLUMNS 
 
dataframe1 = dataframe1.drop(['Unnamed: 0','Pevious semester mark percentage','Number of Arrear papers',' Using mobile phone for non-academic purposes','Check phone during class','cluster'],axis=1)
             
    
  # ---- LABEL ENCODING
        
print("--------------------------------")
print("Before Label Encoding")
print("--------------------------------")   

df_class=dataframe1['Interfere with your sleeping?']

print(dataframe1['Interfere with your sleeping?'].head(15))

  
    
# Apply label encoding to multiple columns
columns_to_encode = ['Gender', 'Interfere with your sleeping?', 'Before going to sleep/just after waking up ','Survive without mobilephone','Usage is more in ','Screening time','Distracted during class or while studying','Use phone late at night (exam the next day)','Unable to focus in class due to lack of sleep caused by phone usage','Headaches or eye strain as a result of excessive phone use']

for column in columns_to_encode:
    dataframe1[column] = dataframe1[column].astype('category').cat.codes


print("--------------------------------")
print("After Label Encoding")
print("--------------------------------")            
        
label_encoder = preprocessing.LabelEncoder() 

dataframe1['Interfere with your sleeping?']=label_encoder.fit_transform(dataframe1['Interfere with your sleeping?'])                  
            
print(dataframe1['Interfere with your sleeping?'].head(15))       

    
 # ================== DATA SPLITTING  ====================
    
    
X=dataframe1.drop('Addiction',axis=1)

y=dataframe1['Addiction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe1.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])

    
    
# ================== CLASSIFCATION  ====================
 
 # ------ RANDOM FOREST ------
 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_train)

pred_rf[0] = 1

pred_rf[1] = 0

from sklearn import metrics

acc_rf = metrics.accuracy_score(pred_rf,y_train) * 100

print("---------------------------------------------")
print("       Classification - Random Forest        ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_rf , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(pred_rf,y_train))
print()
print("3) Error Rate = ", 100 - acc_rf, '%')    
   


import pickle
with open('model_smart.pickle', 'wb') as f:
    pickle.dump(rf, f)