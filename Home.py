#------------ Import Libraries started --------------------#

import pandas as pd
import numpy as np
from numpy import unique
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno


# Import scikit learn modules #
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




# Set plotly as backend for plotting through seaborn

pd.options.plotting.backend = "plotly"

# Import streamlit module
import streamlit as st
from streamlit_option_menu import option_menu

# OS modues
import os.path
import pathlib
from os import listdir
from os.path import isfile, join

#------------ Import Libraries ended --------------------#
st.set_page_config(layout="wide")
#------------- Title --------------------------------#
st.sidebar.header("Models-Comparison")

hide_menu_style = """
                    <style>
                    # MainMenu {visibility: hidden; }
                    footer {visibility: hidden; }
                    </style>


                    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

menu = option_menu(None,["Data","EDA","Models",  "About"],
                   icons=["cloud-upload","list-task", "list-task","house"],
                   menu_icon="cast", default_index=0, orientation="horizontal"
                   
                   )
    
    

if menu == "Data":
    import os.path
    import pathlib
    from os import listdir
    from os.path import isfile, join
    
    upload_file = st.sidebar.file_uploader("Choose a csv file")
    
    

    def upload():
        if upload_file is None:
            st.session_state["upload_state"] = "upload a file first !"
        else:
            data = upload_file.getvalue().decode("utf-8")
            
            parent_path = pathlib.Path(__file__).parent.resolve()
            save_path = os.path.join(parent_path, "data")
            complete_name = os.path.join(save_path, upload_file.name)
            
            
            destination_file = open(complete_name, "w")
            destination_file.write(data)
            destination_file.close()
            st.session_state["upload_state"] = "Saved" + complete_name + " "+ " successfuly !"
            
    st.sidebar.button("Load file for calculation", on_click = upload)
    
    upload_state = st.sidebar.text_area("Upload-Status", "", key = "upload_state")
    
      



if menu == "Data":
    
    
    #------------ Import Libraries started --------------------#

    import pandas as pd
    import numpy as np
    from numpy import unique
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    import missingno as msno


    # Import scikit learn modules #
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



   

    # Set plotly as backend for plotting through seaborn

    pd.options.plotting.backend = "plotly"

    # Import streamlit module
    import streamlit as st
    from streamlit_option_menu import option_menu

    # OS modules
    import os
    import pathlib
    from os import listdir
    from os.path import isfile, join

    #------------ Import Libraries ended --------------------#

    #------------------- Data Read Upload and view-------------#


    
    parent_path = pathlib.Path(__file__).parent.resolve()
    data_path = os.path.join(parent_path, "data")
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    option = st.sidebar.selectbox('Pick a dataset', onlyfiles)
    file_location = os.path.join(data_path, option)
    # use `file_location` as a parameter to the main script

    df = pd.read_csv(file_location)
    # df = df.apply(lambda x: x.fillna(x.mean()))
    st.session_state['df'] = df


    target_var = st.sidebar.selectbox("select target variable : ", list(df.columns))
    st.session_state['target_var'] = target_var
    
    

    tab1, tab2, tab3, tab4 = st.tabs(["🗃 Raw-Data", "📈 Raw-Data plots",  "📈 Dropped-Data",  "📈 Raw-PCA-plot"])

    st.session_state['tab1'] = tab1
    st.session_state['tab2'] = tab2
    
    
    
    with tab1:
        
        dt1c1, dt1c2 = st.columns(2)
        
        with dt1c1:
            st.subheader("Raw data Information :")
            st.write(df.shape)
            st.write(df)
        
        with dt1c2:    
            st.subheader("Description of data")
            st.write(df.shape)
            st.write(df.describe())
    with tab1:
        
        dt1c1, dt1c2 = st.columns(2)
        
        with dt1c1:
            st.subheader("Raw-data info :")
            st.write(df.shape)
            st.write(df.info())
        
        with dt1c2:    
            st.subheader("Null values in Raw-data")
            st.write(df.shape)
            st.write(df.isna().sum())
    
    


    with tab2:
        target_var = st.session_state['target_var']
        
        dt2c1, dt2c2 = st.columns(2)
        
        with dt2c1:
            st.subheader("Target varible bar-chart :")
            st.write(df.shape)
             
            st.write(df[target_var].value_counts().plot(kind="bar"))
        
        with dt2c2:    
            st.subheader("Target varible  Pie-chart")
            st.write(df.shape)
            target_pie = px.pie(df, hole= 0.6, names = target_var, color = target_var)
            target_pie.update_layout(legend = dict(orientation="h",  y= 1.02, x=1))
            st.write(target_pie)
        
    with tab2:
        
        dt2c1, dt2c2 = st.columns(2)
        
        with dt2c1:
            st.subheader("Null values chart")
            st.write(df.shape)
            st.write(df.isnull().sum().plot(kind = "bar"))
        
        with dt2c2:    
            st.subheader("Describe chart")
            st.write(df.shape)
            st.write(df.describe().plot(kind = "line"))
    
   
    
    
    
    
    # Dropped Data 
    
    drop_un_var = st.sidebar.multiselect("select unwanted variables to drop  : ", list(df.columns))
    st.session_state['drop_un_var'] = drop_un_var 
    
    encode_var = st.sidebar.multiselect("select columns for Encoding other than Target variable  : ", list(df.columns))
    st.session_state['encode_var'] = encode_var 
    
    
    df_dropped = df.drop(drop_un_var, axis=1, inplace=False)
    
    enc = OrdinalEncoder( dtype = "int64")
    
    df_dropped[encode_var] = enc.fit_transform(df_dropped[encode_var])
    
    
    st.session_state['df_dropped'] = df_dropped
    
    

    
    st.session_state['tab3'] = tab3
    st.session_state['tab4'] = tab4
    



    
    with tab3:
        
        dt3c1, dt3c2 = st.columns([4,1])
        
        with dt3c1:
            st.subheader("Dropped Data table")
            st.write(df_dropped.shape)
            st.write(df_dropped)
            
        
    
    
    

    #----- Tabs end here --------#

    

    # PCA to visualize data
    
    
    target_var = st.session_state['target_var']
    X_raw = df.drop([target_var], axis=1)
    sklearn_pca = PCA(n_components=2)
    PCs = sklearn_pca.fit_transform(X_raw)
    data_transform = pd.DataFrame(PCs,columns=['PC1','PC2'])
    data_transform = pd.concat([data_transform,df.iloc[:,-1]],axis=1)


    #plot

    pca_fig, axes = plt.subplots(figsize=(10,8))
    sns.set_style("whitegrid")
    sns.scatterplot(x='PC1',y='PC2',data = data_transform,hue=target_var,s=60, cmap='grey')

    tab4.subheader("Raw data PCA plot  :")
    tab4.write(pca_fig)
    


#------------------------ EDA starts here --------------------#


if menu == "EDA":
    
    
    
    #------------ Import Libraries started --------------------#

    import pandas as pd
    import numpy as np
    from numpy import unique
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    import missingno as msno


    # Import scikit learn modules #
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



   

    # Set plotly as backend for plotting through seaborn

    pd.options.plotting.backend = "plotly"

    # Import streamlit module
    import streamlit as st
    from streamlit_option_menu import option_menu
    # OS modules
    import os
    import pathlib
    from os import listdir
    from os.path import isfile, join

    
    #------------ Import Libraries ended --------------------#

    #------------- Get session variables from Data page ----------------#

    df = st.session_state['df'] 
    df_dropped = st.session_state['df_dropped'] 
    
    target_var = st.sidebar.selectbox("select target variable : ", list(df.columns))
    st.session_state['target_var'] = target_var
    
    X_raw = df_dropped.drop([target_var], axis=1)
    cor_mat = X_raw.corr()
    upper_tri = cor_mat.where(np.triu(np.ones(cor_mat.shape),k=1).astype(bool))
    cor_fig, axes = plt.subplots(figsize=(15,15))
    sns.heatmap(upper_tri, annot=True, cmap="YlGnBu")
    

    #-------------------------------------------------------------#



    st.write("""
    # Exploratory Data-Analysis
    """)

    

    
    etab1, etab2, etab3, etab4 = st.tabs(["📈 EDA plots", "📈 Filtered-Data", "📈 Normalized-Data", "📈 TrainTest split"])
    
    etab1.write(cor_fig)
    
    # Drop high correlated variables
    st.write("These are the highly correlated variables > 95 '%' similar and will be deleted automatically while creating Training data ")
    
    
    hc_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

    
    etab1.write(hc_drop)
    
    df_filt = df_dropped.drop(columns= hc_drop, axis=1)
    
    etab2.write(df_filt)
    
    
    
    
    test_size = st.sidebar.slider("Select split size", 1, 100,70)

    # EDA plots
    
    

    # Split the data 

    
    st.session_state['target_var'] = target_var
    
    X = df_filt.drop([target_var], axis=1)
    etab2.subheader("Feature Data array :")
    etab2.write(X.shape)
    etab2.write(X)




    y= df_filt[target_var]
    etab2.subheader("Target array :")
    etab2.write(y.shape)
    etab2.write(y)





    #Scale the data 
    sc = StandardScaler()
    X_norm = sc.fit_transform(X)
    etab3.subheader("Normalized Feature Data array :")
    etab3.write(X_norm.shape)
    etab3.write(X_norm)
    #----------------------------------------------------------- Data Menu -------------------------------------------------------#

    # Split the data
    # set aside 20% of train and test data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size= test_size, shuffle = True, random_state = 8)

    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test

    etab4.subheader("Train data  :")
    etab4.write(X_train.shape)
    etab4.write(X_train)

    etab4.subheader("Test data  :")
    etab4.write(X_test.shape)
    etab4.write(X_test)




#------------------------- EDA ends here --------------------#





if menu == "Models":
    
    #------------ Import Libraries started --------------------#

    import pandas as pd
    import numpy as np
    from numpy import unique
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    import missingno as msno


    # Import scikit learn modules #
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



    

    # Set plotly as backend for plotting through seaborn

    pd.options.plotting.backend = "plotly"

    # Import streamlit module
    import streamlit as st
    from streamlit_option_menu import option_menu
    # OS modules
    import os
    import pathlib
    from os import listdir
    from os.path import isfile, join

    
    #------------ Import Libraries ended --------------------#

    #------------- Get session variables from Data page ----------------#

    df = st.session_state['df'] 
    target_var = st.session_state['target_var']
    
    
    
    

    X_train = st.session_state['X_train'] 
    X_test = st.session_state['X_test'] 
    y_train = st.session_state['y_train'] 
    y_test = st.session_state['y_test']


    #-------------------------------------------------------------#



    st.header("""
    # Model Training and Performance Report 
    """)




    
    mtab1, mtab2, mtab3 = st.tabs(["📈 Model_report", "📈 Accuracy", "📈 Models compared"])




    model_name = st.sidebar.selectbox("select model and model parameters :",("KNN", "SVC", "Random Forest", "LogisticRegression"))

    #  Models name ends here  #

    #--------------------- Parameters Section starts here --------#

    def parameter_list(model_name):

        params = dict()

        if model_name == "KNN":
            K = st.sidebar.slider("K", 1, 15)
            params["K"] = K
        elif model_name == "SVC":
            C = st.sidebar.slider("C", 0.01, 100.0)
            gamma = st.sidebar.slider("gamma",1.0, 0.0001)
            kernel =st.sidebar.selectbox("kernel",["rbf"])
            params["C"] = C
            params["gamma"] = gamma
            params["kernel"] = kernel
        elif model_name == "Random Forest":
            max_depth = st.sidebar.slider("max_depth", 1, 20)
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        
        else:
            
            penalty = st.sidebar.selectbox("penalty",  ['none', 'l1', 'l2', 'elasticnet'])
            C = st.sidebar.slider("C", 0.001, 10.0)
            solver = st.sidebar.selectbox("solver", ["sag","lbfgs",  "saga" , "newton-cg"])
        
            params["penalty"] = penalty
            params["C"] = C
            params["solver"] = solver
        return params

    params = parameter_list(model_name)

    #--------------------- Parameters section ends here --------#

    #---------------------- Model Definition section starts here --------#
    def model_selection(model_name, params):
        
        pred = dict()
        predictions = dict()
        
        
        if model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=params["K"])
            pred[model_name] = model.fit(X_train, y_train).predict(X_test)
            y_pred = model.fit(X_train, y_train).predict(X_test)
            acc =  accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
        elif model_name == "SVC":
            model = SVC(C = params["C"])
            pred[model_name] = model.fit(X_train, y_train).predict(X_test)
            y_pred = model.fit(X_train, y_train).predict(X_test)
            acc =  accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
        
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=params["n_estimators"],
                                            max_depth=params["max_depth"],
                                            random_state = 444 
                                            )
            pred[model_name] = model.fit(X_train, y_train).predict(X_test)
            y_pred = model.fit(X_train, y_train).predict(X_test)
            acc =  accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
        
        else:
            model = LogisticRegression(penalty= params['penalty'] ,
                                       solver=params['solver'], 
                                       C= params["C"],
                                       random_state= 444)
            
            
            pred[model_name] = model.fit(X_train, y_train).predict(X_test)
            y_pred = model.fit(X_train, y_train).predict(X_test)
            acc =  accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
        return model, pred, acc, report

    model, pred , acc, report = model_selection(model_name, params) 
    
    
    
    # predictions = model_selection(model_name, params) 
    
    mtab1.write(model)
    mtab1.write(pred)
    
    mtab2.write(f"classifier = {model_name}")
    mtab2.write(f"Accuracy Score:, {acc} ")


    mtab2.write(f"classifier = {model_name}")
    mtab2.write(f" Classification report :{report}")
    
    
    #---------------------- Model Definition section ends here --------#

    #------------- Model Training and Evaluation ---------------------#
     
    LogisticRegressionModel = LogisticRegression(penalty='l2',solver='sag',C=1.0,random_state=33)
    lr_pred = LogisticRegressionModel.fit(X_train,y_train).predict(X_test)

    KNN_classifierModel = KNeighborsClassifier(n_neighbors = 30)
    KNN_pred=KNN_classifierModel.fit(X_train,y_train).predict(X_test)

    param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
    SVM_classifier = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)
    #SVCModel = SVC(kernel= 'rbf', max_iter=100,C=1.0,gamma='auto')
    svm_pred=SVM_classifier.fit(X_train,y_train).predict(X_test)

    GaussianNBModel = GaussianNB()
    gnb_pred = GaussianNBModel.fit(X_train,y_train).predict(X_test)

    DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=33) 
    dt_pred = DecisionTreeClassifierModel.fit(X_train,y_train).predict(X_test)


    param_grid = {'n_estimators': [10, 100,150,200,250,300,350,400]}
    RandomForestClassifierModel = GridSearchCV(RandomForestClassifier(),param_grid,refit=True, verbose=0)
    #RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=2,random_state=33)
    rf_pred = RandomForestClassifierModel.fit(X_train,y_train).predict(X_test)

    SGDClassifierModel = SGDClassifier(penalty='l2',learning_rate='optimal',random_state=33)
    SGD_pred = SGDClassifierModel.fit(X_train,y_train).predict(X_test)

    GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=33) 
    GBC_pred = GBCModel.fit(X_train,y_train).predict(X_test)

        
    # Predictions reports
    models=['Logistic Regression','KNN', 'SVM','GaussianNB','DecisionTree Classifier','RandomForest Classifier','SGD Classifier','GBCModel']
    preds=[lr_pred,KNN_pred, svm_pred,gnb_pred,dt_pred,rf_pred,SGD_pred,GBC_pred ]
    
    # Accuracy
    acc=[]
    for i in preds:
        accscore=accuracy_score(i,y_test).round(2)
        acc.append(accscore)
        
    
    
    join = zip(models, acc)
    

    result = pd.DataFrame(join, columns=['model', 'accuracy']).sort_values(['accuracy'], ascending=False) 
    st.session_state["result"] = result
    
    # cm_bar= sns.barplot(x = 'model', y= 'accuracy', data= result), plt.xticks(rotation =-90)
    cm_bar = px.bar( result , x = 'model',  y = 'accuracy', color = 'model', title = 'Models comparison according to accuracy ')
    
    
    with mtab3:
        
        mt3c1, mt3c2 = st.columns([3,1])
        
        with mt3c1:
            st.write(cm_bar)
        with mt3c2:
            st.table(result)
            
            
    
        

    #------------- Model Training and Evaluation  ends here---------------------#

    
    
    #---------------------- Report section starts here --------#  

    

    #---------------------- Report section ends here --------#

   


                                                        
                                                        
    