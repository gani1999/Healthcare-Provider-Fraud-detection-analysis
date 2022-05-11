from flask import render_template, request, redirect, Flask
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
from sklearn.linear_model import LogisticRegression



##############################################
def Newfeature_groupby(Test_data, op_col2,op, group_col):
    '''
    This function creates a new feature using groupby operation. it groups the test and train data using col1 
    and apply the aggregate functions on top of col2
    '''
    for g_col in group_col:
        Test=pd.DataFrame()
        for col in op_col2:
            # new column name for the dataframe
            new_name = 'Per'+''.join(g_col)+'_'+col+'_'+op
            Test[new_name] = Test_data.groupby(g_col)[col].transform(op)
            
        Test_data=pd.concat([Test,Test_data], axis=1)
    return Test_data

#################################################
def predict_cluster(Test_data):
   
    X=Test_data.drop(['Provider'],axis=1)
    if 'PotentialFraud' in X.columns:
        X=Test_data.drop(['PotentialFraud', 'Provider'],axis=1)
    with open('std_scaler.pkl', 'rb') as f:
        std_scaler = pickle.load(f)
    std_data=std_scaler.transform(X)
    with open('PCA.pkl', 'rb') as f:
        PCA = pickle.load(f)
    pca_data=PCA.transform(std_data)
    
    with open('kmeans25.pkl', 'rb') as f:
        Kmeans = pickle.load(f)
        
    y_pred=Kmeans.predict(pca_data)
    
    return y_pred

###################################################

def ohc(Final_Test_data):
    print(Final_Test_data)
    print(Final_Test_data.columns)


    return Final_Test_data
    
    
######################################################
def preprocess_data(test_provider_data,test_benf_data, test_inpatient, test_outpatient,files):
    test_benf_data[['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure','ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
                   'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression','ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
                   'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis','ChronicCond_stroke']] = test_benf_data[
                   ['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure','ChronicCond_KidneyDisease', 'ChronicCond_Cancer','ChronicCond_ObstrPulmonary',
                   'ChronicCond_Depression','ChronicCond_Diabetes', 'ChronicCond_IschemicHeart','ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
                   'ChronicCond_stroke']].replace(to_replace=2,value=0)
    
    test_benf_data['risk_score']=test_benf_data['ChronicCond_Alzheimer']+test_benf_data['ChronicCond_Cancer']+test_benf_data['ChronicCond_Depression']\
                              +test_benf_data['ChronicCond_Diabetes']+test_benf_data['ChronicCond_Heartfailure']+test_benf_data['ChronicCond_IschemicHeart']\
                              +test_benf_data['ChronicCond_KidneyDisease']+test_benf_data['ChronicCond_KidneyDisease']+test_benf_data['ChronicCond_Osteoporasis']\
                              +test_benf_data['ChronicCond_Osteoporasis']+test_benf_data['ChronicCond_rheumatoidarthritis']
    
    test_benf_data['DOB'] = pd.to_datetime(test_benf_data['DOB'] , format = '%Y-%m-%d')
    test_benf_data['DOD'] = pd.to_datetime(test_benf_data['DOD'],format = '%Y-%m-%d',errors='ignore')

    test_benf_data['age']= round(((test_benf_data['DOD'] -test_benf_data['DOB']).dt.days)/365)
    test_benf_data['age']=test_benf_data['age'].fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - test_benf_data['DOB']).dt.days)/365))
    
    test_benf_data['died'] = 0
    test_benf_data.loc[test_benf_data.DOD.notna(), 'died'] = 1
    test_benf_data.loc[test_benf_data.age > 90, 'died'] = 1
    
    test_inpatient['ClaimStartDt'] = pd.to_datetime(test_inpatient['ClaimStartDt'] , format = '%Y-%m-%d')
    test_inpatient['ClaimEndDt'] = pd.to_datetime(test_inpatient['ClaimEndDt'],format = '%Y-%m-%d')
    test_inpatient['claim_period'] = ((test_inpatient['ClaimEndDt'] - test_inpatient['ClaimStartDt']).dt.days)+1

    test_inpatient['AdmissionDt'] = pd.to_datetime(test_inpatient['AdmissionDt'] , format = '%Y-%m-%d')
    test_inpatient['DischargeDt'] = pd.to_datetime(test_inpatient['DischargeDt'],format = '%Y-%m-%d')
    test_inpatient['Hospitalized_period'] = ((test_inpatient['DischargeDt'] - test_inpatient['AdmissionDt']).dt.days)+1

    test_inpatient['ExtraClaimDays'] = np.where( test_inpatient['claim_period']>test_inpatient['Hospitalized_period'], test_inpatient['claim_period'] - test_inpatient['Hospitalized_period'], 0)
    test_inpatient['same_physician'] = np.where( test_inpatient['AttendingPhysician']==test_inpatient['OperatingPhysician'], 1, 0)

    test_outpatient['ClaimStartDt'] = pd.to_datetime(test_outpatient['ClaimStartDt'] , format = '%Y-%m-%d')
    test_outpatient['ClaimEndDt'] = pd.to_datetime(test_outpatient['ClaimEndDt'],format = '%Y-%m-%d')
    test_outpatient['claim_period'] = ((test_outpatient['ClaimEndDt'] - test_outpatient['ClaimStartDt']).dt.days)+1

    test_outpatient['same_physician'] = np.where( test_outpatient['AttendingPhysician']==test_outpatient['OperatingPhysician'], 1, 0)

    test_inpatient['In_Outpatient'] = 1
    test_outpatient['In_Outpatient'] = 0
    
    # Merge inpatient and outpatient dataframes based on common columns
    common_columns_test = [ idx for idx in test_outpatient.columns if idx in test_inpatient.columns]
    Inpatient_Outpatient_Merge_Te = pd.merge(test_inpatient, test_outpatient, left_on = common_columns_test, right_on = common_columns_test,how = 'outer')

    # Merge beneficiary details with inpatient and outpatient data
    Inpatient_Outpatient_Beneficiary_Merge_Te = pd.merge(Inpatient_Outpatient_Merge_Te, test_benf_data,left_on='BeneID',right_on='BeneID',how='inner')

    Final_Test_data = pd.merge(Inpatient_Outpatient_Beneficiary_Merge_Te, test_provider_data , how = 'inner', on = 'Provider' )
    Final_Test_data = Final_Test_data.fillna(0)
    

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'age', 'Hospitalized_period', 'claim_period', 'risk_score']
    new_groupby_columns=['BeneID','AttendingPhysician','OperatingPhysician','OtherPhysician',
                    'ClmAdmitDiagnosisCode','DiagnosisGroupCode',
                    'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3',
                    'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6',
                    'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
                    'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
                    'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
                    'ClmProcedureCode_6']

    Final_Test_data =  Newfeature_groupby(Final_Test_data, columns, 'median', new_groupby_columns)

    columns = ['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'age', 'Hospitalized_period', 'claim_period', 'risk_score']
    new_groupby_columns=['Provider']
    Final_Test_data =  Newfeature_groupby( Final_Test_data,columns, 'count', new_groupby_columns)

    #https://datagy.io/pandas-get-dummies/
    # Do one hot encoding for gender and Race
    if files==1:
        print("files",files)
        # Convert type of Gender and Race to categorical
        Final_Test_data.Gender=Final_Test_data.Gender.astype('category')
        Final_Test_data.Race=Final_Test_data.Race.astype('category')
        Final_Test_data=pd.get_dummies(Final_Test_data,columns=['Gender','Race'])
        
    elif files==0:
        Final_Test_data['Race1']=0
        Final_Test_data['Race2']=0
        Final_Test_data['Race3']=0
        Final_Test_data['Race5']=0
        Final_Test_data['Gender1']=0
        Final_Test_data['Gender2']=0
        if 1 in Final_Test_data['Race'].values:
            Final_Test_data['Race1']=1
        elif 2 in Final_Test_data['Race'].values:
            Final_Test_data['Race2']=1
        elif 3 in Final_Test_data['Race'].values:
            Final_Test_data['Race3']=1
        elif 5 in Final_Test_data['Race'].values:
            Final_Test_data['Race5']=1

        if 1 in Final_Test_data['Gender'].values:
            Final_Test_data['Gender1']=1
        elif 2 in Final_Test_data['Gender'].values:
            Final_Test_data['Gender2']=1

        Final_Test_data=Final_Test_data.drop(['Race','Gender'], axis=1)

    

    Final_Test_data['RenalDiseaseIndicator']=Final_Test_data.RenalDiseaseIndicator.replace(['Y'],1)
    if 'PotentialFraud' in Final_Test_data.columns:
        Final_Test_data['PotentialFraud']=Final_Test_data.PotentialFraud.replace(['Yes','No'],[1,0])

    remove_columns=['BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician','OperatingPhysician', 'OtherPhysician',
                'ClmDiagnosisCode_1','ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4','ClmDiagnosisCode_5',
                'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7','ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
                'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3','ClmProcedureCode_4', 'ClmProcedureCode_5',
                'ClmProcedureCode_6','ClmAdmitDiagnosisCode', 'AdmissionDt','DischargeDt', 'DiagnosisGroupCode','DOB', 'DOD','State', 'County']

    Final_Test_data=Final_Test_data.drop(columns=remove_columns, axis=1)
    
    Final_Test_data['cluster']=predict_cluster(Final_Test_data)
    return Final_Test_data


########################################
def final_fun_1(X, files):

    file=files
    # Load the raw train data
    Test_Provider = X['Test_Provider']
    Test_Beneficiary = X['Test_Beneficiary']
    Test_Inpatient = X['Test_Inpatient']
    Test_Outpatient = X['Test_Outpatient']
    
    Final_data= preprocess_data(Test_Provider,Test_Beneficiary,Test_Inpatient,Test_Outpatient,file)
    print(Final_data.shape)
    # drop provider column
    test_provider = Final_data[['Provider']]
    test_data = Final_data.drop(axis=1,columns=['Provider'])

    # Standardize the data
    with open('std_scaler1.pkl', 'rb') as f:
        std_scaler = pickle.load(f)
    std_scaler.transform(test_data)
    
    with open('log_reg.pkl', 'rb') as f:
        best_model = pickle.load(f)
        
    y_pred=best_model.predict(test_data)
    test_provider['PredictedFraud']=y_pred
    
    return test_provider

##################################################
import flask
app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('page1.html')

@app.route('/forms', methods=["GET", "POST"])
def forms():
    redirect('forms')
    return flask.render_template('forms.html')

@app.route('/forms2', methods=["GET", "POST"])
def forms2():
    redirect('forms2')
    return flask.render_template('forms2.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    
    if request.method == 'POST':
        if request.files:
            
            uploaded_file = request.files['filename'] # This line uses the same variable and worked fine
            print(uploaded_file.filename)
            filepath = os.path.join('C:/Users/HP/Downloads/Flask', uploaded_file.filename)
            uploaded_file.save(filepath)
            test_provider_data=pd.read_csv(filepath)
            test_wh_frauddata=test_provider_data.drop(['PotentialFraud'], axis=1)

            uploaded_file = request.files['benefilename'] # This line uses the same variable and worked fine
            bene_filepath = os.path.join('C:/Users/HP/Downloads/Flask', uploaded_file.filename)
            uploaded_file.save(bene_filepath)
            test_benf_data= pd.read_csv(bene_filepath)

            uploaded_file = request.files['infilename'] # This line uses the same variable and worked fine
            in_filepath = os.path.join('C:/Users/HP/Downloads/Flask', uploaded_file.filename)
            uploaded_file.save(in_filepath)
            test_inpatient= pd.read_csv(in_filepath)

            uploaded_file = request.files['outfilename'] # This line uses the same variable and worked fine
            out_filepath = os.path.join('C:/Users/HP/Downloads/Flask', uploaded_file.filename)
            uploaded_file.save(out_filepath)
            test_outpatient= pd.read_csv(out_filepath)

            # create a dictionary which will contain all the files
            X = {"Test_Provider":test_wh_frauddata, "Test_Beneficiary":test_benf_data, "Test_Inpatient":test_inpatient, "Test_Outpatient":test_outpatient}

            data= final_fun_1(X, 1)
            
    return render_template('forms.html',tables=[data.to_html()], titles=[''])

@app.route('/provider', methods=["GET", "POST"])
def provider():
    if request.method == 'POST':
        data=request.form.to_dict()
        test_provider_data= pd.DataFrame([data.values()], columns=data.keys())
        filepath = os.path.join('C:/Users/HP/Downloads/Flask','provider.csv')
        test_provider_data.to_csv(filepath, index=False)
    return render_template('forms2.html',tables1=[test_provider_data.to_html()], titles1=[''])
    

@app.route('/beneficiary', methods=["GET", "POST"])
def beneficiary():
    if request.method == 'POST':
        data=request.form.to_dict()
        test_benf_data= pd.DataFrame([data.values()], columns=data.keys())
        filepath = os.path.join('C:/Users/HP/Downloads/Flask','beneficiary.csv')
        test_benf_data.to_csv(filepath, index=False)
    return render_template('forms2.html',tables2=[test_benf_data.to_html()], titles2=[''])


@app.route('/inpatient', methods=["GET", "POST"])
def inpatient():
    if request.method == 'POST':
        data=request.form.to_dict()
        test_inpatient= pd.DataFrame([data.values()], columns=data.keys())
        filepath = os.path.join('C:/Users/HP/Downloads/Flask','inpatient.csv')
        test_inpatient.to_csv(filepath, index=False)
    return render_template('forms2.html',tables3=[test_inpatient.to_html()], titles3=[''])

@app.route('/outpatient', methods=["GET", "POST"])
def outpatient():
    if request.method == 'POST':
        data=request.form.to_dict()
        test_outpatient= pd.DataFrame([data.values()], columns=data.keys())
        filepath = os.path.join('C:/Users/HP/Downloads/Flask','outpatient.csv')
        test_outpatient.to_csv(filepath, index=False)
    return render_template('forms2.html',tables4=[test_outpatient.to_html()], titles4=[''])
    

@app.route('/detect', methods=["GET", "POST"])
def detect():
    test_benf_data=pd.read_csv('beneficiary.csv')
    test_provider_data=pd.read_csv('provider.csv')
    test_inpatient=pd.read_csv('inpatient.csv')
    test_outpatient=pd.read_csv('outpatient.csv')
    
    X = {"Test_Provider":test_provider_data, "Test_Beneficiary":test_benf_data, "Test_Inpatient":test_inpatient, "Test_Outpatient":test_outpatient}

    data= final_fun_1(X,0)
            
    return render_template('forms2.html',tables=[data.to_html()], titles=[''])



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

app.config['FILE_UPLOADS'] = "C:\\Users\\HP\\Downloads\\Flask"
