# Healthcare-Provider-Fraud-detection-analysis
### Introduction:
Fraud can be defined as a dishonest act committed by an individual or a group of people with the knowledge and for the financial benefits. Healthcare fraud can be defined as misrepresenting information, concealing information or deceiving a person or an entity in order to get benefitted financially. In Health care the fraud involves the health care system by an individual, Physicians, Doctors, Healthcare providers and the insurance companies.

In our case, we will analyze all the details of the healthcare provider and conclude whether legitimate or not. If the providers fill all the details on behalf of the beneficiaries and makes a claim to get benefitted then it is considered as a Fraud. Health care fraud is one of the biggest problems in healthcare domain across world. In USA, the insurance company should clear all the compensations within 30 days of claim. So, there is less time to investigate carefully and also the claims are increasing rapidly so it’s hard to investigate all the claims manually. So, its wise adopt a computerized technique that automatically investigate through the beneficiary details and suggest whether a claim is Fraud or not.

### ML Formulation and Business problem:
From references we understand, according to the survey, it is estimated that over 15% of claims are fraud and insurance companies in USA incur losses over 30 billion USD annually. And in India insurance companies incur approximately 600-800 crores annually.
From given dataset our objective is the predict whether the provider is fraud or not. We need to obtain a probability score of the provider fraudulent activity, by analyzing the beneficiary details and reasons why healthcare provider is fraud. So that we can prevent insurance companies from incurring financial losses.

### Business constraints:
1.Cost of misclassification is very high, where if we predict legitimate provider as fraud (False Positive) it costs for further investigation and also a matter of companies’ reputation. If we predict fraud provider as a legitimate provider (False Negative) then we will end up with huge financial losses.
2. No strict latency requirements.
3. Feature interpretability is highly important- As Insurance company should justify the fraudulent activity of the provider and need to set up manual investigation if needed.
### Mapping to ML:
We need to build a binary classification algorithm based the details filled by the provider, inpatient data, outpatient data and beneficiary data to predict whether the health care provider is fraud or not.

### Data Overview:
We have train and test data with 4 datasets each. Which are
Provider data - ProviderID, Potential Fraud.
                       
### Beneficiary data-
![image](https://user-images.githubusercontent.com/47345492/167853875-f2ab56f4-4836-4b04-a6c6-a20176057ff8.png)

1.	BeneID: It contains the unique id of the beneficiary.
2.	DOB: It contains the Date of Birth of the beneficiary.
3.	DOD: It contains the Date of Death of the beneficiary if the beneficiary id dead else null.
4.	Gender, Race, State, Country: It contains the Gender, Race, State, Country of the beneficiary.
5.	RenalDiseaseIndicator: It contains if the patient has existing kidney disease.
6.	ChronicCond_*: The columns started with “ChronicCond_” indicates if the patient has existing that particular disease. Which also indicates the risk score of that patient.
7.	IPAnnualReimbursementAmt: It consists of the maximum reimbursement amount for hospitalization annually.
8.	IPAnnualDeductibleAmt: It consists of a premium paid by the patient for hospitalization annually.
9.	OPAnnualReimbursementAmt: It consists of the maximum reimbursement amount for outpatient visits annually.
10.	OPAnnualDeductibleAmt: It consists of a premium paid by the patient for outpatient visits annually.

### Inpatient data & Outpatient data –
![image](https://user-images.githubusercontent.com/47345492/167853776-c9f71c67-3286-4b84-92ee-64af8be2644e.png)

1.	BeneID: It contains the unique id of each beneficiary i.e., patients.
2.	ClaimID: It contains the unique id of the claim submitted by the provider.
3.	ClaimStartDt: It contains the date when the claim started in yyyy-mm-dd format.
4.	ClaimEndDt: It contains the date when the claim ended in yyyy-mm-dd format.
5.	Provider: It contains the unique id of the provider.
6.	InscClaimAmtReimbursed: It contains the amount reimbursed for that particular claim.
7.	AttendingPhysician: It contains the id of the Physician who attended the patient.
8.	OperatingPhysician: It contains the id of the Physician who operated on the patient.
9.	OtherPhysician: It contains the id of the Physician other than AttendingPhysician and OperatingPhysician who treated the patient.
10.	ClmDiagnosisCode: It contains codes of the diagnosis performed by the provider on the patient for that claim.
11.	ClmProcedureCode: It contains the codes of the procedures of the patient for treatment for that particular claim.
12.	DeductibleAmtPaid: It consists of the amount by the patient. That is equal to Total_claim_amount - Reimbursed_amount.
13.	AdmissionDt: It contains the date on which the patient was admitted into the hospital in yyyy-mm-dd format.
14.	DischargeDt: It contains the date on which the patient was discharged from the hospital in yyyy-mm-dd format.
15.	DiagnosisGroupCode: It contains a group code for the diagnosis done on the patient.

### PERFORMANCE METRICS:
As our health care dataset is highly imbalanced, accuracy score is not a good metric to measure performance. And cost of misclassification is high we opt for following metrics.
1.	Confusion matrix
2.	F1 score – Harmonic mean of precision and recall.
3.	AUC score – Area under the curve, close to 1 better the model performance.
4.	FPR, FNR - cost of misclassification is high we need to check out these two metrics carefully, should be low for better model.


