Summary
This Mini-Project is to understand the use of AWS Sagemake pipelines through a Customer churn model development using Studio notebooks.
https://aws.amazon.com/blogs/machine-learning/build-tune-and-deploy-an-end-to-end-churn-prediction-model-using-amazon-sagemaker-pipelines/
Followed steps in Customer_Churn_Modeling.ipynb (https://github.com/aws-samples/customer-churn-sagemaker-pipelines-sample/blob/main/Customer_Churn_Modeling.ipynb) 

In a AWS Jupyter Lab Space, opened a JupyterNotebook with a python3.x kernel.

The dataset used is customer data of an online retail tea store got from https://www.kaggle.com/uttamp/store-data. Downloaded and moved this data to an S3 bucket for further use, through https://us-east-2.console.aws.amazon.com/s3

/home/sagemaker-user 
storedata_total.csv  storedata_total_data_dict.txt

The columns as in the storedata_total_data_dict.txt are as below.
custid		Computer generated ID to identify customers throughout the database
retained	"1, if customer is assumed to be active, 0 = otherwise"
created		Date when the contact was created in the database - when the customer joined
firstorder	Date when the customer placed first order
lastorder	Date when the customer placed last order
esent		Number of emails sent
eopenrate	Number of emails opened divided by number of emails sent
eclickrate	Number of emails clicked divided by number of emails sent
avgorder	Average order size for the customer
ordfreq		Number of orders divided by customer tenure
paperless	1 if customer subscribed for paperless communication (only online)
refill		1 if customer subscribed for automatic refill
doorstep	1 if customer subscribed for doorstep delivery
train		1 if customer is in the training database
favday	Customer's favorite delivery day
city	City where the customer resides in

Installed shap, smdebug and s3fs as outlined in the example. Some of this is already in python3.11. Upgraded sagemaker-jupyterlab-extension-common to avoid a version compatibility error. SHapley Additive exPlanations, more commonly known as SHAP, is used to explain the output of Machine Learning models. Related references are below.
https://aws.amazon.com/blogs/machine-learning/explaining-amazon-sagemaker-autopilot-models-with-shap/
https://sagemaker-debugger.readthedocs.io/en/website/README.html

The name Boto (pronounced boh-toh) comes from a freshwater dolphin native to the Amazon River. Boto3 is the official Amazon Web Services (AWS) Software Development Kit (SDK) for Python, and a Boto3 client is a low-level interface that provides access to AWS services.

Imported necessary packages and constants. Created a default s3 bucket using aws api and a s3_client = boto3.client('s3'), where boto3 is AWS's python SDK. Uploaded the data to the s3 bucket, s3_client.upload_file. Then created a Sagemaker session to work further with the data.

Read the data to a pandas dataframe. To preprocess the data do the following.
- convert 'firstorder' and 'lastorder' columns to datetime format
- drop rows with null values
- create a new column 'first_last_days_diff' that gives the diff in days
- create a new column 'created_first_days_diff' that gives days between when the customer record was created and the first order
- then drop columns 'custid','created','firstorder','lastorder'
- apply one hot encoding on favday and city columns

Next steps is to split the data into train, test and validation datasets
Column 'retained' to indicate customer retention, is the y (target).
70% of the data for train, 70-85% for validation and the last 15% for test.
They are then stored in the S3 default bucket  path as data/train/train.csv, data/validation/validation.csv and data/test/test.csv.

In the sagemaker session, next setup an estimator, which is a high level interface for sagemaker training. We use the XGBoost training algorithm container, using logistic regression for binary classification, output probability. And perform Hyperparameter Tuning 
for num_round=100 iterations, with rate_drop=0.3 to prevent overfitting. The evauation metric is auc (area under curve). ContinuousParameter ranges are used for eta (learning rate), min_child_weight, alpha and max_depth. With these settings, the HyperparameterTuner is instantiated and fit with the training and validation data.

The best hyperparameters are as below.
 'TunedHyperParameters': {'alpha': '0.6258914576986621',
                          'eta': '0.046565659236665',
                          'max_depth': '10',
                          'min_child_weight': '5.4938961090798255'}

The XGBoost Model with the Best Parameters is then tried, with a DebuggerHookConfig for analysis. This includes a train-auc and validation-auc plot for different iterations. The feature importance plot iteration on the X axis, shows esent (emails sent) as the top feature, followed by first_last_days_diff.
