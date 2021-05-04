import tensorflow_cloud as tfc
tfc.run(entry_point='GoogleTest.py')

# gcloud projects add-iam-policy-binding %PROJECT_ID% --member serviceAccount:%SA_NAME%@%PROJECT_ID%.iam.gserviceaccount.com --role 'roles/editor'
# gcloud iam service-accounts keys create ~/key.json --iam-account %SA_NAME%@%PROJECT_ID%.iam.gserviceaccount.com