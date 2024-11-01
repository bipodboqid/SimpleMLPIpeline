# when test.py finished successfully, 
# run this module to create pipeline definition for Prod.

if __name__ == "__main__":
	
	
	import glob
	import os
	from google.cloud import aiplatform
	from google.cloud.aiplatform import pipeline_jobs
	import pipeline
	import time
	
	start_time = time.time()
	
	PROD_GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
	PROD_GOOGLE_CLOUD_REGION = 'asia-northeast1'
	PROD_GCS_BUCKET_NAME = 'mlpipelineportfolio_bucket_01'
	PROD_PIPELINE_NAME = 'mdl04-pipelineautomation'
	PROD_PIPELINE_ROOT = 'gs://{}/for_production/pipeline_root/{}'.format(PROD_GCS_BUCKET_NAME, PROD_PIPELINE_NAME)
	PROD_TRAIN_DATA_ROOT = 'gs://mlpipelineportfolio_bucket_01/for_production/upload_from_github/tfrecord/plantvillage-train-15percent'
	PROD_TEST_DATA_ROOT = 'gs://mlpipelineportfolio_bucket_01/for_production/upload_from_github/tfrecord/plantvillage-test-5percent'
	PROD_ENDPOINT_NAME = 'prediction-' + PROD_PIPELINE_NAME
	PROD_PIPELINE_DEFINITION_DIR = '/home/jupyter/SimpleMLPIpeline/Mdl03_DevProdIF_AutoUploadToGCS/pipeline_definition/prod/'
	PROD_PIPELINE_DEFINITION_FILE = PROD_PIPELINE_DEFINITION_DIR + PROD_PIPELINE_NAME + '_pipeline.json'
	PROD_SERVICE_ACCOUNT = 'pj03-vertex-account@mlpipelineportfolio.iam.gserviceaccount.com'
	PROD_MODULE_FILE = 'gs://mlpipelineportfolio_bucket_01/for_production/upload_from_github/modules/utils.py'
	
	# remove old pipeline definition jsons
	files = glob.glob(os.path.join(PROD_PIPELINE_DEFINITION_DIR, '*'))
	for file in files:
		try:
			os.remove(file)
			print(f"Deleted {file}")
		except Exception as e:
			print(f"Couldn't delete {file}. error: {e}")
	
	# create and get the path of definition json
	prod_template_path = pipeline.save_pipeline_definition(pipeline_name=PROD_PIPELINE_NAME,
														   pipeline_root=PROD_PIPELINE_ROOT,
														   train_data_root=PROD_TRAIN_DATA_ROOT,
														   test_data_root=PROD_TEST_DATA_ROOT,
														   module_file=PROD_MODULE_FILE,
														   endpoint_name=PROD_ENDPOINT_NAME,
														   project_id=PROD_GOOGLE_CLOUD_PROJECT,
														   region=PROD_GOOGLE_CLOUD_REGION,
														   pipeline_definition_file=PROD_PIPELINE_DEFINITION_FILE)
	
	file_mod_time = os.path.getmtime(prod_template_path)
	
	# check json file's timestamp against start time
	if file_mod_time > start_time:
		print('Successfully updated/created pipeline definition')
