# when test.py finished successfully, 
# run this module to create pipeline definition for Prod.

if __name__ == "__main__":
	# record current time to check against definition json's timestamp
	import time
	start_time = time.time()
	
	import os
	import glob
	from google.cloud import aiplatform
	from google.cloud.aiplatform import pipeline_jobs
	import pipeline
	
	# constants of prod env
	PROD_GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
	PROD_GOOGLE_CLOUD_REGION = 'asia-northeast1'
	PROD_GCS_BUCKET_NAME = 'mlpipelineportfolio_bucket_01'
	PROD_PIPELINE_NAME = 'poc-ph03-pipeline-from-cloudfunction'
	PROD_PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(PROD_GCS_BUCKET_NAME, PROD_PIPELINE_NAME)
	PROD_DATA_ROOT = 'gs://mlpipelineportfolio_bucket_01/uploadFromGitHub/PoC/tfrecord'
	PROD_ENDPOINT_NAME = 'prediction-' + PROD_PIPELINE_NAME
	PROD_PIPELINE_DEFINITION_DIR = '/home/jupyter/SimpleMLPIpeline/uploadFromGitHub/PoC/pipeline_def_json/'
	PROD_PIPELINE_DEFINITION_FILE = PROD_PIPELINE_DEFINITION_DIR + PROD_PIPELINE_NAME + '_pipeline_' + str(int(start_time)) + '.json'
	PROD_SERVICE_ACCOUNT = 'pj03-vertex-account@mlpipelineportfolio.iam.gserviceaccount.com'
	PROD_MODULE_FILE = 'gs://mlpipelineportfolio_bucket_01/uploadFromGitHub/PoC/modules/utils.py'
	
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
														   data_root=PROD_DATA_ROOT,
														   module_file=PROD_MODULE_FILE,
														   endpoint_name=PROD_ENDPOINT_NAME,
														   project_id=PROD_GOOGLE_CLOUD_PROJECT,
														   region=PROD_GOOGLE_CLOUD_REGION,
														   pipeline_definition_file=PROD_PIPELINE_DEFINITION_FILE)
	
	file_mod_time = os.path.getmtime(prod_template_path)
	
	# check json file's timestamp against start time
	if file_mod_time > start_time:
		print('Successfully updated/created pipeline definition')
