# This module is used to produce json definition of a pipeline 
# which is meant to be run on production environment.

if __name__ == "__main__":
	from google.cloud import aiplatform
	from google.cloud.aiplatform import pipeline_jobs
	import pipeline


	# constants of dev env
	DEV_GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
	DEV_GOOGLE_CLOUD_REGION = 'asia-northeast1'
	DEV_GCS_BUCKET_NAME = 'mlpipelineportfolio_bucket_01'
	DEV_PIPELINE_NAME = 'poc-ph03-pipeline-from-cloudfunction-devrun'
	DEV_PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(DEV_GCS_BUCKET_NAME, DEV_PIPELINE_NAME)
	DEV_DATA_ROOT = 'gs://mlpipelineportfolio_bucket_01/data/poc-ph03-pipeline-from-cloudfunction-devrun'
	DEV_ENDPOINT_NAME = 'prediction-' + DEV_PIPELINE_NAME
	DEV_PIPELINE_DEFINITION_FILE = '/home/jupyter/Poc/Ph03/PipelineDefinitionOnDev/' + DEV_PIPELINE_NAME + '_pipeline.json'
	DEV_SERVICE_ACCOUNT = 'pj03-vertex-account@mlpipelineportfolio.iam.gserviceaccount.com'
	DEV_MODULE_FILE = DEV_DATA_ROOT + '/modules/utils.py'
	
	# constants of prod env
	PROD_GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
	PROD_GOOGLE_CLOUD_REGION = 'asia-northeast1'
	PROD_GCS_BUCKET_NAME = 'mlpipelineportfolio_bucket_01'
	PROD_PIPELINE_NAME = 'poC-ph03-pipeline-from-cloudfunction'
	PROD_PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(PROD_GCS_BUCKET_NAME, PROD_PIPELINE_NAME)
	PROD_DATA_ROOT = 'gs://mlpipelineportfolio_bucket_01/uploadFromGitHub/PoC/tfrecord'
	PROD_ENDPOINT_NAME = 'prediction-' + PROD_PIPELINE_NAME
	PROD_PIPELINE_DEFINITION_FILE = PROD_PIPELINE_NAME + '_pipeline.json'
	PROD_SERVICE_ACCOUNT = 'pj03-vertex-account@mlpipelineportfolio.iam.gserviceaccount.com'
	PROD_MODULE_FILE = 'gs://mlpipelineportfolio_bucket_01/uploadFromGitHub/PoC/modules/utils.py'

	# run and save definition of pipeline on devenv
# 	template_path = pipeline.save_pipeline_definition(pipeline_name=DEV_PIPELINE_NAME,
# 													  pipeline_root=DEV_PIPELINE_ROOT,
# 													  data_root=DEV_DATA_ROOT,
# 													  module_file=DEV_MODULE_FILE,
# 													  endpoint_name=DEV_ENDPOINT_NAME,
# 													  project_id=DEV_GOOGLE_CLOUD_PROJECT,
# 													  region=DEV_GOOGLE_CLOUD_REGION,
# 													  pipeline_definition_file=DEV_PIPELINE_DEFINITION_FILE)

# 	# submit the pipeline for testing on devenv
# 	aiplatform.init(project=DEV_GOOGLE_CLOUD_PROJECT, location=DEV_GOOGLE_CLOUD_REGION)

# 	job = pipeline_jobs.PipelineJob(template_path=template_path,
# 									display_name=DEV_PIPELINE_NAME)
	
# 	print('successfully created PipelineJob Instance')

# 	job.submit(service_account=DEV_SERVICE_ACCOUNT)
	
# 	print('submit complete')

# 	# wait for the result of the test
# 	job.wait()

# 	# get and use the result as a condition of downstream processes
# 	pipeline_state = job.state
# 	print('pipelinestate: {}'.format(pipeline_state))

# 	# save pipeline definition for prod env
# 	# only when the pipeline successfully finishes on the dev env
# 	if pipeline_state == 4:
# 		prod_template_path = pipeline.save_pipeline_definition(pipeline_name=PROD_PIPELINE_NAME,
# 														  pipeline_root=PROD_PIPELINE_ROOT,
# 														  data_root=PROD_DATA_ROOT,
# 														  module_file=PROD_MODULE_FILE,
# 														  endpoint_name=PROD_ENDPOINT_NAME,
# 														  project_id=PROD_GOOGLE_CLOUD_PROJECT,
# 														  region=PROD_GOOGLE_CLOUD_REGION,
# 														  pipeline_definition_file=PROD_PIPELINE_DEFINITION_FILE)
# 	else:
# 		pass
	
	prod_template_path = pipeline.save_pipeline_definition(pipeline_name=PROD_PIPELINE_NAME,
														   pipeline_root=PROD_PIPELINE_ROOT,
														   data_root=PROD_DATA_ROOT,
														   module_file=PROD_MODULE_FILE,
														   endpoint_name=PROD_ENDPOINT_NAME,
														   project_id=PROD_GOOGLE_CLOUD_PROJECT,
														   region=PROD_GOOGLE_CLOUD_REGION,
														   pipeline_definition_file=PROD_PIPELINE_DEFINITION_FILE)
	
