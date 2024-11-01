# Run pipeline on devenv and check the status

if __name__ == "__main__":
        from google.cloud import aiplatform
        from google.cloud.aiplatform import pipeline_jobs
        import pipeline
        
        # constants of dev env
        DEV_GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
        DEV_GOOGLE_CLOUD_REGION = 'asia-northeast1'
        DEV_GCS_BUCKET_NAME = 'mlpipelineportfolio_bucket_01'
        DEV_PIPELINE_NAME = 'mdl02-pipelinedefinition-devrun'
        DEV_PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(DEV_GCS_BUCKET_NAME, DEV_PIPELINE_NAME)
        DEV_TRAIN_DATA_ROOT = 'gs://mlpipelineportfolio_bucket_01/for_production/tfrecord/plantvillage-train-15percent'
        DEV_TEST_DATA_ROOT = 'gs://mlpipelineportfolio_bucket_01/for_production/tfrecord/plantvillage-test-5percent'
        DEV_ENDPOINT_NAME = 'prediction-' + DEV_PIPELINE_NAME
        DEV_PIPELINE_DEFINITION_FILE = '/home/jupyter/SimpleMLPIpeline/uploadFromGitHub/pipeline_definition/dev/' + DEV_PIPELINE_NAME + '_pipeline.json'
        DEV_SERVICE_ACCOUNT = 'pj03-vertex-account@mlpipelineportfolio.iam.gserviceaccount.com'
        DEV_MODULE_FILE = 'gs://mlpipelineportfolio_bucket_01/for_production/modules/utils.py'
        
        # run and save definition of pipeline on devenv
        template_path = pipeline.save_pipeline_definition(pipeline_name=DEV_PIPELINE_NAME,
                                                          pipeline_root=DEV_PIPELINE_ROOT,
                                                          train_data_root=DEV_TRAIN_DATA_ROOT,
                                                          test_data_root=DEV_TEST_DATA_ROOT,
                                                          module_file=DEV_MODULE_FILE,
                                                          endpoint_name=DEV_ENDPOINT_NAME,
                                                          project_id=DEV_GOOGLE_CLOUD_PROJECT,
                                                          region=DEV_GOOGLE_CLOUD_REGION,
                                                          pipeline_definition_file=DEV_PIPELINE_DEFINITION_FILE)
        
        # submit the pipeline for testing on devenv
        aiplatform.init(project=DEV_GOOGLE_CLOUD_PROJECT, location=DEV_GOOGLE_CLOUD_REGION)
        
        job = pipeline_jobs.PipelineJob(template_path=template_path,
                                        display_name=DEV_PIPELINE_NAME)
        
        print('successfully created PipelineJob Instance')
        
        job.submit(service_account=DEV_SERVICE_ACCOUNT)
        
        print('submit complete')
        
        # wait for the result of the test
        job.wait()
        
        # check the result; '4' means 'success'
        pipeline_state = job.state
        print('pipelinestate: {}'.format(pipeline_state))