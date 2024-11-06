import base64
import json
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from google.cloud import scheduler_v1
import os
from google.cloud import bigquery

PROD_GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
PROD_GOOGLE_CLOUD_REGION = 'asia-northeast1'
PROD_SERVICE_ACCOUNT = 'pj03-vertex-account@mlpipelineportfolio.iam.gserviceaccount.com'
template_path='gs://mlpipelineportfolio_bucket_01/for_production/upload_from_github/pipeline_definition/prod/mdl04-pipelineautomation1730457948.412411_pipeline.json'
dataset_id = "dimension_tables"

def Mdl04_PipelineAutomation_update_BQtables(event, context):
    # メッセージデータの取得
    if 'data' in event:
        payload = base64.b64decode(event['data']).decode('utf-8')
        data = json.loads(payload)
        # ここで渡されたデータ (data) を利用
        print(f"Received data: {data}")
        pipelinejob_id = data['pipelinejob_id']
        schedulerjob_id = data['schedulerjob_id']
        git_commit = data['git_commit']

        # PipelineJobを得る
        pipeline_job = _get_pipeline_job(pipelinejob_id)
        pipeline_job_details = pipeline_job.gca_resource
        pipeline_run_name = pipeline_job_details.name
        
        pipeline_state = pipeline_job.state
        if pipeline_state == 4:
            print('Job Succeeded')
            model_full_id = _get_pushed_model_uri(pipeline_job_details)
            print(model_full_id)
            if model_full_id:
                endpoint_model_details = _get_endpoint_model_details(model_full_id)
                destination_endpoint = endpoint_model_details['destination_endpoint']
                endpoint_create_time = endpoint_model_details['endpoint_create_time']
                endpoint_update_time = endpoint_model_details['endpoint_update_time']
                endpoint_display_name = endpoint_model_details['endpoint_display_name']
                model_create_time = endpoint_model_details['model_create_time']
                model_display_name = endpoint_model_details['model_display_name']
                deployed_model_id = endpoint_model_details['deployed_model_id']
            else:
                print('cannot get model_full_id')

            evaluation_uri = _get_evaluation_uri(pipeline_job_details)
            if evaluation_uri:
                auc_score = _get_auc_score(evaluation_uri)
            else:
                print('cannot get evaluation_uri')

            if (model_full_id and destination_endpoint and auc_score):
                model_id = deployed_model_id
                endpoint_id = _obtain_column_values_endpointid(destination_endpoint)
                auc = _obtain_column_values_auc(auc_score)
            else:
                print('required value missing')

            client = bigquery.Client(project=PROD_GOOGLE_CLOUD_PROJECT)

            # 読み込みたいテーブルの情報
            deployment_table_ref = f"{PROD_GOOGLE_CLOUD_PROJECT}.{dataset_id}.dim_deployments"
            model_table_ref = f"{PROD_GOOGLE_CLOUD_PROJECT}.{dataset_id}.dim_models"
            endpoint_table_ref = f"{PROD_GOOGLE_CLOUD_PROJECT}.{dataset_id}.dim_endpoints"

            if not _check_endpoint_exist(endpoint_id, client, endpoint_table_ref):
                _insert_endpoint(endpoint_id, endpoint_display_name, endpoint_create_time,
                                 endpoint_update_time, client, endpoint_table_ref)

            if not _check_model_exist(model_id, client, model_table_ref):
                _insert_model(model_id, model_display_name, model_create_time,
                              auc, git_commit, client, model_table_ref)

            if not _check_deployment_exist(endpoint_id, model_id, client, deployment_table_ref):
                _insert_deployment(model_id, endpoint_id, pipeline_run_name,
                                   client, deployment_table_ref)
            
            _delete_scheduler_job(schedulerjob_id)
            
            # デプロイされたモデルの推論結果を監視するためのCloud Schedulerジョブを作成
            # ジョブに渡すペイロードの設定
            project = 'mlpipelineportfolio'
            location = 'asia-northeast1'
            topic = "projects/mlpipelineportfolio/topics/Mdl05_ModelMonitoring_run-monitoring-function"
            job_name = "mdl05-modelmonitoring-cloudscheduler-for-monitoring-function"
            schedulerjob_id = f"projects/{project}/locations/{location}/jobs/{job_name}"
            baseline_unwell_rate = 0.628

            payload = json.dumps({"baseline_unwell_rate": baseline_unwell_rate, "deployed_model_id": model_id}).encode("utf-8")
            encoded_payload = base64.b64encode(payload).decode("utf-8")

            # ジョブの設定
            job = {
                "name": schedulerjob_id,
                "pubsub_target": {
                    "topic_name": topic,
                    "data": encoded_payload,
                },
                "schedule": "0 0 * * *",  # 毎日UTC0時に実行
                "time_zone": "UTC",
            }

            # ジョブの作成
            client = scheduler_v1.CloudSchedulerClient()
            client.create_job(parent=f"projects/{project}/locations/{location}", job=job)
            
        elif pipeline_state == 5 or pipeline_state == 7:
            print('Job Canceled or Failed')
            _delete_scheduler_job(schedulerjob_id)
        else:
            print('Job not finished')
    else:
        print("No data found in Pub/Sub message")

def _delete_scheduler_job(schedulerjob_id):
    client = scheduler_v1.CloudSchedulerClient()
    client.delete_job(name=schedulerjob_id)
    print(f"Deleted job: {schedulerjob_id}")

def _get_pipeline_job(pipelinejob_id):
    aiplatform.init(project=PROD_GOOGLE_CLOUD_PROJECT, location=PROD_GOOGLE_CLOUD_REGION)
    job = pipeline_jobs.PipelineJob(template_path=template_path, display_name='dummy')
    return job.get(resource_name=pipelinejob_id)

def _get_pushed_model_uri(pipeline_job_details):
    print('getting model_full_id')
    model_full_id = None
    for task in pipeline_job_details.job_detail.task_details:
        if task.task_name == 'Pusher':
            pusher = task
            break
    model_full_id = pusher.outputs['pushed_model'].artifacts[0].metadata['pushed_destination']
    print(f'model_full_id: {model_full_id}')
    return model_full_id

def _get_endpoint_model_details(model_full_id):
    print('getting destination_endpoint')
    newest_endpoint = aiplatform.Endpoint.list(order_by='update_time')[-1]  
    destination_endpoint = None
    for model in newest_endpoint.list_models():
        if model.model == model_full_id:
            destination_endpoint = newest_endpoint.resource_name
            deployed_model = model
            break
    print(f'destination_endpoint: {destination_endpoint}')
    return {'destination_endpoint': destination_endpoint,
            'endpoint_create_time': newest_endpoint.create_time,
            'endpoint_update_time': newest_endpoint.update_time,
            'endpoint_display_name': newest_endpoint.display_name,
            'deployed_model_id' : deployed_model.id,
            'model_create_time': deployed_model.create_time,
            'model_display_name': deployed_model.display_name}

def _get_evaluation_uri(pipeline_job_details):
    evaluation_uri = None
    for task in pipeline_job_details.job_detail.task_details:
        if task.task_name == 'Evaluator':
            evaluation_uri = task.outputs['evaluation'].artifacts[0].uri
            break
    print(f"evaluation_uri: {evaluation_uri}")
    return evaluation_uri

def _get_auc_score(evaluation_uri):
    import tensorflow_model_analysis as tfma
    auc_score = None
    evaluation_result = tfma.load_eval_result(evaluation_uri)
    auc_score = evaluation_result.get_metrics_for_all_slices()[()]['auc']['doubleValue']
    print(f'auc_score: {auc_score}')
    return auc_score
    
def _obtain_column_values_endpointid(destination_endpoint):
    destination_list = destination_endpoint.split(sep='/')
    destination_list[1] = 'mlpipelineportfolio'
    endpoint_id = os.sep.join(destination_list)
    return endpoint_id

def _obtain_column_values_auc(auc_score):
    return float(auc_score)

def _check_endpoint_exist(endpoint_id, client, endpoint_table_ref):
    query = f'SELECT endpoint FROM `{endpoint_table_ref}`;'
    query_job = client.query(query)
    rows = query_job.result()
    for row in rows:
        if row.endpoint == endpoint_id:
            return True
    return False

def _check_model_exist(model_id, client, model_table_ref):
    query = f'SELECT deployed_model_id FROM `{model_table_ref}`;'
    query_job = client.query(query)
    rows = query_job.result()
    for row in rows:
        if row.deployed_model_id == model_id:
            return True
    return False

def _check_deployment_exist(endpoint_id, model_id, client, deployment_table_ref):
    query = f'SELECT deployed_model_id, endpoint FROM `{deployment_table_ref}`;'
    query_job = client.query(query)
    rows = query_job.result()
    for row in rows:
        if row.endpoint == endpoint_id and row.deployed_model_id == model_id:
            return True
    return False

def _insert_endpoint(endpoint, display_name, created_at, updated_at,
                     client, endpoint_table_ref):
    rows_to_insert = []
    new_row = {
    "endpoint": endpoint,
    "display_name": display_name[0],
    "created_at": _convert_to_bigquery_timestamp(created_at[0]),
    "updated_at": _convert_to_bigquery_timestamp(updated_at[0])
    }
    rows_to_insert.append(new_row)

    errors = client.insert_rows_json(endpoint_table_ref, rows_to_insert)
    if errors:
        print("Errors occurred while inserting rows:", errors)
    else:
        print("Rows inserted into dim_endpoints successfully.")

def _insert_model(deployed_model_id, display_name, created_at,
                  metrics_auc, git_commit, client, model_table_ref):
    rows_to_insert = []
    new_row = {
    "deployed_model_id": deployed_model_id,
    "display_name": display_name,
    "created_at": _convert_to_bigquery_timestamp(created_at),
    "metrics_auc": metrics_auc,
    "git_commit": git_commit
    }
    rows_to_insert.append(new_row)

    errors = client.insert_rows_json(model_table_ref, rows_to_insert)
    if errors:
        print("Errors occurred while inserting rows:", errors)
    else:
        print("Rows inserted into dim_models successfully.")

def _insert_deployment(deployed_model_id, endpoint, pipeline_run_name,
                       client, deployment_table_ref):
    rows_to_insert = []
    new_row = {
    "deployed_model_id": deployed_model_id,
    "endpoint": endpoint,
    "pipeline_run_name": pipeline_run_name
    }
    rows_to_insert.append(new_row)

    errors = client.insert_rows_json(deployment_table_ref, rows_to_insert)
    if errors:
        print("Errors occurred while inserting rows:", errors)
    else:
        print("Rows inserted into dim_deployments successfully.")  

def _convert_to_bigquery_timestamp(datetime_with_nano):
    from google.protobuf.timestamp_pb2 import Timestamp
    datetime_rounded = datetime_with_nano.replace(microsecond=int(datetime_with_nano.microsecond))
    return datetime_rounded.isoformat()