import base64
import functions_framework
import json
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import requests
from google.cloud import scheduler_v1
from google.protobuf import timestamp_pb2

# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def Mdl04_PipelineAutomation_trigger_pipeline(cloud_event):
    
    # 更新されたファイルが.jsonならば取得
    file_name = _check_file_extension(cloud_event)
    
    if file_name: 
        # PipelineJobの作成・submitに必要な引数を設定
        template_path = 'gs://mlpipelineportfolio_bucket_01/' + file_name 
        pipeline_name = 'mdl04-pipelineautomation-prodrun'
        project = 'mlpipelineportfolio'
        location = 'asia-northeast1'
        service_account = 'pj03-vertex-account@mlpipelineportfolio.iam.gserviceaccount.com'
        
        # PipelineJobの作成・submit
        aiplatform.init(project=project, location=location)
        job = pipeline_jobs.PipelineJob(template_path=template_path, display_name=pipeline_name)
        print('successfully created pipeline job instance')
        job.submit(service_account=service_account)
        
        # パイプラインの実行状況を監視するCloud Schedulerを作成
        # ジョブに渡すペイロードの設定
        pipelinejob_id = job.gca_resource.name
        git_commit = _get_latest_commit('bipodboqid', 'SimpleMLPipeline')
        topic = "projects/mlpipelineportfolio/topics/Mdl04_PipelineAutomation_update-BQ-tables"
        job_name = "mdl04-pipelineautomation-cloudscheduler-for-bqupdate"
        schedulerjob_id = f"projects/{project}/locations/{location}/jobs/{job_name}"
        
        payload = json.dumps({"pipelinejob_id": pipelinejob_id, "schedulerjob_id": schedulerjob_id, "git_commit": git_commit}).encode("utf-8")
        encoded_payload = base64.b64encode(payload).decode("utf-8")

        # ジョブの設定
        job = {
            "name": schedulerjob_id,
            "pubsub_target": {
                "topic_name": topic,
                "data": encoded_payload,
            },
            "schedule": "*/15 * * * *",  # 15分ごとに実行
            "time_zone": "UTC",
        }

        # ジョブの作成
        client = scheduler_v1.CloudSchedulerClient()
        client.create_job(parent=f"projects/{project}/locations/{location}", job=job)
        
    else:
        print('function triggered but updated file was invalid')
    
def _check_file_extension(cloud_event):
    pubsub_data = base64.b64decode(cloud_event.data['message']['data'])
    data_json = json.loads(pubsub_data)
    file_name = data_json['name']
    print(file_name)
    if file_name.endswith('.json'):
        return file_name
    return False
    
def _get_latest_commit(owner, repo, branch='main'):
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
        response = requests.get(url)
        if response.status_code == 200:
            commit_data = response.json()
            return commit_data['sha']
        else:
            raise Exception(f"Failed to get latest commit: {response.status_code}")