import base64
import json
from google.cloud import bigquery
from google.cloud import pubsub_v1
from datetime import datetime, timedelta

PROD_GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
dataset_id = "fact_tables"
alert_threshold = 0.3

def Mdl05_ModelMonitoring_monitor_BQtable(event, context):
    # メッセージデータの取得
    if 'data' in event:
        payload = base64.b64decode(event['data']).decode('utf-8')
        data = json.loads(payload)
        # ここで渡されたデータ (data) を利用
        print(f"Received data: {data}")
        baseline = data['baseline_unwell_rate']
        model_id = data['deployed_model_id']

        # 読み込みたいテーブルの情報
        requests_table_ref = f"{PROD_GOOGLE_CLOUD_PROJECT}.{dataset_id}.fact_requests"
        
        client = bigquery.Client(project=PROD_GOOGLE_CLOUD_PROJECT)

        ten_days_ago = datetime.utcnow() - timedelta(days=10)

        # クエリの作成
        query = f"""
            SELECT
                response_payload
            FROM
                `{requests_table_ref}`
            WHERE
                logging_time >= TIMESTAMP("{ten_days_ago.strftime('%Y-%m-%d %H:%M:%S')}")
                AND deployed_model_id = "{model_id}"
        """

        # クエリの実行
        query_job = client.query(query)
        results = query_job.result()

        # 条件に合致するレコードのカウントと、0.5より大きいもののカウント
        total_count = 0
        greater_than_0_5_count = 0

        for row in results:
            # response_payloadから推論結果のfloatを取得
            payload = float(row.response_payload[0].lstrip('[').rstrip(']')) 
            
            total_count += 1
            if payload > 0.5:
                greater_than_0_5_count += 1

        # 0.5より大きいものの率を計算
        if total_count > 0:
            rate = greater_than_0_5_count / total_count
            print(f"ratio of data predicted as unwell: {rate:.2%}")
            if abs(rate - baseline) > alert_threshold:
                message = {
                    "alert": "モデル監視アラート",
                    "details": f"モデル: {model_id} ベース陽性率: {baseline} 推論結果の10日間平均陽性率: {rate}"
                }
                TOPIC_ID = 'Mdl05_ModelMonitoring_send-alert'
                publish_message(message, PROD_GOOGLE_CLOUD_PROJECT, TOPIC_ID)
        else:
            print(f'no request sent to deployed_model_id: {model_id} yet')

        

            
def publish_message(message, PROJECT_ID, TOPIC_ID):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    future = publisher.publish(topic_path, json.dumps(message).encode("utf-8"))
    future.result()