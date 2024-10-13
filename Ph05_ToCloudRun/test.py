import tensorflow as tf
import base64
import os
from google.cloud import aiplatform
from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI アプリケーションの作成
app = FastAPI()

ENDPOINT_ID = '1706270522494418944'
GOOGLE_CLOUD_REGION = 'asia-northeast1'
GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
image_path = 'mnist_156.jpg'  # コンテナ内の画像パス

# Example作成関数
def create_example(image_path):
    features = {
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.io.read_file(image_path).numpy()])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    examples = example_proto.SerializeToString()
    return base64.b64encode(examples).decode()

# 推論関数
def endpoint_predict_sample(project: str, location: str, instances: list, endpoint: str):
    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint(endpoint)
    prediction = endpoint.predict(instances=instances)
    return prediction

# リクエスト用のデータモデル
class PredictionRequest(BaseModel):
    image_path: str

# GETリクエストに対応するエンドポイントの作成
@app.get("/test-predict")
async def predict():
    try:
        instances = [
            {
                'b64': create_example(image_path),  # コンテナ内の画像を使用
            },
        ]
        result = endpoint_predict_sample(
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_REGION,
            instances=instances,
            endpoint=ENDPOINT_ID
        )
        return {"predictions": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    import uvicorn
    # 環境変数 PORT からポート番号を取得 (デフォルトは8080)
    port = int(os.environ.get("PORT", 8080))
    # UvicornでFastAPIアプリケーションを実行
    uvicorn.run(app, host="0.0.0.0", port=port)