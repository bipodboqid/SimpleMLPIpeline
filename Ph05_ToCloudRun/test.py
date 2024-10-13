import tensorflow as tf
import base64
from google.cloud import aiplatform

ENDPOINT_ID = '1706270522494418944'
GOOGLE_CLOUD_REGION = 'asia-northeast1'
GOOGLE_CLOUD_PROJECT = 'mlpipelineportfolio'
image_path = 'mnist_156.jpg'
print('import and constants configuration ended')

def create_example(image_path):
    features = {
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[
                tf.io.read_file(image_path).numpy()
            ])),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    examples = example_proto.SerializeToString()
    return base64.b64encode(examples).decode()

def endpoint_predict_sample(
    project: str, location: str, instances: list, endpoint: str
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint)

    prediction = endpoint.predict(instances=instances)
    print(prediction)
    return prediction

if __name__ == '__main__':
	instances = [
		{
			'b64': create_example(image_path),
		},
	]

	result = endpoint_predict_sample(project=GOOGLE_CLOUD_PROJECT,
							location=GOOGLE_CLOUD_REGION,
							instances=instances,
							endpoint=ENDPOINT_ID)
