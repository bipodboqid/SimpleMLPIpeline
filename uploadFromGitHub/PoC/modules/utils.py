from typing import List, Text
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow_hub as hub

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

_FEATURE_KEY = 'image_raw'
_LABEL_KEY = 'label'

_FEATURE_KEYS = ['image_raw',]

_TRAIN_BATCH_SIZE = 100
_EVAL_BATCH_SIZE = 10

_IMG_HEIGHT = 299
_IMG_WIDTH = 299

# TFX Transform will call this function
def preprocessing_fn(inputs):
	outputs = {}
	
	raw_image_dataset = tf.sparse.to_dense(inputs[_FEATURE_KEY])
	image_tensor = tf.map_fn(
		fn=lambda x : tf.image.resize(
			tf.io.decode_image(x[0], dtype=tf.float32, channels=3, expand_animations=False),
			[_IMG_HEIGHT, _IMG_WIDTH]),
		elems=raw_image_dataset,
		fn_output_signature=tf.TensorSpec((_IMG_HEIGHT, _IMG_WIDTH, 3), dtype=tf.float32, name=None),
		infer_shape = True)
	outputs[_FEATURE_KEY] = image_tensor
	label = tf.sparse.to_dense(inputs[_LABEL_KEY])
	with tf.init_scope():
		table_keys = ['[0]','[1]','[2]','[3]','[4]','[5]','[6]','[7]','[8]','[9]']
		initializer = tf.lookup.KeyValueTensorInitializer(
			keys=table_keys,
			values=tf.cast(tf.range(len(table_keys)), tf.int64),
			key_dtype=tf.string,
			value_dtype=tf.int64)
		table = tf.lookup.StaticHashTable(initializer, default_value=-1)
	label = table.lookup(inputs[_LABEL_KEY])
	label = tf.sparse.to_dense(label)
	label = tf.reshape(label, [-1])
	outputs[_LABEL_KEY] = label
	
	return outputs

def _apply_preprocessing(raw_features, tft_layer):
	transformed_features = tft_layer(raw_features)
	if _LABEL_KEY in raw_features:
		transformed_label = transformed_features.pop(_LABEL_KEY)
		return transformed_features, transformed_label
	else:
		return transformed_features, None

def _get_serve_tf_examples_fn(model, tf_transform_output):
	model.tft_layer = tf_transform_output.transform_features_layer()
	@tf.function(input_signature=[
	  tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
	])
	def serve_tf_examples_fn(serialized_tf_examples):
		feature_spec = tf_transform_output.raw_feature_spec()
		required_feature_spec = {
			k: v for k, v in feature_spec.items() if k in _FEATURE_KEYS
		}
		parsed_features = tf.io.parse_example(serialized_tf_examples,
											  required_feature_spec)
		transformed_features, _ = _apply_preprocessing(parsed_features,
													   model.tft_layer)
		return model(transformed_features)

	return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
			  data_accessor: tfx.components.DataAccessor,
			  tf_transform_output: tft.TFTransformOutput,
			  batch_size: int = 100) -> tf.data.Dataset:
	dataset = data_accessor.tf_dataset_factory(
		file_pattern,
		tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
		schema=tf_transform_output.raw_metadata.schema)

	transform_layer = tf_transform_output.transform_features_layer()
	def apply_transform(raw_features):
		return _apply_preprocessing(raw_features, transform_layer)

	return dataset.map(apply_transform).repeat()


def _freeze_model_by_percentage(model: tf.keras.Model, percentage: float):
	if percentage < 0 or percentage > 1:
		raise ValueError('Freeze percentage should between 0.0 and 1.0')
		
	if not model.trainable:
		raise ValueError(
			'The model is not trainable, please set model.trainable to True')
		
	num_layers = len(model.layers)
	num_layers_to_freeze = int(num_layers * percentage)
	for idx, layer in enumerate(model.layers):
		if idx < num_layers_to_freeze:
			layer.trainable = False
		else:
			layer.trainable = True

def _build_keras_model() -> tf.keras.Model:
	inputs = [
		keras.layers.Input(shape=(_IMG_HEIGHT, _IMG_WIDTH, 3), name=_FEATURE_KEY)
	]
	
	inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
	
	feature_extractor_model = inception_v3
	
	feature_extractor_layer = hub.KerasLayer(
		feature_extractor_model,
		input_shape=(_IMG_HEIGHT, _IMG_WIDTH, 3),
		trainable=False
	)
	
	model = tf.keras.Sequential([
		keras.layers.Input(shape=(_IMG_HEIGHT, _IMG_WIDTH, 3), name=_FEATURE_KEY),
		feature_extractor_layer,
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(10)
	])
	
	model.compile(
		optimizer=keras.optimizers.Adam(1e-2),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[keras.metrics.SparseCategoricalAccuracy()])
	
	model.summary(print_fn=logging.info)
	return model

# TFX Trainer will call this function
def run_fn(fn_args: tfx.components.FnArgs):
	tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
	
	train_dataset = _input_fn(
		fn_args.train_files,
		fn_args.data_accessor,
		tf_transform_output,
		batch_size=_TRAIN_BATCH_SIZE)
	eval_dataset = _input_fn(
		fn_args.eval_files,
		fn_args.data_accessor,
		tf_transform_output,
		batch_size=_EVAL_BATCH_SIZE)
	
	model = _build_keras_model()
	model.fit(
		train_dataset,
		steps_per_epoch=fn_args.train_steps,
		validation_data=eval_dataset,
		validation_steps=fn_args.eval_steps)
	
	signatures = {
		'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
	}
	model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)