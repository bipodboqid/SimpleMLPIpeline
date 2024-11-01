import tensorflow as tf
from tfx import v1 as tfx
import kfp
import tensorflow_model_analysis as tfma

import os

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import logging

from tfx.components import ImportExampleGen, StatisticsGen, SchemaGen, Transform, Trainer

import os

def _create_pipeline(pipeline_name: str, pipeline_root: str,
                     train_data_root: str, test_data_root: str,
                     module_file: str, endpoint_name: str, project_id: str,
                     region: str) -> tfx.dsl.Pipeline:
	
	example_gen = ImportExampleGen(input_base=train_data_root)
	
	statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
	
	schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
						   infer_feature_shape=False)
	
	transform = Transform(examples=example_gen.outputs['examples'],
						  schema=schema_gen.outputs['schema'],
						  module_file=module_file)
	
	trainer = Trainer(
		module_file=module_file,
		examples=example_gen.outputs['examples'],
		schema=schema_gen.outputs['schema'],
		transform_graph=transform.outputs['transform_graph'],
		train_args=tfx.proto.TrainArgs(num_steps=24),
		eval_args=tfx.proto.EvalArgs(num_steps=22))
	
	model_resolver = tfx.dsl.Resolver(
            strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
            model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
            model_blessing=tfx.dsl.Channel(
                    type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
            'latest_blessed_model_resolver')
	
	example_gen_for_eval = ImportExampleGen(
            input_base=test_data_root).with_id('ImportExampleGenForEvaluator')
	
	eval_config = tfma.EvalConfig(
            model_specs=[tfma.ModelSpec(label_key='label')],
            slicing_specs=[tfma.SlicingSpec()],
            metrics_specs=tfma.metrics.default_binary_classification_specs())
	
	evaluator = tfx.components.Evaluator(
            examples=example_gen_for_eval.outputs['examples'],
            model=trainer.outputs['model'],
            baseline_model=model_resolver.outputs['model'],
            eval_config=eval_config)
	
	vertex_serving_spec = {
		'project_id': project_id,
		'endpoint_name': endpoint_name,
		'machine_type': 'n1-standard-4'}
	
	serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
	
	pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
		model=trainer.outputs['model'],
		custom_config={
			tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:True,
			tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:region,
			tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:serving_image,
			tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:vertex_serving_spec})
	
	components = [
		example_gen,
		statistics_gen,
		schema_gen,
		transform,
		trainer,
		model_resolver,
		example_gen_for_eval,
		evaluator,
		pusher
	]
	
	return tfx.dsl.Pipeline(
		pipeline_name=pipeline_name,
		pipeline_root=pipeline_root,
		components=components)

def save_pipeline_definition(pipeline_name: str, pipeline_root: str, 
                             train_data_root: str, test_data_root: str,
                             module_file: str, endpoint_name: str, project_id: str,
                             region: str, pipeline_definition_file: str) -> str:
	
	runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
		config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
		output_filename=pipeline_definition_file)
	
	_ = runner.run(
		_create_pipeline(
			pipeline_name=pipeline_name,
			pipeline_root=pipeline_root,
			train_data_root=train_data_root,
			test_data_root=test_data_root,
			module_file=module_file,
			endpoint_name=endpoint_name,
			project_id=project_id,
			region=region))
	
	return pipeline_definition_file
