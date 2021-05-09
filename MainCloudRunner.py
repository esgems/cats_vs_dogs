import tensorflow_cloud as tfc

tfc.run(
	entry_point="Main.py",
	docker_image_bucket_name="projectcatsbucket",
	distribution_strategy='auto',
	requirements_txt='requirements.txt',
	chief_config=tfc.MachineConfig(
			cpu_cores=8,
			memory=30,
			accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_P100,
			accelerator_count=2),
	worker_count=0)

# gsutil -m cp "gs://projectcatsbucket/checkpoints/checkpoint" "gs://projectcatsbucket/checkpoints/train_60.tf.data-00000-of-00001" "gs://projectcatsbucket/checkpoints/train_60.tf.index" .
# gsutil -m cp "gs://projectcatsbucket/saved_models/modelVgg/checkpoint" "gs://projectcatsbucket/saved_models/modelVgg/modelVgg.data-00000-of-00001" "gs://projectcatsbucket/saved_models/modelVgg/modelVgg.index" .
# gsutil -m cp "gs://projectcatsbucket/saved_models/modelVgg/checkpoint" "gs://projectcatsbucket/saved_models/modelVgg/modelVgg.data-00000-of-00001" "gs://projectcatsbucket/saved_models/modelVgg/modelVgg.index" .