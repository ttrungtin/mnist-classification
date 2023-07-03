project_up:
	docker compose -f docker/docker-compose.yml up -d

run_model:
	python -m src.model_trainer

run_main:
	python -m src.main -c config/train_base.yml

run_tensorboard:
	tensorboard --logdir log/