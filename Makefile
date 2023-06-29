project_up:
	docker compose -f docker/docker-compose.yml up -d

run:
	python -m src.model_trainer

run_tensorboard:
	tensorboard --logdir log/