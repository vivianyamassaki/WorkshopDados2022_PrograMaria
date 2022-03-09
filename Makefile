IMAGE = programaria/workshop-mlops

run:
	python predictor/api.py

build:
	sudo docker image build -t ${IMAGE} .

docker_run: build
	sudo docker run -d --name mlops -it --net=\"host\" --restart=unless-stopped ${IMAGE}