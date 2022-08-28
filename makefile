run-docker:
	docker run -it -p 8092:8092 text_analysis_service

run-docker-debug:
	docker run -it -p 8092:8092 text_analysis_service_debug

create-docker-debug:
	docker build -f Dockerfile-local -t text_analysis_service_debug .

create-docker:
	docker build -t text_analysis_service .
run:
	export FLASK_DEBUG=1 && flask run
run-eb:
	eb local run --port 8094

run-bash-docker:
	docker run --rm -it --entrypoint bash text_analysis_service

run-inside-docker:
	python3 app.py -dd

run-inside-docker-production:
	python3 app.py -p

install:
	pip3 install -r requirement.txt
	python3 -m spacy download fr_core_news_sm
	python3 init.py

deploy:
	rm -r ./models/*
	eb deploy