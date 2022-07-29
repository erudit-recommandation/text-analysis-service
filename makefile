run-docker:
	docker run -it -p 8092:8092 text_analysis_service

create-docker:
	docker build -t text_analysis_service .
run:
	export FLASK_DEBUG=1
	flask run
run-eb:
	eb local run --port 8094

run-bash-docker:
	docker run --rm -it --entrypoint bash text_analysis_service