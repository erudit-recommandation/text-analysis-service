run-docker:
	docker run -it -p 8092:8092 text_analysis_service

create-docker:
	docker build -t text_analysis_service .
run:
	export FLASK_DEBUG=1
	flask run