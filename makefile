run-docker:
	docker run -it -p 5000:5000 text_analysis_service

create-docker:
	docker build -t text_analysis_service .