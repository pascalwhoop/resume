build:
	docker build -t resume .

.ONESHELL:
pdf: build
	rm -rf public/resume.pdf
	node yaml2json.js
	docker run --rm\
		-v ${PWD}/public:/app/public/ \
		-v ${PWD}/resume.json:/app/resume.json \
		--name resume \
		resume export --format pdf --theme ./dependencies/jsonresume-theme-macchiato public/resume.pdf
	# if not on CI, open the pdf
	if [ -z ${CI} ]; then open public/resume.pdf; fi
	

serve: build
	docker run --rm\
		-v ${PWD}/public:/app/public/ \
		-v ${PWD}/theme:/app/theme/ \
		-v ${PWD}/resume.yaml:/app/resume.yaml \
		--name resume \
		-p 4000:4000 \
		resume serve --theme ./theme/jsonresume-theme-macchiato

stop: 
	docker stop resume
	docker rm resume

bash: build
	docker run --rm -it --entrypoint /bin/sh resume