install:
	uv sync

build: install
	mkdir -p output
	uv run rendercv render "Pascal_Brokmeier_CV.yaml" --pdf-path output/pascal_brokmeier_cv_long.pdf 
	uv run rendercv render "Pascal_Brokmeier_short_CV.yaml" --pdf-path output/pascal_brokmeier_cv_short.pdf 

watch: install
	uv run rendercv render --watch "Pascal_Brokmeier_CV.yaml"
