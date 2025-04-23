install:
	pip install -r requirements.txt

evaluate:
	cd $(shell pwd) && python scripts/run_evaluation.py