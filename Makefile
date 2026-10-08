.PHONY: setup run test clean

setup:
	python -m pip install -r requirements.txt

run:
	python run.py

test:
	python -m pytest -q

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['reports/figures','reports/tables']]"
