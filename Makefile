.PHONY: install train app test clean

install:
	pip install -r requirements.txt

train:
	python main.py

train-advanced:
	python main.py --epochs 5 --batch-size 16

app:
	streamlit run app/streamlit_app.py

test:
	python -m pytest tests/ -v --tb=short

test-all:
	python -m pytest tests/ -v

clean:
	rm -rf models/ results/ logs/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

lint:
	python -m py_compile main.py
	python -m py_compile app/streamlit_app.py
	python -m py_compile src/config.py
	python -m py_compile src/data/loader.py
	python -m py_compile src/model/architecture.py
	python -m py_compile src/training/trainer.py
