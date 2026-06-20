run:
	bash run.sh

prepare:
	python src/data_ingest.py

model:
	python src/modelling.py

stream:
	python src/growth_streaming.py

clean:
	rm -rf data/processed/* outputs/* models/* checkpoints/* data/stream/incoming/*
