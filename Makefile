train-all: train-1d-cnn train-mtt train-tcn train-mtc

# Data preprocessing
preprocess:
	python -m preprocess

# Model training
train-1d-cnn:
	python -m models.one_d_cnn

train-mtt:
	python -m models.mtt

train-tcn:
	python -m models.tcn

train-mtc:
	python -m models.mtc
