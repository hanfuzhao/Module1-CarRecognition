.PHONY: install data train app docker clean

install:        ## install dependencies
	pip install -r requirements.txt

data:           ## download dataset into data/raw + write metadata
	python scripts/make_dataset.py

train:          ## train all three models and run experiments
	python setup.py

app:            ## run the web app at http://localhost:5000
	python main.py

docker:         ## build and run the app in Docker
	docker build -t car-recognition .
	docker run -p 7860:7860 car-recognition

clean:          ## remove cached features and outputs
	rm -f data/processed/*.npy data/outputs/*.png data/outputs/*.json
