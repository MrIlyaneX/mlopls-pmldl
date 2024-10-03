set -e
cd /opt
mlflow server -h localhost -p 8090 &
sleep 10
export MLFLOW_TRACKING_URI=http://localhost:8090
cd /opt/code/models/
python training.py
mlflow models generate-dockerfile --model-uri "models:/BasicModel@Champion" -d ./mlflow_api --install-mlflow
#docker build -t mls mlflow_api