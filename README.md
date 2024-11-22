## To do
- lora (torchtune?)
- workflow (metaflow?)
- lr scheduling
- freezing
- mlflow dataset tracking

# Running
`mlflow ui --waitress-opts '--threads=6 --send-bytes=9000 --max-request-body-size=2000000000 --expose-tracebacks' --backend-store-uri file:mlruns --artifacts-destination mlartifacts`

### Dagster
```dagster dev```