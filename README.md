# Machine Learning for Web Log

```bash
python -m app.main inputs/sample.log outputs/sample.csv
```

```bash
docker build -t mlwl .
docker run --rm -v "$PWD/inputs:/app/inputs" -v "$PWD/outputs:/app/outputs" mlwl:latest inputs/sample.log outputs/sample.csv
```