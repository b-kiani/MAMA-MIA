FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /work
COPY . /work
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["python", "-m", "mmonc.pipeline", "--synthetic", "--out", "results/synthetic_smoke"]
