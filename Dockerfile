# Exact environment for the benchmark framework.
# The pinned versions in requirements.txt (torch 1.13.1, gym 0.18.0,
# numpy 1.23.5) require Python 3.8.16 — this image reproduces that environment
# so the code runs exactly as it did for the papers.
FROM python:3.8.16-slim

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e ./ic3net-envs

# visdom is only needed for the optional --plot live dashboards.
CMD ["bash"]
