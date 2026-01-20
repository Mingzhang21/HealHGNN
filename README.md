# Heterophily-Agnostic-Hypergraph-Neural-Networks-with-Riemannian-Adaptive-Exchanger
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

## Getting Started

### Dependency

To run our code, the following Python libraries which are required to run our code:
```
python=3.12
pytorch-cuda=12.1
pytorch=2.3.0
torch-geometric
torch-scatter
torch-sparse
torch-cluster
```
### Data Preparation
Download the main datasets from the [HuggingFace Hub](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset).
Then put the downloaded directory under the root folder of HealHGNN. The directory structure should look like:

```
HealHGNN/
    <source code files>
    ...
    raw_data/
        coauthor_cora
        coauthor_dblp
        citeseer
        house-committees
        walmart-trips
        senate-committees
        ...
```

## Training
This is a sample implementation of HealHGNN.
Simply run the command
```shell
python main.py --task NC --dataset senate-committees
```
