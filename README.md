# Dynamic Metapath Discovery via Explainability-Guided Attention

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A novel framework for automatic metapath discovery in heterogeneous graph neural networks using explainability-guided attention mechanisms. Implemented and tested on the DBLP dataset for node classification.

## Problem Statement
Traditional HGNNs require manual metapath definition by domain experts, leading to:
- Domain knowledge dependency
- Limited adaptability
- Scalability issues
- Potential selection bias

## Proposed Solution
Dynamic metapath discovery through:
- **Relation-level attention**: Adaptive relation importance scoring
- **Metapath-level attention**: Automatic metapath ranking
- **Explainability module**: Post-hoc metapath visualization

## Key Features
- Dynamic metapath generation 
- Hierarchical attention mechanisms
- Seed node initialization (paper/author/conference)
- Edge type filtering to prevent loops
- Softmax normalization within path length groups

## Core components:
- `MetapathGenerator`: Handles recursive path expansion
- `RelationalAttention`: Implements scaled dot-product attention
- `DynamicMetapath`: Main classifier integrating components

## Dataset (DBLP)
Node types:
- Author (4,057 nodes)
- Paper (14,328 nodes)
- Term (7,723 nodes)
- Conference (20 nodes)

Relations:
- Author-Paper (A-P)
- Paper-Author (P-A)
- Paper-Term (P-T)
- Term-Paper (T-P)
- Paper-Conference (P-C)
- Conference-Paper (C-P)

## Usage
1. Install dependencies
2. Load data and initialize model
3. Train model
4. Analyze results using get_top_metatpaths method

## Results
Achieved 87% validation accuracy on DBLP node classification. Discovered metapaths include:
paper -> paper -> conference | Score: 0.2000
conference -> conference -> paper | Score: 0.2000
paper -> paper -> author | Score: 0.2000


## Implementation Note
Full implementation and training code available in [final.ipynb](final.ipynb)

## Future Work
- Extend to other heterogeneous graphs
- Implement multi-head attention
- Develop hierarchical metapath learning
- Add link prediction capabilities
