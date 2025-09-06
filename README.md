# scGraphTrans

**scGraphTrans** (Single-cell Graph Transfer via Pseudolabel-Guided Graph Structure Learning and Domain Adaptation)is a graph-based framework developed for precise cell type annotation and interpretable tumor microenvironment (TME) modeling using single-cell RNA-seq (scRNA-seq) data. The framework combines Graph Structure Learning (GSL) with Graph Domain Adaptation (GDA) to jointly capture biological relevance and enhance cross-dataset generalization. By incorporating pathway-informed pseudolabels and bridging inter-sample graphs, scGraphTrans provides a robust solution for heterogeneous single-cell data analysis.

---

## Features
- **Pathway-guided Graph Structure Learning (GSL):** Refines both reference and query graphs by integrating 14 hallmark functional states (e.g., proliferation, stemness, EMT), enabling the graph to reflect biological function beyond gene expression similarity..
- **Graph Domain Adaptation (GDA):** Implements a knowledge-bridged learning strategy based on Bridged-GNN to enable accurate label transfer across datasets without requiring target annotations.
- **Two-stage pipeline:** The GSL module learns biologically meaningful graph structures guided by pathway pseudolabels; the GDA module then leverages these structures to perform source-supervised cell type annotation via cross-domain message passing.

---

## Project Structure
```
scGraphTrans/
│
├── data/                     # Input scRNA-seq datasets (expression matrices, labels)
├── GSL/                      # Graph Structure Learning module
│   ├── train_gsl.py          # Main entry point for GSL
│   └── ...
├── GDA/                      # Graph Domain Adaptation module
│   ├── GSE132509/            # Dataset-specific GDA scripts
│   │   └── AYL050-OX1164/    # GDA experiments for AYL050 -> OX1164
│   │       └── run_gda.py    # Main script for GDA
│   └── ...
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/scGraphTrans.git
   cd scGraphTrans
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux / Mac
   venv\Scripts\activate       # Windows

   pip install -r requirements.txt
   ```

---

## Usage

### Step 1: Graph Structure Learning (GSL)
Run the following command to generate an optimized graph:
```bash
cd GSL
python train_gsl.py
```
This step constructs and refines a cell–cell graph using functional pseudo-labels.

---

### Step 2: Graph Domain Adaptation (GDA)
Use the GSL output along with reference data for cell type annotation:
```bash
cd ../GDA/AYL050-OX1164
python run_gda.py
```
This step applies the **KBL** algorithm to transfer knowledge from the reference dataset to the query dataset.

---

## Example Workflow
```
1. Prepare your dataset in the data/ directory.
2. Run the GSL module to generate a refined cell–cell graph.
3. Use the GSL results as input to the GDA module.
4. Obtain final annotated labels for the query dataset.
```

---

## Requirements
The required packages are listed in `requirements.txt`:
```
python>=3.8
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.0
tqdm>=4.62.0
torch>=1.13.1
torchvision>=0.14.0
torchaudio>=0.13.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
torch-sparse>=0.6.15
torch-cluster>=1.6.0
torch-spline-conv>=1.2.1
networkx>=2.6
numba>=0.53.0
texttable>=1.6.2
Pillow>=8.3.0
cvxpy>=1.2.1
scanpy>=1.9.1
seaborn>=0.11.2
```


## scGraphTrans: Pathway-Guided Graph Learning and Domain Adaptation for Cell Type Annotation in Single-Cell RNA-seq

Yue-Chao Li1, Xinyuan Li1, Meng-Meng Wei3, Xin-Fei Wang4, Jie Pan5,6, Zhonghao Ren7, Zhi-An Huang8, Zhu-Hong You1,*, Yu-An Huang1,2,*
```
1 School of Computer Science, Northwestern Polytechnical University, Shaanxi 710129, China
2 Research & Development Institute of Northwestern Polytechnical University in Shenzhen, Shenzhen, 518063, China 
3 School of Computer Science and Technology, China University of Mining and Technology, Xuzhou 221116
4 College of Computer Science and Technology, Jilin University, Changchun, 130012, China.
5 Institute of Urology, The Third Affiliated Hospital of Shenzhen University, Shenzhen 518000, People's Republic of China.
6 School of Biomedical Engineering & Suzhou Institute for Advanced Research, University of Science and Technology of China, Suzhou, China
7 College of Computer Science and Electronic Engineering, Hunan University, Changsha, 410082, China
8 Research Office, City University of Hong Kong (Dongguan), Dongguan, 523000, China
*corresponding authors

```


