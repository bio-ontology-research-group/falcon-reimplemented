# Causal-FALCON: Causal and neuro-symbolic reasoning over Description Logic ontologies

## Requirements

- **Python**: 3.10 or higher recommended.
- **Python Packages**: Install required packages using pip:
  ```bash
  pip install -r requirements.txt
  ```
  *Note on PyTorch*: The `requirements.txt` file lists `torch==2.1.0`. You might need to install a specific version compatible with your hardware (CPU or CUDA version). Please refer to the official PyTorch installation guide: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- **Other Tools**:
    * groovy == 4.0.0 (or compatible)
    * JVM == 1.8.0_333 (or compatible Java 8+)
    * Protégé (https://protege.stanford.edu/) - For data preparation steps.


## Run

- **Single Fuzzy Model (using `train_single_model.py`)**
    > This script trains a single fuzzy interpretation based on provided TBox and ABox axioms in Functional Syntax NNF.

    > `cd ./scripts`

    > Example using Pizza ontology data (assuming TBox and ABox files exist):
    ```bash
    python train_single_model.py \
        --tbox_path ../data/Pizza/pizzaTBox.txt \
        --abox_path ../data/Pizza/pizzaABox.txt \
        --output_dir . \
        --embedding_dim 128 \
        --lr 0.001 \
        --epochs 10 \
        --output_membership_filename pizza \
        --membership_format tsv
    ```
    > *Note*: The example above requires `pizzaTBox.txt` and `pizzaABox.txt`. The `pizzaTBox.txt` can be generated using the steps in "Data Preparation". An `pizzaABox.txt` file containing ABox axioms in Functional Syntax NNF needs to be provided separately. The script will output the trained model (`.pth`) and the membership matrix (`pizza.tsv` in this example) to the specified `output_dir` (the `scripts` directory in this case).

- **Original Models (Family, Pizza, HPO)**
    - Family Ontology
        > `cd ./code/model`

        > `python family.py`
    - Pizza Ontology
        > `cd ./code/model`

        > `python pizza.py`
    - Human Phenotype Ontology
        > `cd ./data/HPO/ && unzip BIOGRID-ALL-4.4.211.tab.zip && cd ../../code/model`

        > `sh run_hpo.sh`

## Data Preparation
We elaborate the steps of data preparation to foster further research. This section is unnecessary for running the experiments unless generating input files for `train_single_model.py`.

- **Human Phenotype Ontology (HPO)**
    - Download datasets
        > `cd ./data/HPO/`

        > `wget http://purl.obolibrary.org/obo/hp.owl`

        > `wget https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-4.4.211/BIOGRID-ALL-4.4.211.tab.zip`

        > `unzip ./BIOGRID-ALL-4.4.211.tab.zip` # Adjusted path assuming unzip in current dir

        > `wget http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt`
    - Semantic Entailment (Generating the True Testing Axioms)
        > Open `hp.owl` with the graphical interface of `Protégé`

        > Select `ELK` as the logical reasoner

        > Save the inferred ontology as `hpInferred.owl`
    - OWL to Axioms (Functional Syntax)
        > *Note: These Groovy scripts convert OWL to a specific axiom format. Ensure the output is suitable or convert it to Functional Syntax NNF if needed for `train_single_model.py`.*
        > `groovy ../../code/ppc/GetTBox.groovy ./hp.owl > ./TBox.txt` # Adjusted path

        > `groovy ../../code/ppc/GetTBox.groovy ./hpInferred.owl > ./TBoxInferred.txt` # Adjusted path
        > *(ABox generation might require different tools or scripts)*

- **Pizza Ontology**
    - Download datasets
        > `cd ./data/Pizza/`

        > `wget https://protege.stanford.edu/ontologies/pizza/pizza.owl`

    - Semantic Entailment (Generating the True Testing Axioms)
        > Open `pizza.owl` with the graphical interface of `Protégé`

        > Select `HermiT` as the logical reasoner

        > Save the inferred ontology as `pizzaInferred.owl`

    - OWL to Axioms (Functional Syntax)
        > *Note: These Groovy scripts convert OWL to a specific axiom format. Ensure the output is suitable or convert it to Functional Syntax NNF if needed for `train_single_model.py`.*
        > `groovy ../../code/ppc/GetTBox.groovy ./pizza.owl > ./pizzaTBox.txt` # Adjusted path

        > `groovy ../../code/ppc/GetTBox.groovy ./pizzaInferred.owl > ./pizzaTBoxInferred.txt` # Adjusted path
        > *(ABox generation might require different tools or scripts, e.g., extracting ClassAssertion, ObjectPropertyAssertion)*
