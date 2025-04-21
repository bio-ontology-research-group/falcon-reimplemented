import torch
import torch.nn as nn
from typing import Dict, List, Optional
import argparse
from pathlib import Path
import sys

# Add project root to path if needed, assuming standard structure
# project_root = Path(__file__).resolve().parent.parent
# sys.path.append(str(project_root))

# Assuming single_model.py is in the same directory or accessible via python path
from .single_model import FuzzyOWLModel
from .utils import extract_vocabulary, read_list_from_file # For example usage

class FuzzyOntologyModel(nn.Module):
    """
    Represents a fuzzy ontology as a collection of multiple fuzzy interpretations (models).
    Performs semantic entailment by aggregating truth values across these models.
    """
    def __init__(self, num_models: int, num_concepts: int, num_roles: int, num_individuals: int, embedding_dim: int, fuzzy_logic: str = 'godel', device: torch.device = torch.device('cpu')):
        """
        Initializes the FuzzyOntologyModel.

        Args:
            num_models: The number of individual fuzzy models (interpretations) to create.
            num_concepts: Number of unique concepts (classes).
            num_roles: Number of unique roles (object properties).
            num_individuals: Number of unique individuals (entities).
            embedding_dim: The dimension for embeddings in each model.
            fuzzy_logic: The type of fuzzy logic to use ('godel', 'lukasiewicz', 'product').
            device: The torch device to run computations on.
        """
        super().__init__()
        self.num_models = num_models
        self.num_concepts = num_concepts
        self.num_roles = num_roles
        self.num_individuals = num_individuals
        self.embedding_dim = embedding_dim
        self.fuzzy_logic = fuzzy_logic
        self.device = device

        # Create a list to hold the individual fuzzy models
        self.models = nn.ModuleList()
        for _ in range(num_models):
            # Ensure num_individuals >= 1 for embedding layer in FuzzyOWLModel
            model_num_individuals = max(1, num_individuals)
            single_model = FuzzyOWLModel(
                num_concepts=num_concepts,
                num_roles=num_roles,
                num_individuals=model_num_individuals,
                embedding_dim=embedding_dim,
                fuzzy_logic=fuzzy_logic,
                device=device # Pass device to single model constructor
            )
            self.models.append(single_model)

        # Vocabulary mappings - store them once here
        self.concept_to_idx: Optional[Dict[str, int]] = None
        self.role_to_idx: Optional[Dict[str, int]] = None
        self.individual_to_idx: Optional[Dict[str, int]] = None

        self.to(device) # Move the ModuleList and its contents to the device

    def set_vocab_mappings(self, concept_to_idx: Dict[str, int], role_to_idx: Dict[str, int], individual_to_idx: Dict[str, int]):
        """Sets the vocabulary mapping dictionaries for all underlying models."""
        self.concept_to_idx = concept_to_idx
        self.role_to_idx = role_to_idx
        self.individual_to_idx = individual_to_idx
        for model in self.models:
            model.set_vocab_mappings(concept_to_idx, role_to_idx, individual_to_idx)

    def forward(self, axiom_str: str) -> torch.Tensor:
        """
        Evaluates the degree of semantic entailment for an axiom across all models.

        Args:
            axiom_str: The OWL axiom in Functional Syntax NNF.

        Returns:
            A scalar tensor representing the aggregated truth value [0, 1]
            (typically the minimum truth value across all models).
        """
        if not self.models:
            raise RuntimeError("No individual models have been initialized.")
        if self.concept_to_idx is None:
             raise RuntimeError("Vocabulary mappings must be set before calling forward.")
        # Handle case where no individuals exist, which might make some axioms un-evaluable
        if self.num_individuals == 0 and ("ClassAssertion" in axiom_str or "ObjectPropertyAssertion" in axiom_str):
             print(f"Warning: Cannot evaluate ABox axiom '{axiom_str}' with zero individuals. Returning truth value 0.")
             return torch.tensor(0.0, device=self.device)


        truth_values = []
        for model in self.models:
            # Each model computes the truth value in its own interpretation
            try:
                single_truth_value = model(axiom_str)
                # Ensure it's a scalar before appending
                if single_truth_value.numel() == 1:
                    truth_values.append(single_truth_value)
                else:
                    # Handle cases where a single model might return a non-scalar
                    # (e.g., if evaluating a class expression instead of an axiom)
                    # For entailment, we expect axioms yielding scalar truth values.
                    print(f"Warning: Non-scalar output from single model for '{axiom_str}'. Skipping this model's result.")
                    # Or raise an error depending on desired behavior
                    # raise ValueError(f"Expected scalar truth value from single model, got shape {single_truth_value.shape}")
            except (NotImplementedError, ValueError, RuntimeError, IndexError) as e:
                 print(f"Warning: Error evaluating axiom '{axiom_str}' in one model: {e}. Skipping this model's result.")
                 # Continue to next model, or decide on stricter error handling
            except Exception as e:
                 print(f"Warning: Unexpected error evaluating axiom '{axiom_str}' in one model: {e}. Skipping this model's result.")


        if not truth_values:
            # Handle case where no model could evaluate the axiom
            print(f"Warning: Could not evaluate axiom '{axiom_str}' in any model. Returning truth value 0.")
            return torch.tensor(0.0, device=self.device)

        # Stack the truth values into a tensor
        all_truth_values = torch.stack(truth_values) # Shape: [num_models]

        # Aggregate: Infimum (minimum) for semantic entailment
        # "An axiom is entailed if it is true in all models"
        aggregated_truth_value, _ = torch.min(all_truth_values, dim=0)

        return aggregated_truth_value


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # --- Configuration ---
    NUM_MODELS = 5
    EMBEDDING_DIM = 30
    FUZZY_LOGIC = 'godel'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Dummy Data (Replace with actual data loading) ---
    # Example: Create dummy vocab from simple axioms
    dummy_axioms = [
        "SubClassOf(<urn:A> <urn:B>)",
        "ClassAssertion(<urn:A> <urn:i>)",
        "ObjectPropertyAssertion(<urn:r> <urn:i> <urn:j>)"
    ]
    concepts, roles, individuals = extract_vocabulary(dummy_axioms)
    concept_list = sorted(list(concepts))
    role_list = sorted(list(roles))
    individual_list = sorted(list(individuals))

    concept_to_idx = {name: i for i, name in enumerate(concept_list)}
    role_to_idx = {name: i for i, name in enumerate(role_list)}
    individual_to_idx = {name: i for i, name in enumerate(individual_list)}

    num_concepts = len(concept_list)
    num_roles = len(role_list)
    num_individuals = len(individual_list) # Use actual count here for logic, FuzzyOWLModel handles >=1 internally

    print(f"Device: {DEVICE}")
    print(f"Num Models: {NUM_MODELS}")
    print(f"Vocab: {num_concepts} Concepts, {num_roles} Roles, {num_individuals} Individuals")

    # --- Instantiate the Ontology Model ---
    ontology_model = FuzzyOntologyModel(
        num_models=NUM_MODELS,
        num_concepts=num_concepts,
        num_roles=num_roles,
        num_individuals=num_individuals, # Pass actual count
        embedding_dim=EMBEDDING_DIM,
        fuzzy_logic=FUZZY_LOGIC,
        device=DEVICE
    )
    ontology_model.set_vocab_mappings(concept_to_idx, role_to_idx, individual_to_idx)
    ontology_model.eval() # Set to evaluation mode if not training

    # --- Evaluate an Axiom ---
    axiom_to_test = "SubClassOf(<urn:A> <urn:B>)"
    print(f"\nEvaluating axiom: {axiom_to_test}")

    with torch.no_grad():
        # Get truth values from individual models (optional, for inspection)
        individual_truths = []
        for i, single_model in enumerate(ontology_model.models):
             try:
                 truth = single_model(axiom_to_test)
                 if truth.numel() == 1:
                     individual_truths.append(truth.item())
                 else:
                     individual_truths.append(float('nan')) # Indicate error/non-scalar
             except Exception as e:
                 print(f"Error in model {i}: {e}")
                 individual_truths.append(float('nan')) # Indicate error

        # Get the aggregated truth value (semantic entailment degree)
        entailment_degree = ontology_model(axiom_to_test)

    print(f"Individual model truth values: {[f'{t:.4f}' if not torch.isnan(torch.tensor(t)) else 'NaN' for t in individual_truths]}")
    print(f"Aggregated Entailment Degree: {entailment_degree.item():.4f}")

    # Example with another axiom
    axiom_to_test_2 = "ClassAssertion(<urn:B> <urn:i>)"
    print(f"\nEvaluating axiom: {axiom_to_test_2}")
    with torch.no_grad():
        entailment_degree_2 = ontology_model(axiom_to_test_2)
    print(f"Aggregated Entailment Degree: {entailment_degree_2.item():.4f}")

    # Example with axiom that might fail if no individuals
    axiom_to_test_3 = "ClassAssertion(<urn:A> <urn:i>)"
    print(f"\nEvaluating axiom: {axiom_to_test_3}")
    with torch.no_grad():
        entailment_degree_3 = ontology_model(axiom_to_test_3)
    print(f"Aggregated Entailment Degree: {entailment_degree_3.item():.4f}")
