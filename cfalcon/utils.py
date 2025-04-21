import pickle
import re
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from pathlib import Path # Keep for type hints or potential future use
from typing import List, Set, Tuple, Dict

# --- File I/O ---

def load_obj(path):
    """Loads an object from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, path):
    """Saves an object to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def read_list_from_file(filepath):
    """Reads lines from a file into a list, stripping whitespace."""
    filepath = Path(filepath) # Ensure it's a Path object
    if not filepath.exists(): # Use Path directly
        print(f"Warning: File not found {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def read_tbox_test_axioms(filepath):
    """Reads axiom strings directly from a file, one per line."""
    filepath = Path(filepath) # Ensure it's a Path object
    if not filepath.exists():
        print(f"Warning: Test TBox file not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# --- OWL Functional Syntax Handling ---

# Regex to find potential IRIs or prefixed names (simplified)
# Handles <...> and potentially things like :conceptName or prefix:name
# Also captures owl:Thing and owl:Nothing explicitly
ENTITY_RE = re.compile(r"(<[^>]+>|owl:Thing|owl:Nothing|\b\w+:\w+\b|\b:\w+\b)")

def extract_vocabulary(axiom_list: List[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Extracts concepts, roles, and individuals from a list of functional syntax axioms.

    This is a best-effort extraction based on common patterns and might need
    refinement depending on the exact syntax variations (e.g., prefixed names).
    It relies on identifying the context (e.g., ClassAssertion vs ObjectPropertyAssertion)
    to differentiate between entity types.

    Args:
        axiom_list: A list of OWL axioms in functional syntax.

    Returns:
        A tuple containing:
        - Set of concept IRIs/names.
        - Set of role IRIs/names.
        - Set of individual IRIs/names.
    """
    concepts = set(['owl:Thing', 'owl:Nothing']) # Always include these
    roles = set()
    individuals = set()

    # Patterns to identify context
    class_assertion_match = re.compile(r"ClassAssertion\((.*)\s+([^)]+)\)")
    obj_prop_assertion_match = re.compile(r"ObjectPropertyAssertion\(([^ ]+)\s+([^ ]+)\s+([^)]+)\)")
    # Patterns for quantifiers (identifying roles and concepts within)
    some_values_match = re.compile(r"ObjectSomeValuesFrom\(([^ ]+)\s+(.*)\)")
    all_values_match = re.compile(r"ObjectAllValuesFrom\(([^ ]+)\s+(.*)\)")
    # SubClassOf, EquivalentClasses, DisjointClasses primarily involve concepts
    # SubObjectPropertyOf involves roles
    sub_prop_match = re.compile(r"SubObjectPropertyOf\(([^ ]+)\s+([^)]+)\)")
    # DifferentIndividuals involves individuals
    diff_ind_match = re.compile(r"DifferentIndividuals\((.*)\)")

    for axiom in axiom_list:
        # Find all potential entities (IRIs, owl:Thing, owl:Nothing)
        all_entities = set(ENTITY_RE.findall(axiom))

        # Try to determine context
        ca_match = class_assertion_match.search(axiom)
        opa_match = obj_prop_assertion_match.search(axiom)
        svf_match = some_values_match.search(axiom)
        avf_match = all_values_match.search(axiom)
        sp_match = sub_prop_match.search(axiom)
        di_match = diff_ind_match.search(axiom)

        if ca_match:
            # Argument 1 is a class expression, Argument 2 is an individual
            individuals.add(ca_match.group(2).strip())
            # Concepts within the class expression are handled below
        elif opa_match:
            # Argument 1 is a role, Arguments 2 and 3 are individuals
            roles.add(opa_match.group(1).strip())
            individuals.add(opa_match.group(2).strip())
            individuals.add(opa_match.group(3).strip())
        elif svf_match:
            # Argument 1 is a role, Argument 2 is a class expression
            roles.add(svf_match.group(1).strip())
        elif avf_match:
            # Argument 1 is a role, Argument 2 is a class expression
            roles.add(avf_match.group(1).strip())
        elif sp_match:
            # Both arguments are roles
            roles.add(sp_match.group(1).strip())
            roles.add(sp_match.group(2).strip())
        elif di_match:
            # All arguments are individuals
            ind_args = di_match.group(1).strip().split()
            individuals.update(ind_args)

        # Assume remaining entities found are concepts unless identified as roles/individuals
        # This is an approximation. A full parser would be needed for perfect accuracy.
        potential_concepts = all_entities - roles - individuals
        concepts.update(potential_concepts)


    # Clean up potential misclassifications (e.g., if owl:Thing was somehow added to roles)
    individuals.discard('owl:Thing')
    individuals.discard('owl:Nothing')
    roles.discard('owl:Thing')
    roles.discard('owl:Nothing')


    return concepts, roles, individuals


def reconstruct_functional_syntax(axiom_type, parts):
    """
    Reconstructs OWL functional syntax strings from TSV parts.
    Focuses on patterns relevant to the original FALCON TBox loss.
    """
    try:
        if axiom_type == "SubClassOf" and len(parts) == 2:
            # C <= D -> ObjectIntersectionOf(<C> ObjectComplementOf(<D>))
            return f"ObjectIntersectionOf({parts[0]} ObjectComplementOf({parts[1]}))"
        elif axiom_type == "EquivalentClasses" and len(parts) == 2:
            # C <=> D -> two axioms C <= D and D <= C
            ax1 = f"ObjectIntersectionOf({parts[0]} ObjectComplementOf({parts[1]}))"
            ax2 = f"ObjectIntersectionOf({parts[1]} ObjectComplementOf({parts[0]}))"
            return [ax1, ax2]
        elif axiom_type == "DisjointClasses" and len(parts) == 2:
            # Disjoint(C, D) -> C and D <= Nothing
            # Reconstruct format similar to original GetTBox.groovy for consistency:
            return f"ObjectIntersectionOf(ObjectIntersectionOf({parts[0]} {parts[1]}) ObjectComplementOf(owl:Nothing))"
        elif axiom_type == "SubClassOf_SomeValuesFrom" and len(parts) == 3:
            # C <= exists R.D -> ObjectIntersectionOf(<C> ObjectComplementOf(ObjectSomeValuesFrom({parts[1]} {parts[2]})))
            return f"ObjectIntersectionOf({parts[0]} ObjectComplementOf(ObjectSomeValuesFrom({parts[1]} {parts[2]})))"
        elif axiom_type == "SomeValuesFrom_SubClassOf" and len(parts) == 3:
            # exists R.C <= D -> ObjectIntersectionOf(ObjectSomeValuesFrom(<R> <C>) ObjectComplementOf(<D>))
            return f"ObjectIntersectionOf(ObjectSomeValuesFrom({parts[0]} {parts[1]}) ObjectComplementOf({parts[2]}))"
        elif axiom_type == "EquivalentClasses_SomeValuesFrom" and len(parts) == 3:
            # C <=> exists R.D -> C <= exists R.D and exists R.D <= C
            ax1 = f"ObjectIntersectionOf({parts[0]} ObjectComplementOf(ObjectSomeValuesFrom({parts[1]} {parts[2]})))"
            ax2 = f"ObjectIntersectionOf(ObjectSomeValuesFrom({parts[1]} {parts[2]}) ObjectComplementOf({parts[0]}))"
            return [ax1, ax2]
        elif axiom_type == "SubClassOf_AllValuesFrom" and len(parts) == 3:
            # C <= forall R.D -> ObjectIntersectionOf(<C> ObjectComplementOf(ObjectAllValuesFrom(<R> <D>)))
            return f"ObjectIntersectionOf({parts[0]} ObjectComplementOf(ObjectAllValuesFrom({parts[1]} {parts[2]})))"
        # Ignore Domain, Range, SubProperty for TBox loss as they likely weren't used before
        elif axiom_type in ["ObjectPropertyDomain", "ObjectPropertyRange", "SubObjectPropertyOf"]:
            return None
        else:
            # print(f"Warning: Unsupported TBox axiom type for reconstruction: {axiom_type} with parts {parts}")
            return None
    except IndexError:
        # print(f"Warning: IndexError during reconstruction for {axiom_type} with parts {parts}")
        return None

def concept_replacer(axiom_str, all_concepts_list):
    """Replaces a random concept in a functional syntax string axiom."""
    # This is a simplified replacer based on string manipulation.
    # It might break on complex nested structures.
    # A proper parser would be more robust.
    replaced_axiom = axiom_str
    # Find potential concept IRIs (simple regex) and owl:Thing/Nothing
    potential_concepts = re.findall(r'<[^>]+>', axiom_str)
    potential_concepts += re.findall(r'owl:Thing|owl:Nothing', axiom_str)

    # Filter potential concepts to those actually present in the provided list
    concepts_in_axiom = [c for c in potential_concepts if c in all_concepts_list]

    if not concepts_in_axiom:
        return axiom_str # No known concepts to replace

    concept_to_replace = random.choice(concepts_in_axiom)
    replacement_concept = random.choice(all_concepts_list)
    # Avoid replacing with Nothing or the same concept if possible
    while replacement_concept == concept_to_replace or replacement_concept == 'owl:Nothing':
         if len(all_concepts_list) <= 2: break # Avoid infinite loop if only few concepts exist
         replacement_concept = random.choice(all_concepts_list)

    # Simple string replacement (might replace unintended substrings if IRIs are substrings of others)
    # Using count=1 ensures only the first occurrence is replaced.
    replaced_axiom = replaced_axiom.replace(concept_to_replace, replacement_concept, 1)

    return replaced_axiom

# --- Data Generation ---

def tbox_test_neg_generator(tbox_train_axioms, tbox_test_pos_axioms, all_concepts_list, k):
    """Generates negative TBox test axioms by replacing concepts."""
    # Combine known positive axioms to avoid regenerating them
    all_pos_axioms = set(tbox_train_axioms) | set(tbox_test_pos_axioms)
    tbox_test_neg = []
    attempts = 0
    max_attempts = k * 100 # Limit attempts to avoid infinite loops

    # Ensure there are axioms to sample from
    source_axioms = tbox_test_pos_axioms if tbox_test_pos_axioms else tbox_train_axioms
    if not source_axioms:
        print("Warning: No source axioms provided for negative generation.")
        return []

    while len(tbox_test_neg) < k and attempts < max_attempts:
        # Choose a positive axiom
        axiom_pos = random.choice(source_axioms)
        # Replace a concept
        axiom_neg = concept_replacer(axiom_pos, all_concepts_list)
        # Ensure it's different and not already a known positive
        if axiom_neg != axiom_pos and axiom_neg not in all_pos_axioms:
            tbox_test_neg.append(axiom_neg)
            all_pos_axioms.add(axiom_neg) # Add generated negative to avoid duplicates
        attempts += 1

    if len(tbox_test_neg) < k:
        print(f"Warning: Could only generate {len(tbox_test_neg)} unique negative test axioms out of {k} requested.")

    return tbox_test_neg

def get_abox_ec_created(all_concepts_list, k):
    """Generates DataFrame for created ABox EC axioms (Entity Type Concept)."""
    ret = []
    counter = 0
    # Ensure owl:Thing and owl:Nothing are not used if present
    concepts_to_use = [c for c in all_concepts_list if c not in ['owl:Thing', 'owl:Nothing']]
    if not concepts_to_use:
        print("Warning: No concepts available (excluding Thing/Nothing) to generate EC axioms.")
        return pd.DataFrame(columns=['h', 't'])

    for concept in concepts_to_use:
        if counter < k:
            # Simple heuristic to create a new entity IRI based on the concept
            # Ensure IRI format is maintained if concept is an IRI
            if concept.startswith('<') and concept.endswith('>'):
                base_iri = concept[:-1] # Remove closing '>'
                entity_iri = f"{base_iri}_generated_1>"
            else: # Handle non-IRI concepts (like simple names)
                entity_iri = f"{concept}_generated_1"

            ret.append([entity_iri, concept])
            counter += 1
        else:
            break
    return pd.DataFrame(ret, columns=['h', 't'])

# --- Evaluation Metrics ---

def compute_metrics(preds):
    """Computes MAE(pos), AUC, AUPR, Fmax for TBox evaluation."""
    if not preds: return 0.0, 0.0, 0.0, 0.0
    n_total = len(preds)
    # Assume balanced positive/negative test set
    n_pos = n_total // 2
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        print("Warning: Cannot compute metrics with zero positive or negative samples.")
        return 0.0, 0.0, 0.0, 0.0

    # Labels: 1 for positive (true axiom), 0 for negative (false axiom)
    labels = [1] * n_pos + [0] * n_neg

    # Ensure preds are numpy array for calculations
    preds = np.array(preds)

    # MAE on positive examples (lower is better, measures deviation from 1.0)
    mae_pos = round(np.mean(1.0 - preds[:n_pos]), 4) if n_pos > 0 else 0.0

    try:
        auc = round(roc_auc_score(labels, preds), 4)
        aupr = round(average_precision_score(labels, preds), 4)
        precision, recall, _ = precision_recall_curve(labels, preds)
        # Calculate F1 score, handling division by zero
        f1_scores = np.divide(2 * recall * precision, recall + precision, out=np.zeros_like(recall), where=(recall + precision) > 0)
        fmax = round(np.max(f1_scores), 4) if len(f1_scores) > 0 else 0.0
    except ValueError as e:
        print(f"Warning: Could not compute metrics (possibly due to uniform predictions): {e}")
        auc, aupr, fmax = 0.0, 0.0, 0.0

    return mae_pos, auc, aupr, fmax

# --- Iterators ---

def iterator(dataloader):
    """Creates an infinite iterator over a dataloader."""
    while True:
        for data in dataloader:
            yield data
