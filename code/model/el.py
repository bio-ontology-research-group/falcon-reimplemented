from email.policy import default
from multiprocessing.sharedctypes import Value
from tabnanny import verbose
import torch
import pdb
import argparse
import os
import pandas as pd
import numpy as np
import random
import tqdm
import re # Keep re for now, might be needed if reconstruction needs it, though unlikely
import pickle
import torch.utils.checkpoint as checkpoint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import warnings
from pathlib import Path # Use pathlib for path manipulation

warnings.filterwarnings('ignore')


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# Removed get_rights - Not needed as Groovy script handles structure
# Removed get_all_concepts_and_relations - Not needed
# Removed extract_nodes - Not needed
# Removed read_file - Replaced by reading standardized files

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
            # C <= exists R.D -> ObjectIntersectionOf(<C> ObjectComplementOf(ObjectSomeValuesFrom(<R> <D>)))
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


def read_list_from_file(filepath):
    """Reads lines from a file into a list, stripping whitespace."""
    if not filepath.exists():
        print(f"Warning: File not found {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_abox_ec_created(all_concepts_list, k):
    """Generates DataFrame for created ABox EC axioms."""
    ret = []
    counter = 0
    # Ensure owl:Thing and owl:Nothing are not used if present
    concepts_to_use = [c for c in all_concepts_list if c not in ['owl:Thing', 'owl:Nothing']]
    for concept in concepts_to_use:
        if counter < k:
            # Simple heuristic to create a new entity IRI
            entity_iri = concept[:-1] + '_generated_1>' if concept.endswith('>') else concept + '_generated_1'
            ret.append([entity_iri, concept])
            counter += 1
        else:
            break
    return pd.DataFrame(ret, columns=['h', 't'])

def get_data(cfg):
    """Loads data from preprocessed files specified by cfg.data_path."""
    data_path = Path(cfg.data_path)
    print(f"Loading data from: {data_path.resolve()}")

    # Load concepts, relations, entities
    all_concepts_list = read_list_from_file(data_path / "concepts.txt")
    all_relations_list = read_list_from_file(data_path / "relations.txt")
    all_entities_list = read_list_from_file(data_path / "entities.txt")

    # Add special concepts/relations if needed by the model logic (e.g., subClassOf)
    if 'owl:Thing' not in all_concepts_list: all_concepts_list.append('owl:Thing')
    if 'owl:Nothing' not in all_concepts_list: all_concepts_list.append('owl:Nothing')
    if 'subClassOf' not in all_relations_list: all_relations_list.append('subClassOf') # Used in tbox_name

    # Load ABox data
    try:
        abox_ec = pd.read_csv(data_path / "abox_ec.tsv", sep='\t', header=None, names=['h', 't'], keep_default_na=False)
        # Add relation column (assuming a default if not present, e.g., 'hasPhenotype' or 'isA')
        # The original el.py used 'hasPhenotype'. Let's assume this relation IRI exists or add it.
        pheno_relation = '<http://hasPhenotype>' # Example IRI, adjust if needed
        if pheno_relation not in all_relations_list: all_relations_list.append(pheno_relation)
        abox_ec['r'] = pheno_relation
        abox_ec = abox_ec[['h', 'r', 't']] # Reorder columns
    except FileNotFoundError:
        print(f"Warning: {data_path / 'abox_ec.tsv'} not found. ABox EC will be empty.")
        abox_ec = pd.DataFrame(columns=['h', 'r', 't'])
    except pd.errors.EmptyDataError:
         print(f"Warning: {data_path / 'abox_ec.tsv'} is empty. ABox EC will be empty.")
         abox_ec = pd.DataFrame(columns=['h', 'r', 't'])


    try:
        abox_ee = pd.read_csv(data_path / "abox_ee.tsv", sep='\t', header=None, names=['h', 'r', 't'], keep_default_na=False)
    except FileNotFoundError:
        print(f"Warning: {data_path / 'abox_ee.tsv'} not found. ABox EE will be empty.")
        abox_ee = pd.DataFrame(columns=['h', 'r', 't'])
    except pd.errors.EmptyDataError:
        print(f"Warning: {data_path / 'abox_ee.tsv'} is empty. ABox EE will be empty.")
        abox_ee = pd.DataFrame(columns=['h', 'r', 't'])

    # Load and reconstruct TBox data
    tbox_name_list = []
    tbox_desc = []
    try:
        with open(data_path / "tbox.tsv", 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if not parts: continue
                axiom_type = parts[0]
                axiom_parts = parts[1:]

                # Reconstruct functional syntax
                reconstructed = reconstruct_functional_syntax(axiom_type, axiom_parts)

                if reconstructed:
                    if isinstance(reconstructed, list): # Handle EquivalentClasses producing two axioms
                        for axiom_str in reconstructed:
                            # Check if it matches the 'name' pattern C <= D
                            name_match = re.match(r'ObjectIntersectionOf\((<.*?>|owl:Thing) ObjectComplementOf\((<.*?>|owl:Thing)\)\)', axiom_str, re.M|re.I)
                            if name_match:
                                matched = name_match.groups()
                                tbox_name_list.append([matched[0], 'subClassOf', matched[1]])
                            else:
                                tbox_desc.append(axiom_str)
                    else:
                        axiom_str = reconstructed
                        # Check if it matches the 'name' pattern C <= D
                        name_match = re.match(r'ObjectIntersectionOf\((<.*?>|owl:Thing) ObjectComplementOf\((<.*?>|owl:Thing)\)\)', axiom_str, re.M|re.I)
                        if name_match:
                            matched = name_match.groups()
                            tbox_name_list.append([matched[0], 'subClassOf', matched[1]])
                        else:
                            tbox_desc.append(axiom_str)
    except FileNotFoundError:
        print(f"Warning: {data_path / 'tbox.tsv'} not found. TBox will be empty.")

    tbox_name = pd.DataFrame(tbox_name_list, columns=['h', 'r', 't'])

    # Ensure all concepts/relations from loaded axioms are in the lists
    for c in pd.concat([tbox_name['h'], tbox_name['t'], abox_ec['t']]).unique():
        if c not in all_concepts_list: all_concepts_list.append(c)
    for r in pd.concat([tbox_name['r'], abox_ec['r'], abox_ee['r']]).unique():
         if r not in all_relations_list: all_relations_list.append(r)
    # Entities are trickier if new ones appear only in ABox
    abox_entities = set(abox_ec['h']) | set(abox_ee['h']) | set(abox_ee['t'])
    new_entities = abox_entities - set(all_entities_list)
    if new_entities:
        print(f"Warning: Found {len(new_entities)} entities in ABox not listed in entities.txt. Adding them.")
        all_entities_list.extend(list(new_entities))

    # Generate created entities for ABox EC
    abox_ec_created = get_abox_ec_created(all_concepts_list, k=cfg.n_abox_ec_created)
    created_entities_list = list(abox_ec_created['h'].unique())

    # Create dictionaries (ensure stable order)
    all_concepts_list = sorted(list(set(all_concepts_list)))
    all_relations_list = sorted(list(set(all_relations_list)))
    all_entities_list = sorted(list(set(all_entities_list)))
    created_entities_list = sorted(list(set(created_entities_list)))

    c_dict = {k: v for v, k in enumerate(all_concepts_list)}
    e_dict = {k: v for v, k in enumerate(all_entities_list)}
    # Ensure created entity IDs start after existing entity IDs
    e_dict_created = {k: v + len(e_dict) for v, k in enumerate(created_entities_list)}
    # Combine entity dictionaries
    e_dict_more = {**e_dict, **e_dict_created}
    r_dict = {k: v for v, k in enumerate(all_relations_list)}

    # --- Map IRIs to IDs ---
    # Need to handle potential missing keys if filtering happened before dict creation
    tbox_name['h'] = tbox_name['h'].map(c_dict.get)
    tbox_name['r'] = tbox_name['r'].map(r_dict.get)
    tbox_name['t'] = tbox_name['t'].map(c_dict.get)
    tbox_name.dropna(inplace=True) # Remove rows where mapping failed
    tbox_name = tbox_name.astype(int)

    abox_ec['h'] = abox_ec['h'].map(e_dict.get)
    abox_ec['r'] = abox_ec['r'].map(r_dict.get)
    abox_ec['t'] = abox_ec['t'].map(c_dict.get)
    abox_ec.dropna(inplace=True)
    abox_ec = abox_ec.astype(int)

    abox_ec_created['h'] = abox_ec_created['h'].map(e_dict_more.get)
    abox_ec_created['t'] = abox_ec_created['t'].map(c_dict.get)
    abox_ec_created.dropna(inplace=True)
    abox_ec_created = abox_ec_created.astype(int)

    abox_ee['h'] = abox_ee['h'].map(e_dict.get)
    abox_ee['r'] = abox_ee['r'].map(r_dict.get)
    abox_ee['t'] = abox_ee['t'].map(e_dict.get)
    abox_ee.dropna(inplace=True)
    abox_ee = abox_ee.astype(int)

    # --- Split ABox EE (GGI data) ---
    # Use the loaded abox_ee for train/test split
    # This replaces the previous loading of separate train/test files
    # Simple random split for demonstration. Consider using predefined splits if available.
    if len(abox_ee) > 0:
        abox_ee_test = abox_ee.sample(frac=0.2, random_state=42) # Use a fixed state for reproducibility
        abox_ee_train = abox_ee.drop(abox_ee_test.index)
    else:
        abox_ee_train = pd.DataFrame(columns=['h', 'r', 't'])
        abox_ee_test = pd.DataFrame(columns=['h', 'r', 't'])

    # Create already known dictionaries for evaluation filtering
    already_ts_dict = {}
    already_hs_dict = {}
    if not abox_ee_train.empty:
        already_ts = abox_ee_train.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
        already_hs = abox_ee_train.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
        for record in already_ts:
            # Ensure keys are integers
            already_ts_dict[(int(record[0]), int(record[1]))] = [int(x) for x in record[2]]
        for record in already_hs:
            # Ensure keys are integers
            already_hs_dict[(int(record[0]), int(record[1]))] = [int(x) for x in record[2]]

    # Note: The original el.py had separate valid/test sets. This simplified version only has train/test.
    # Adjust if validation set is needed.

    # Return mapped data and dictionaries
    return tbox_name, tbox_desc, abox_ec, abox_ec_created, abox_ee_train, abox_ee_test, \
           c_dict, e_dict, e_dict_more, r_dict, already_ts_dict, already_hs_dict


class GGIDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, e_dict_len, already_ts_dict, already_hs_dict, stage): # Pass length instead of dict
        super().__init__()
        self.stage = stage
        self.cfg = cfg
        # self.e_dict = e_dict # No longer needed directly
        self.e_dict_len = e_dict_len
        self.data = torch.tensor(data.values)
        self.all_candidate = torch.arange(self.e_dict_len).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng//2, 1, dtype=torch.long) # Match tensor type
        self.already_ts_dict = already_ts_dict
        self.already_hs_dict = already_hs_dict

    def sampling(self, pos):
        head, rel, tail = pos
        # Ensure keys exist and handle potential errors
        ts_key = (head.item(), rel.item())
        hs_key = (tail.item(), rel.item())
        already_ts = torch.tensor(self.already_ts_dict.get(ts_key, []), dtype=torch.long)
        already_hs = torch.tensor(self.already_hs_dict.get(hs_key, []), dtype=torch.long)

        # Efficient negative sampling avoiding known positives
        neg_pool_t = torch.ones(self.e_dict_len, dtype=torch.bool)
        if len(already_ts) > 0:
             neg_pool_t[already_ts] = 0
        neg_pool_t = neg_pool_t.nonzero().squeeze() # Get indices where value is True

        neg_pool_h = torch.ones(self.e_dict_len, dtype=torch.bool)
        if len(already_hs) > 0:
            neg_pool_h[already_hs] = 0
        neg_pool_h = neg_pool_h.nonzero().squeeze()

        # Handle cases where pool might be empty (shouldn't happen in dense KGs)
        num_neg_t = len(neg_pool_t) if neg_pool_t.dim() > 0 else 0
        num_neg_h = len(neg_pool_h) if neg_pool_h.dim() > 0 else 0

        neg_t_indices = torch.randint(num_neg_t, (self.cfg.num_ng//2,)) if num_neg_t > 0 else torch.tensor([], dtype=torch.long)
        neg_h_indices = torch.randint(num_neg_h, (self.cfg.num_ng//2,)) if num_neg_h > 0 else torch.tensor([], dtype=torch.long)

        neg_t = neg_pool_t[neg_t_indices].unsqueeze(-1) if num_neg_t > 0 else torch.zeros((0, 1), dtype=torch.long)
        neg_h = neg_pool_h[neg_h_indices].unsqueeze(-1) if num_neg_h > 0 else torch.zeros((0, 1), dtype=torch.long)

        # If not enough negatives found, pad with random samples (less ideal)
        # This part needs careful handling based on how many negatives are strictly required
        num_needed = self.cfg.num_ng // 2
        if len(neg_t) < num_needed:
             # Sample randomly, excluding the positive tail
             random_negs = torch.randint(self.e_dict_len, (num_needed - len(neg_t),))
             # Very basic exclusion, might still sample known positives if not in 'already_ts'
             random_negs = random_negs[random_negs != tail.item()]
             neg_t = torch.cat([neg_t, random_negs.unsqueeze(-1)], dim=0) if len(random_negs) > 0 else neg_t
             # Ensure correct size if still too few
             if len(neg_t) < num_needed:
                 neg_t = torch.cat([neg_t, torch.randint(self.e_dict_len, (num_needed - len(neg_t), 1))], dim=0)


        if len(neg_h) < num_needed:
             random_negs = torch.randint(self.e_dict_len, (num_needed - len(neg_h),))
             random_negs = random_negs[random_negs != head.item()]
             neg_h = torch.cat([neg_h, random_negs.unsqueeze(-1)], dim=0) if len(random_negs) > 0 else neg_h
             if len(neg_h) < num_needed:
                 neg_h = torch.cat([neg_h, torch.randint(self.e_dict_len, (num_needed - len(neg_h), 1))], dim=0)

        return neg_t[:num_needed], neg_h[:num_needed] # Ensure correct size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx] # Should be tensor of shape (3,)
        if self.stage == 'train':
            neg_t, neg_h = self.sampling(pos)
            # Ensure neg_t and neg_h have the expected shape for concatenation
            if neg_t.shape[0] != self.cfg.num_ng // 2 or neg_h.shape[0] != self.cfg.num_ng // 2:
                 # Fallback or error handling if sampling failed to produce enough negatives
                 # print(f"Warning: Sampling issue at index {idx}. Got {neg_t.shape[0]} tail negs, {neg_h.shape[0]} head negs.")
                 # Using random as fallback:
                 neg_t = torch.randint(self.e_dict_len, (self.cfg.num_ng // 2, 1), dtype=torch.long)
                 neg_h = torch.randint(self.e_dict_len, (self.cfg.num_ng // 2, 1), dtype=torch.long)

            # Expand pos components correctly
            pos_h_expanded = pos[0].expand_as(neg_t) # Shape [num_ng/2, 1]
            pos_r_expanded_t = pos[1].expand_as(neg_t)
            pos_r_expanded_h = pos[1].expand_as(neg_h)
            pos_t_expanded = pos[2].expand_as(neg_h)

            replace_tail = torch.cat([pos_h_expanded, pos_r_expanded_t, neg_t], dim=-1)
            replace_head = torch.cat([neg_h, pos_r_expanded_h, pos_t_expanded], dim=-1)

            # Ensure pos is unsqueezed correctly
            return torch.cat([pos.unsqueeze(0), replace_tail, replace_head], dim=0)

        elif self.stage == 'test':
            # Expand pos components for all candidates
            pos_h_expanded = pos[0].expand_as(self.all_candidate)
            pos_r_expanded = pos[1].expand_as(self.all_candidate)
            pos_t_expanded = pos[2].expand_as(self.all_candidate)

            replace_tail = torch.cat([pos_h_expanded, pos_r_expanded, self.all_candidate], dim=-1)
            replace_head = torch.cat([self.all_candidate, pos_r_expanded, pos_t_expanded], dim=-1)
            return torch.cat([replace_head, replace_tail], dim=0), pos
        else:
            raise ValueError(f"Invalid stage: {self.stage}")


class AboxECDataset(torch.utils.data.Dataset):
     def __init__(self, cfg, data, e_dict_len): # Pass length
        super().__init__()
        self.cfg = cfg
        # self.e_dict = e_dict
        self.e_dict_len = e_dict_len
        self.data = torch.tensor(data.values)
        self.all_candidate = torch.arange(self.e_dict_len).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng, 1, dtype=torch.long) # Match type

     def __len__(self):
        return len(self.data)

     def __getitem__(self, idx):
        pos = self.data[idx] # Shape (3,) : h, r, t
        # Sample negative heads
        negs = torch.randint(self.e_dict_len, (self.cfg.num_ng, 1), dtype=torch.long)

        # Expand relation and tail from pos
        pos_r_expanded = pos[1].expand_as(negs)
        pos_t_expanded = pos[2].expand_as(negs)

        replace_head = torch.cat([negs, pos_r_expanded, pos_t_expanded], dim=-1)
        return torch.cat([pos.unsqueeze(0), replace_head], dim=0)


class AboxECCreatedDataset(torch.utils.data.Dataset):
    # This dataset seems to be for axioms like GeneratedEntity_1 Type ConceptA
    # The original code sampled negative *heads* (entities) for a positive (gen_entity, concept) pair.
    # Let's adapt based on the columns provided by get_abox_ec_created: 'h', 't'
    def __init__(self, cfg, data, e_dict_len, c_dict_len): # Pass lengths
        super().__init__()
        self.cfg = cfg
        # self.e_dict = e_dict # Use length
        self.e_dict_len = e_dict_len
        self.c_dict_len = c_dict_len # Need concept length if sampling concepts
        self.data = torch.tensor(data.values) # Shape (N, 2): entity_h, concept_t

        # Decide what to sample: negative entities or negative concepts?
        # Original code sampled negative *heads* (entities) for EC axioms. Let's stick to that.
        self.all_candidate_entities = torch.arange(self.e_dict_len).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng, 1, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx] # Shape (2,): h, t (entity, concept)
        pos_h = pos[0]
        pos_t = pos[1]

        # Sample negative entities (heads)
        neg_entities = torch.randint(self.e_dict_len, (self.cfg.num_ng, 1), dtype=torch.long)

        # Expand the positive concept
        pos_t_expanded = pos_t.expand_as(neg_entities)

        # Create negative samples: (neg_entity, pos_concept)
        neg_samples = torch.cat([neg_entities, pos_t_expanded], dim=-1)

        # Return positive sample and negative samples
        # Shape: [1+num_ng, 2]
        return torch.cat([pos.unsqueeze(0), neg_samples], dim=0)


class NaiveDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class FALCON(torch.nn.Module):
    # Assuming c_dict, e_dict, r_dict passed to __init__ are now the dictionaries mapping IRI->ID
    # Need to adjust embedding sizes based on the *length* of these dictionaries
    def __init__(self, c_dict_len, e_dict_len, r_dict_len, cfg, device): # Pass lengths
        super().__init__()
        # Store dicts if needed for forward lookup, otherwise just lengths
        # self.c_dict = c_dict
        # self.e_dict = e_dict
        # self.r_dict = r_dict
        self.c_dict_len = c_dict_len
        self.e_dict_len = e_dict_len # This is the length *including* created entities, but *excluding* anon
        self.r_dict_len = r_dict_len

        self.n_entity_base = e_dict_len # Number of entities from file + created
        self.anon_e = cfg.anon_e
        self.n_entity_total = self.n_entity_base + self.anon_e # Total entities including anonymous
        self.cfg = cfg

        # Embeddings sized by dictionary lengths
        self.c_embedding = torch.nn.Embedding(self.c_dict_len, cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(self.r_dict_len, cfg.emb_dim)
        # Embedding for base entities (from file + created)
        self.e_embedding_base = torch.nn.Embedding(self.n_entity_base, cfg.emb_dim)

        # Linear layers (no change needed)
        self.fc_0 = torch.nn.Linear(cfg.emb_dim * 2, 1)
        # self.fc_1 = torch.nn.Linear(cfg.emb_dim, 1)

        # Initialization (no change needed)
        torch.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.e_embedding_base.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_0.weight.data)
        # torch.nn.init.xavier_uniform_(self.fc_1.weight.data)

        # Fuzzy logic parameters (no change needed)
        self.max_measure = cfg.max_measure
        self.t_norm = cfg.t_norm
        self.nothing_fs = torch.zeros(self.n_entity_total).to(device) # Fuzzy set for owl:Nothing
        self.residuum = cfg.residuum
        self.device = device

        # Store IRI->ID maps if needed for parsing inside forward (if reconstruction fails)
        # This is avoided by reconstructing strings before calling forward
        # self.c_dict_rev = {v: k for k, v in c_dict.items()} # Example reverse map

    # --- Fuzzy Logic Operators --- (No changes needed)
    def _logical_and(self, x, y):
        if self.t_norm == 'product':
            return x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return torch.cat([x, y], dim=-2).min(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return (((x + y -1) > 0) * (x + y - 1)).squeeze(dim=-2)
        else:
            raise ValueError

    def _logical_or(self, x, y):
        if self.t_norm == 'product':
            return x + y - x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return torch.cat([x, y], dim=-2).max(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return 1 - ((((1-x) + (1-y) -1) > 0) * ((1-x) + (1-y) - 1)).squeeze(dim=-2)
        else:
            raise ValueError

    def _logical_not(self, x):
        return 1 - x

    def _logical_residuum(self, r_fs, c_fs):
        if self.residuum == 'notCorD':
            # Ensure c_fs is broadcastable over r_fs's entity dimensions
            # r_fs shape: [batch?, entity, entity] or [entity, entity]
            # c_fs shape: [batch?, entity] or [entity]
            # We need c_fs to be [batch?, 1, entity] or [1, entity]
            c_fs_expanded = c_fs.unsqueeze(-2) # Add dimension for broadcasting
            return self._logical_or(self._logical_not(r_fs), c_fs_expanded)
        else:
            raise ValueError

    def _logical_exist(self, r_fs, c_fs):
        # r_fs shape: [batch?, entity, entity] or [entity, entity]
        # c_fs shape: [batch?, entity] or [entity]
        # We need c_fs to be [batch?, 1, entity] or [1, entity] for broadcasting
        c_fs_expanded = c_fs.unsqueeze(-2)
        # Product/Min happens element-wise -> [batch?, entity, entity]
        # Max over the *last* dimension (the object entity 'y' in R(x,y) and C(y))
        return self._logical_and(r_fs, c_fs_expanded).max(dim=-1)[0]

    def _logical_forall(self, r_fs, c_fs):
        # Residuum gives [batch?, entity, entity]
        # Min over the *last* dimension (the object entity 'y')
        return self._logical_residuum(r_fs, c_fs).min(dim=-1)[0]

    # --- Fuzzy Set Calculation ---
    def _get_all_entity_embeddings(self, anon_e_emb):
        """Combines base entity embeddings with current anonymous embeddings."""
        # Ensure anon_e_emb has the correct shape and device
        if anon_e_emb.shape[0] != self.anon_e or anon_e_emb.shape[1] != self.cfg.emb_dim:
             raise ValueError(f"Incorrect shape for anon_e_emb: {anon_e_emb.shape}")
        if anon_e_emb.device != self.device:
             anon_e_emb = anon_e_emb.to(self.device)

        return torch.cat([self.e_embedding_base.weight, anon_e_emb], dim=0)

    def _get_c_fs(self, c_emb, all_e_emb):
        """Calculates fuzzy set for a concept C over all entities."""
        # all_e_emb shape: [n_entity_total, emb_dim]
        # c_emb shape: [emb_dim]
        c_emb_expanded = c_emb.expand_as(all_e_emb) # Shape: [n_entity_total, emb_dim]
        emb = torch.cat([c_emb_expanded, all_e_emb], dim=-1) # Shape: [n_entity_total, emb_dim * 2]
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1) # Shape: [n_entity_total]

    def _get_c_fs_batch(self, c_emb_batch, all_e_emb):
        """Calculates fuzzy sets for a batch of concepts C over all entities."""
        # c_emb_batch shape: [batch_size, emb_dim]
        # all_e_emb shape: [n_entity_total, emb_dim]
        num_entities = all_e_emb.size(0)
        batch_size = c_emb_batch.size(0)

        # Expand dimensions for broadcasting
        # all_e_emb_expanded shape: [batch_size, n_entity_total, emb_dim]
        all_e_emb_expanded = all_e_emb.unsqueeze(0).expand(batch_size, num_entities, self.cfg.emb_dim)
        # c_emb_batch_expanded shape: [batch_size, n_entity_total, emb_dim]
        c_emb_batch_expanded = c_emb_batch.unsqueeze(1).expand(batch_size, num_entities, self.cfg.emb_dim)

        # Concatenate
        emb = torch.cat([c_emb_batch_expanded, all_e_emb_expanded], dim=-1) # Shape: [batch_size, n_entity_total, emb_dim * 2]

        # Apply layers
        # fs_batch = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        fs_batch = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1) # Shape: [batch_size, n_entity_total]
        return fs_batch

    def _get_r_fs(self, r_emb, all_e_emb):
        """Calculates fuzzy set for a relation R over all entity pairs (x, y)."""
        # all_e_emb shape: [n_entity_total, emb_dim]
        # r_emb shape: [emb_dim]
        num_entities = all_e_emb.size(0)

        # Prepare left (subject) and right (object) embeddings
        # l_emb = (all_e_emb + r_emb.unsqueeze(0)) # Add relation to subject: [n_entity_total, emb_dim]
        # r_emb_plain = all_e_emb # Object embedding: [n_entity_total, emb_dim]

        # Expand for pairs (x, y)
        # l_emb_expanded shape: [n_entity_total, n_entity_total, emb_dim] (x varies first)
        # l_emb_expanded = l_emb.unsqueeze(1).expand(num_entities, num_entities, self.cfg.emb_dim)
        # r_emb_expanded shape: [n_entity_total, n_entity_total, emb_dim] (y varies first)
        # r_emb_expanded = r_emb_plain.unsqueeze(0).expand(num_entities, num_entities, self.cfg.emb_dim)

        # Alternative expansion (more memory efficient if needed, but less explicit)
        e_emb_repeated_rows = all_e_emb.repeat_interleave(num_entities, dim=0) # [N*N, D]
        e_emb_repeated_cols = all_e_emb.repeat(num_entities, 1) # [N*N, D]
        l_emb_flat = e_emb_repeated_rows + r_emb.unsqueeze(0) # [N*N, D]
        r_emb_flat = e_emb_repeated_cols # [N*N, D]

        emb_flat = torch.cat([l_emb_flat, r_emb_flat], dim=-1) # Shape: [N*N, emb_dim * 2]

        # Apply layers
        # fs_flat = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb_flat), negative_slope=0.1))).squeeze(dim=-1)
        fs_flat = torch.sigmoid(self.fc_0(emb_flat)).squeeze(dim=-1) # Shape: [N*N]

        # Reshape to [n_entity_total, n_entity_total]
        fs = fs_flat.view(num_entities, num_entities)
        return fs

    def _get_r_fs_batch(self, r_emb_batch, all_e_emb):
        """Calculates fuzzy sets for a batch of relations R over all entity pairs."""
        # r_emb_batch shape: [batch_size, emb_dim]
        # all_e_emb shape: [n_entity_total, emb_dim]
        num_entities = all_e_emb.size(0)
        batch_size = r_emb_batch.size(0)

        # Expand embeddings
        # all_e_emb: [N, D] -> [B, N, N, D]
        e_emb_expanded_l = all_e_emb.unsqueeze(0).unsqueeze(2).expand(batch_size, num_entities, num_entities, self.cfg.emb_dim)
        e_emb_expanded_r = all_e_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, num_entities, num_entities, self.cfg.emb_dim)
        # r_emb_batch: [B, D] -> [B, N, N, D]
        r_emb_expanded = r_emb_batch.unsqueeze(1).unsqueeze(2).expand(batch_size, num_entities, num_entities, self.cfg.emb_dim)

        # Calculate subject embedding: E_l + R
        l_emb = e_emb_expanded_l + r_emb_expanded
        r_emb = e_emb_expanded_r

        emb = torch.cat([l_emb, r_emb], dim=-1) # Shape: [B, N, N, 2*D]

        # Apply layers
        # fs_batch = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        fs_batch = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1) # Shape: [B, N, N]
        return fs_batch

    # --- Forward Pass for Axioms ---
    # This method now relies on the input `axiom` being the reconstructed functional syntax string
    def forward(self, axiom_str, anon_e_emb):
        # Get combined entity embeddings for this forward pass
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)

        # --- Recursive Parsing and Calculation ---
        # This part remains largely the same, operating on the reconstructed string `axiom_str`
        # It needs the IRI->ID dictionaries (c_dict, r_dict) to get embedding indices.
        # These dictionaries should be stored in `self` during __init__ if this approach is used.
        # *** Correction: The current code gets IDs *outside* forward (e.g., in forward_name).
        # *** Let's assume forward receives the *axiom string* and looks up IDs internally.
        # *** This requires storing c_dict, r_dict in self.

        # Store dictionaries if needed (Add to __init__)
        # self.c_dict = c_dict
        # self.r_dict = r_dict

        # --- Base Cases ---
        if axiom_str[0] == '<' or axiom_str == 'owl:Thing':
            try:
                # Use stored dictionary (assuming it exists)
                # c_id = torch.tensor(self.c_dict[axiom_str]).to(self.device)
                # Need c_dict available here! Pass it or store it.
                # Let's assume c_dict is available globally or passed somehow for now.
                # *** Revised approach: Pass dicts to forward if not storing in self ***
                # def forward(self, axiom_str, anon_e_emb, c_dict, r_dict):
                # c_id = torch.tensor(c_dict[axiom_str]).to(self.device)

                # *** Safest approach: Modify callers (forward_name, etc.) to pass IDs ***
                # *** OR: Keep the original structure where forward *only* handles structure ***
                # *** and IDs are looked up *before* calling forward. This seems best.      ***
                # *** Let's revert `forward` to assume it gets called recursively with strings, ***
                # *** and the initial call (e.g. from loss calculation) handles the top-level ID lookup. ***

                # If forward is called with a concept IRI string:
                if axiom_str not in c_dict: # Use global c_dict for check
                     print(f"Warning: Concept IRI '{axiom_str}' not in c_dict during forward.")
                     # Return fuzzy set for Nothing or handle error
                     return self.nothing_fs
                c_id = torch.tensor(c_dict[axiom_str]).to(self.device) # Use global c_dict
                c_emb = self.c_embedding(c_id)
                # Use checkpointing if needed
                # ret = self._get_c_fs(c_emb, all_e_emb)
                ret = checkpoint.checkpoint(self._get_c_fs, c_emb, all_e_emb, use_reentrant=False) # Checkpoint requires use_reentrant=False for newer PyTorch
                return ret
            except KeyError:
                 print(f"Error: Concept IRI '{axiom_str}' not found in c_dict during forward.")
                 return self.nothing_fs # Return fuzzy set for Nothing


        elif axiom_str == 'owl:Nothing':
            return self.nothing_fs

        # --- Recursive Steps ---
        # These rely on string parsing and recursive calls to self.forward
        elif axiom_str.startswith('ObjectIntersectionOf('):
            # Simplified parsing assuming structure: ObjectIntersectionOf(LEFT RIGHT)
            content = axiom_str[21:-1] # Get content inside brackets
            # Find split point (first space after matching parentheses)
            split_index = -1
            paren_level = 0
            for i, char in enumerate(content):
                if char == '(': paren_level += 1
                elif char == ')': paren_level -= 1
                elif char == ' ' and paren_level == 0:
                    split_index = i
                    break
            if split_index == -1: raise ValueError(f"Cannot parse ObjectIntersectionOf: {axiom_str}")

            left_str = content[:split_index].strip()
            right_str = content[split_index:].strip()

            # Handle multiple intersections (A and B and C...) - assuming right-associativity from get_rights
            # The reconstruction might produce nested intersections or just pairs.
            # Assume pairs for now, consistent with reconstruction.
            # If get_rights logic is needed, it must be added back here or in reconstruction.
            fs_left = self.forward(left_str, anon_e_emb)
            fs_right = self.forward(right_str, anon_e_emb)
            return self._logical_and(fs_left, fs_right)
            # TODO: Adapt if reconstruction produces different structures or if get_rights is needed.


        elif axiom_str.startswith('ObjectUnionOf('):
             # Simplified parsing: ObjectUnionOf(LEFT RIGHT)
            content = axiom_str[14:-1]
            split_index = -1
            paren_level = 0
            for i, char in enumerate(content):
                if char == '(': paren_level += 1
                elif char == ')': paren_level -= 1
                elif char == ' ' and paren_level == 0:
                    split_index = i
                    break
            if split_index == -1: raise ValueError(f"Cannot parse ObjectUnionOf: {axiom_str}")
            left_str = content[:split_index].strip()
            right_str = content[split_index:].strip()
            fs_left = self.forward(left_str, anon_e_emb)
            fs_right = self.forward(right_str, anon_e_emb)
            return self._logical_or(fs_left, fs_right)


        elif axiom_str.startswith('ObjectSomeValuesFrom('):
            # Format: ObjectSomeValuesFrom(<R> C)
            parts = axiom_str[21:-1].split(' ', 1) # Split only on the first space
            relation_iri = parts[0]
            concept_str = parts[1]
            if relation_iri not in r_dict: # Use global r_dict
                 print(f"Warning: Relation IRI '{relation_iri}' not in r_dict during forward.")
                 return self.nothing_fs
            r_id = torch.tensor(r_dict[relation_iri]).to(self.device) # Use global r_dict
            r_emb = self.r_embedding(r_id)
            # r_fs = self._get_r_fs(r_emb, all_e_emb)
            r_fs = checkpoint.checkpoint(self._get_r_fs, r_emb, all_e_emb, use_reentrant=False)
            c_fs = self.forward(concept_str, anon_e_emb)
            return self._logical_exist(r_fs, c_fs)


        elif axiom_str.startswith('ObjectAllValuesFrom('):
            # Format: ObjectAllValuesFrom(<R> C)
            parts = axiom_str[20:-1].split(' ', 1)
            relation_iri = parts[0]
            concept_str = parts[1]
            if relation_iri not in r_dict: # Use global r_dict
                 print(f"Warning: Relation IRI '{relation_iri}' not in r_dict during forward.")
                 return self.nothing_fs
            r_id = torch.tensor(r_dict[relation_iri]).to(self.device) # Use global r_dict
            r_emb = self.r_embedding(r_id)
            # r_fs = self._get_r_fs(r_emb, all_e_emb)
            r_fs = checkpoint.checkpoint(self._get_r_fs, r_emb, all_e_emb, use_reentrant=False)
            c_fs = self.forward(concept_str, anon_e_emb)
            return self._logical_forall(r_fs, c_fs)


        elif axiom_str.startswith('ObjectComplementOf('):
            concept_str = axiom_str[19:-1]
            fs = self.forward(concept_str, anon_e_emb)
            return self._logical_not(fs)

        else:
            print(f"Error: Unrecognized axiom structure in forward: {axiom_str}")
            # raise ValueError(f"Unrecognized axiom structure: {axiom_str}")
            return self.nothing_fs # Or some default error state


    # --- Loss Functions ---
    def forward_name(self, x, anon_e_emb):
        """ Calculates loss for TBox 'name' axioms (C <= D). """
        # x shape: [batch_size, 3] (c_id_left, r_id_subclassof, c_id_right)
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)

        c_emb_left = self.c_embedding(x[:, 0])   # Shape: [batch_size, emb_dim]
        c_emb_right = self.c_embedding(x[:, 2])  # Shape: [batch_size, emb_dim]

        # Get fuzzy sets for C and D over all entities
        fs_left = self._get_c_fs_batch(c_emb_left, all_e_emb)   # Shape: [batch_size, n_entity_total]
        fs_right = self._get_c_fs_batch(c_emb_right, all_e_emb) # Shape: [batch_size, n_entity_total]

        # Loss for C <= D is minimizing max(C and not D)
        fs_intersection = self._logical_and(fs_left, self._logical_not(fs_right))
        loss = self.get_cc_loss(fs_intersection) # Calculate max and log loss
        return loss.mean() # Average over batch

    def get_cc_loss(self, fs_intersection):
        """ Calculates the loss for concept containment based on max measure. """
        # fs_intersection represents C and not D, or more generally, the set that should be empty.
        if self.max_measure == 'max':
            # Maximize 1 - max(intersection) => Minimize max(intersection)
            # Loss = -log(1 - max(intersection) + epsilon)
            max_val = fs_intersection.max(dim=-1)[0]
            return -torch.log(1 - max_val + 1e-10)
        elif self.max_measure.startswith('pmean'):
            # Loss = -log(1 - pmean(intersection) + epsilon)
            try:
                p = int(self.max_measure[-1])
            except ValueError:
                raise ValueError(f"Invalid pmean value: {self.max_measure}")
            pmean_val = ((fs_intersection ** p).mean(dim=-1))**(1/p)
            return -torch.log(1 - pmean_val + 1e-10)
        else:
            raise ValueError(f"Unknown max_measure: {self.max_measure}")

    def forward_abox_ec(self, x, anon_e_emb):
        """ Calculates loss for ABox EC axioms (like C(a)). """
        # x shape: [batch_size, 1 + num_neg, 3] (h_id, r_id, t_id)
        # Positive sample: x[:, 0, :]
        # Negative samples: x[:, 1:, :] (negative heads)
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)

        # Get embeddings
        # Entity embeddings need careful indexing - they might be base or anonymous
        # The current dataloader samples negative *heads* from base entities.
        # Let's assume h_id refers to indices in the base embedding table.
        e_emb = self.e_embedding_base(x[:, :, 0]) # Shape: [batch, 1+neg, dim]
        r_emb = self.r_embedding(x[:, 0, 1])      # Shape: [batch, dim] (relation is same for pos/neg)
        c_emb = self.c_embedding(x[:, 0, 2])      # Shape: [batch, dim] (concept is same for pos/neg)

        # Calculate fuzzy sets for the target concept C over all entities
        # Shape: [batch, n_entity_total]
        c_fs_batch = self._get_c_fs_batch(c_emb, all_e_emb)

        # Calculate fuzzy sets for the relation R over all entity pairs
        # Shape: [batch, n_entity_total, n_entity_total]
        r_fs_batch = self._get_r_fs_batch(r_emb, all_e_emb)

        # Calculate fuzzy set for exists R.C
        # Shape: [batch, n_entity_total] (membership for each entity x in exists R.C)
        exists_rc_fs = self._logical_exist(r_fs_batch, c_fs_batch)

        # Get membership degree for positive and negative entities
        # Need to gather the membership degree for the specific entities in the batch
        # x[:, :, 0] contains the entity IDs (indices for e_embedding_base)
        entity_ids = x[:, :, 0] # Shape: [batch, 1+neg]

        # Gather corresponding values from exists_rc_fs
        # exists_rc_fs has shape [batch, n_entity_total]
        # entity_ids has shape [batch, 1+neg]
        # We need to select using batch indices and entity indices
        batch_indices = torch.arange(x.size(0)).unsqueeze(1).expand_as(entity_ids) # Shape: [batch, 1+neg]
        dofm = exists_rc_fs[batch_indices, entity_ids] # Shape: [batch, 1+neg]

        # Calculate classification loss
        loss_pos = -torch.log(dofm[:, 0] + 1e-10)       # Positive samples
        loss_neg = -torch.log(1 - dofm[:, 1:] + 1e-10) # Negative samples
        return (loss_pos.mean() + loss_neg.mean()) / 2

    def forward_abox_ec_created(self, x):
        """ Calculates loss for ABox EC axioms involving created entities. """
        # x shape: [batch_size, 1 + num_neg, 2] (entity_id, concept_id)
        # Positive sample: x[:, 0, :]
        # Negative samples: x[:, 1:, :] (negative entities, same concept)

        # Entity IDs can be base or created. Need combined embedding lookup.
        entity_ids = x[:, :, 0] # Shape: [batch, 1+neg]
        concept_ids = x[:, 0, 1] # Shape: [batch] (concept is same for pos/neg)

        # Get embeddings
        # Use e_embedding_base directly as IDs should correspond to its indices
        e_emb = self.e_embedding_base(entity_ids) # Shape: [batch, 1+neg, dim]
        c_emb = self.c_embedding(concept_ids)     # Shape: [batch, dim]

        # Expand concept embedding for batch operation
        # Shape: [batch, 1+neg, dim]
        c_emb_expanded = c_emb.unsqueeze(1).expand(-1, x.size(1), -1)

        # Calculate direct C(a) membership using the scoring function (fc layers)
        emb = torch.cat([c_emb_expanded, e_emb], dim=-1) # Shape: [batch, 1+neg, 2*dim]

        if self.cfg.loss_type == 'c': # Classification loss
            # dofm = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
            dofm = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1) # Shape: [batch, 1+neg]
            loss_pos = -torch.log(dofm[:, 0] + 1e-10)
            loss_neg = -torch.log(1 - dofm[:, 1:] + 1e-10)
            return (loss_pos.mean() + loss_neg.mean()) / 2
        elif self.cfg.loss_type == 'r': # Ranking loss
            # scores = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1)).squeeze(dim=-1)
            scores = self.fc_0(emb).squeeze(dim=-1) # Shape: [batch, 1+neg]
            pos_scores = scores[:, 0] # Shape: [batch]
            neg_scores = scores[:, 1:] # Shape: [batch, neg]
            # Compare positive score against all negative scores
            margin_loss = -torch.nn.functional.logsigmoid(pos_scores.unsqueeze(-1) - neg_scores)
            return margin_loss.mean()
        else:
            raise ValueError(f"Invalid loss_type: {self.cfg.loss_type}")

    def forward_ggi(self, x, stage='train'):
        """ Calculates loss or scores for GGI triples (entity-relation-entity). """
        # x shape: [batch_size, 1 + num_neg, 3] (h_id, r_id, t_id) for train
        # x shape: [num_eval_triples, 3] for test (where num_eval_triples = num_entities * 2)

        # Assume h, r, t IDs refer to base embeddings
        e1_emb = self.e_embedding_base(x[:, :, 0]) # Head embeddings
        r_emb = self.r_embedding(x[:, :, 1])       # Relation embeddings
        e2_emb = self.e_embedding_base(x[:, :, 2]) # Tail embeddings

        # Score using a DistMult-like approach (or TransE if adapted)
        # Original code uses: score(h+r, t)
        emb = torch.cat([e1_emb + r_emb, e2_emb], dim=-1) # Shape: [batch, 1+neg, 2*dim] or [eval, 2*dim]

        if stage == 'train':
            if self.cfg.loss_type == 'c': # Classification loss
                # dofm = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
                dofm = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1) # Shape: [batch, 1+neg]
                loss_pos = -torch.log(dofm[:, 0] + 1e-10)
                loss_neg = -torch.log(1 - dofm[:, 1:] + 1e-10)
                return (loss_pos.mean() + loss_neg.mean()) / 2
            elif self.cfg.loss_type == 'r': # Ranking loss
                # scores = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1)).squeeze(dim=-1)
                scores = self.fc_0(emb).squeeze(dim=-1) # Shape: [batch, 1+neg]
                pos_scores = scores[:, 0]
                neg_scores = scores[:, 1:]
                margin_loss = -torch.nn.functional.logsigmoid(pos_scores.unsqueeze(-1) - neg_scores)
                return margin_loss.mean()
            else:
                raise ValueError(f"Invalid loss_type: {self.cfg.loss_type}")
        elif stage == 'test':
            # Return scores for evaluation
            # scores = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).flatten()
            scores = torch.sigmoid(self.fc_0(emb)).flatten() # Shape: [eval]
            return scores
        else:
             raise ValueError(f"Invalid stage: {stage}")


# --- Evaluation Functions --- (Modified to use dict lengths)
def get_ranks(logits, y, already_ts_dict, already_hs_dict, flag):
    """Calculates filtered ranks for head/tail prediction."""
    # logits: scores for all possible replacements [num_entities]
    # y: the positive triple (h, r, t) as tensor
    # flag: 'head' or 'tail' prediction

    logits_sorted_indices = torch.argsort(logits.cpu(), dim=-1, descending=True)

    if flag == 'head':
        true_entity_id = y[0].item() # True head ID
        # Find rank of the true head
        rank_raw = (logits_sorted_indices == true_entity_id).nonzero(as_tuple=True)[0].item() + 1
        # Get entities to filter (known objects for this ?h, r, t query)
        filter_key = (y[2].item(), y[1].item()) # (tail, relation)
        already = set(already_hs_dict.get(filter_key, []))
    elif flag == 'tail':
        true_entity_id = y[2].item() # True tail ID
        # Find rank of the true tail
        rank_raw = (logits_sorted_indices == true_entity_id).nonzero(as_tuple=True)[0].item() + 1
        # Get entities to filter (known objects for this h, r, ?t query)
        filter_key = (y[0].item(), y[1].item()) # (head, relation)
        already = set(already_ts_dict.get(filter_key, []))
    else:
        raise ValueError("flag must be 'head' or 'tail'")

    # Calculate filtered rank
    rank_filtered = rank_raw
    # Iterate through entities ranked higher than the true entity
    for i in range(rank_raw - 1):
        higher_ranked_entity_id = logits_sorted_indices[i].item()
        # If a higher-ranked entity is a known positive (and not the true entity itself), decrement rank
        if higher_ranked_entity_id in already and higher_ranked_entity_id != true_entity_id:
            rank_filtered -= 1

    # Calculate metrics (using filtered rank)
    h10_raw = (rank_raw <= 10)
    h100_raw = (rank_raw <= 100)
    h10_filter = (rank_filtered <= 10)
    h100_filter = (rank_filtered <= 100)

    return float(h10_raw), float(h100_raw), float(h10_filter), float(h100_filter)


def ggi_evaluate(model, loader, e_dict_len, device, already_ts_dict, already_hs_dict): # Pass length
    """Evaluates GGI performance using Hits@k."""
    model.eval()
    mh10_raw, mh100_raw, mh10_filter, mh100_filter = 0.0, 0.0, 0.0, 0.0
    num_samples = 0

    with torch.no_grad():
        # Use tqdm if available and loader is suitable
        try:
            loader_iter = tqdm.tqdm(loader, desc="GGI Evaluation")
        except:
            loader_iter = loader

        for X, y in loader_iter:
            X = X.to(device) # Shape [2 * num_entities, 3]
            y = y.to(device) # Shape [1, 3] -> squeeze -> [3]
            y_squeezed = y.squeeze(0)

            # Ensure model is compatible (adjust if using other KGE models)
            if model.__class__.__name__ == 'FALCON':
                logits = model.forward_ggi(X, stage='test') # Shape [2 * num_entities]
            # elif model.__class__.__name__ == 'KGCModel': # Example for other models
            #     logits = model.forward(X).flatten()
            else:
                 raise TypeError(f"Unsupported model type for ggi_evaluate: {model.__class__.__name__}")

            # Split logits for head and tail prediction
            logits_head, logits_tail = logits[:e_dict_len], logits[e_dict_len:]

            # Calculate ranks for head prediction (predicting y[0])
            h10r_h, h100r_h, h10f_h, h100f_h = get_ranks(logits_head, y_squeezed, already_ts_dict, already_hs_dict, flag='head')
            mh10_raw += h10r_h
            mh100_raw += h100r_h
            mh10_filter += h10f_h
            mh100_filter += h100f_h

            # Calculate ranks for tail prediction (predicting y[2])
            h10r_t, h100r_t, h10f_t, h100f_t = get_ranks(logits_tail, y_squeezed, already_ts_dict, already_hs_dict, flag='tail')
            mh10_raw += h10r_t
            mh100_raw += h100r_t
            mh10_filter += h10f_t
            mh100_filter += h100f_t

            num_samples += 1

    # Average metrics over all samples (each sample contributes 2 predictions: head and tail)
    total_predictions = num_samples * 2
    if total_predictions == 0: return 0.0, 0.0, 0.0, 0.0 # Avoid division by zero

    mh10_raw /= total_predictions
    mh100_raw /= total_predictions
    mh10_filter /= total_predictions
    mh100_filter /= total_predictions

    return round(mh10_raw, 3), round(mh100_raw, 3), round(mh10_filter, 3), round(mh100_filter, 3)


# --- Utility Functions ---
def iterator(dataloader):
    """Creates an infinite iterator over a dataloader."""
    while True:
        for data in dataloader:
            yield data

# --- Argument Parsing ---
def parse_args(args=None):
    parser = argparse.ArgumentParser(description="FALCON model training for EL++")
    # Data path argument
    parser.add_argument('--data_path', default='../../data/el/processed', type=str, # Default to a 'processed' subdirectory
                        help='Path to the directory containing preprocessed data files (tsv/txt)')
    # Tunable hyperparameters
    parser.add_argument('--model', default='FALCON', type=str, help='Model name (currently only FALCON supported)')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--emb_dim', default=32, type=int, help='Embedding dimension')
    parser.add_argument('--num_ng', default=8, type=int, help='Number of negative samples')
    # parser.add_argument('--n_models', default=2, type=int) # Seems unused in this script's main loop
    parser.add_argument('--loss_type', default='r', type=str, choices=['r', 'c'], help='Loss type: r for ranking, c for classification')
    parser.add_argument('--bs_kgc', default=64, type=int, help='Batch size for KGC (GGI) data') # Renamed for clarity
    parser.add_argument('--bs_ee', default=64, type=int, help='Batch size for GGI data (synonym for bs_kgc)')
    parser.add_argument('--bs_ec', default=64, type=int, help='Batch size for ABox EC data')
    parser.add_argument('--bs_ec_created', default=64, type=int, help='Batch size for ABox EC created data') # Added specific BS
    parser.add_argument('--bs_tbox_name', default=64, type=int, help='Batch size for TBox name axioms')
    parser.add_argument('--bs_tbox_desc', default=4, type=int, help='Batch size for TBox description axioms')
    parser.add_argument('--anon_e', default=4, type=int, help='Number of anonymous entities')
    # parser.add_argument('--n_e', default=1500, type=int) # No longer needed, inferred from data
    parser.add_argument('--n_abox_ec_created', default=1500, type=int, help='Number of ABox EC axioms to create')
    # parser.add_argument('--n_inconsistent', default=0, type=int) # Seems unused
    parser.add_argument('--t_norm', default='product', type=str, choices=['product', 'minmax', 'Łukasiewicz'], help='T-norm for fuzzy logic')
    parser.add_argument('--residuum', default='notCorD', type=str, help='Residuum operator')
    parser.add_argument('--max_measure', default='max', type=str, help='Aggregation for CC loss (max, pmeanX)')
    # KGE model specific (if other models were used)
    # parser.add_argument('--scoring_fct_norm', default=2, type=float)
    # parser.add_argument('--kernel_size', default=3, type=int)
    # parser.add_argument('--convkb_drop_prob', default=0.2, type=float)
    # parser.add_argument('--out_channels', default=8, type=int)
    # Untunable execution parameters
    # parser.add_argument('--data_root', default='../../data/el/', type=str) # Replaced by data_path
    parser.add_argument('--max_steps', default=100000, type=int, help='Maximum training steps')
    parser.add_argument('--valid_interval', default=100, type=int, help='Validation interval (steps)')
    parser.add_argument('--tolerance', default=10, type=int, help='Early stopping tolerance (validation intervals)')
    parser.add_argument('--verbose', default=1, type=int, help='Verbosity level (1 for tqdm, 0 for silent)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use (-1 for CPU)')
    return parser.parse_args(args)

# --- Main Execution ---
if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)

    # --- Data Loading ---
    # Pass cfg to get_data
    tbox_name, tbox_desc, abox_ec, abox_ec_created, abox_ee_train, abox_ee_test, \
        c_dict, e_dict, e_dict_more, r_dict, already_ts_dict, already_hs_dict = get_data(cfg)

    # Get dictionary lengths for model and datasets
    c_dict_len = len(c_dict)
    e_dict_len = len(e_dict) # Base entities from file
    e_dict_more_len = len(e_dict_more) # Base + created entities
    r_dict_len = len(r_dict)

    print(f'Concepts: {c_dict_len}\tEntities (Base): {e_dict_len}\tEntities (Created): {e_dict_more_len - e_dict_len}\tAnon: {cfg.anon_e}\tRelations: {r_dict_len}', flush=True)
    print(f'TBox Name Axioms: {len(tbox_name)}\tTBox Desc Axioms: {len(tbox_desc)}', flush=True)
    print(f'ABox EC: {len(abox_ec)}\tABox EC Created: {len(abox_ec_created)}', flush=True)
    print(f'ABox EE Train: {len(abox_ee_train)}\tABox EE Test: {len(abox_ee_test)}', flush=True)

    # --- Dataset Creation ---
    # Pass lengths instead of dictionaries where appropriate
    ggi_dataset_train = GGIDataset(cfg, abox_ee_train, e_dict_len, already_ts_dict, already_hs_dict, stage='train')
    ggi_dataset_test = GGIDataset(cfg, abox_ee_test, e_dict_len, already_ts_dict, already_hs_dict, stage='test')
    abox_ec_dataset = AboxECDataset(cfg, abox_ec, e_dict_len)
    # Pass e_dict_more_len as it includes created entities which might be sampled as negatives
    abox_ec_created_dataset = AboxECCreatedDataset(cfg, abox_ec_created, e_dict_more_len, c_dict_len)
    tbox_name_dataset = NaiveDataset(torch.tensor(tbox_name.values)) if not tbox_name.empty else None
    tbox_desc_dataset = NaiveDataset(tbox_desc) if tbox_desc else None

    # --- DataLoader Creation ---
    # Use appropriate batch sizes from cfg
    ggi_dataloader_train = torch.utils.data.DataLoader(
        dataset=ggi_dataset_train, batch_size=cfg.bs_ee, shuffle=True, drop_last=True # num_workers=4
    ) if len(ggi_dataset_train) > 0 else None

    ggi_dataloader_test = torch.utils.data.DataLoader(
        dataset=ggi_dataset_test, batch_size=1, shuffle=False, drop_last=False # num_workers=8
    ) if len(ggi_dataset_test) > 0 else None

    abox_ec_dataloader = torch.utils.data.DataLoader(
        dataset=abox_ec_dataset, batch_size=cfg.bs_ec, shuffle=True, drop_last=True # num_workers=4
    ) if len(abox_ec_dataset) > 0 else None

    abox_ec_created_dataloader = torch.utils.data.DataLoader(
        dataset=abox_ec_created_dataset, batch_size=cfg.bs_ec_created, shuffle=True, drop_last=True # num_workers=4
    ) if len(abox_ec_created_dataset) > 0 else None

    tbox_name_dataloader = torch.utils.data.DataLoader(
        dataset=tbox_name_dataset, batch_size=cfg.bs_tbox_name, shuffle=True, drop_last=True
    ) if tbox_name_dataset else None

    tbox_desc_dataloader = torch.utils.data.DataLoader(
        dataset=tbox_desc_dataset, batch_size=cfg.bs_tbox_desc, shuffle=True, drop_last=True
    ) if tbox_desc_dataset else None

    # Wrap non-empty dataloaders in infinite iterators
    dataloaders = {}
    if ggi_dataloader_train: dataloaders['ggi'] = iterator(ggi_dataloader_train)
    if abox_ec_dataloader: dataloaders['abox_ec'] = iterator(abox_ec_dataloader)
    if abox_ec_created_dataloader: dataloaders['abox_ec_created'] = iterator(abox_ec_created_dataloader)
    if tbox_name_dataloader: dataloaders['tbox_name'] = iterator(tbox_name_dataloader)
    if tbox_desc_dataloader: dataloaders['tbox_desc'] = iterator(tbox_desc_dataloader)

    if not dataloaders:
        print("Error: No data loaded. Exiting.")
        sys.exit(1)

    # --- Model and Optimizer ---
    device = torch.device(f'cuda:{cfg.gpu}' if cfg.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Pass dictionary lengths to the model
    model = FALCON(c_dict_len, e_dict_more_len, r_dict_len, cfg, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # --- Training Loop ---
    weights = {'ggi': 5, 'abox_ec': 1, 'abox_ec_created': 1, 'tbox_name': 1, 'tbox_desc': 1}
    total_weight = sum(weights[k] for k in dataloaders.keys()) # Adjust total weight based on available data

    if cfg.verbose:
        ranger = tqdm.tqdm(range(cfg.max_steps), desc="Training Steps")
    else:
        ranger = range(cfg.max_steps)

    losses_log = [] # Log average loss per interval
    results = [] # Log validation results
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda')) # Enable AMP only for CUDA

    best_metric = -1.0 # Track best validation metric (e.g., H@10 Filter)
    steps_since_best = 0

    # Global dictionaries needed for forward pass string parsing (make accessible)
    # This is a workaround for not storing them in the model object itself.
    global_c_dict = c_dict
    global_r_dict = r_dict

    for step in ranger:
        model.train()
        optimizer.zero_grad()

        total_loss = 0
        loss_components = {}

        # Generate anonymous entity embeddings for this step
        # Ensure they are created on the correct device directly
        anon_e_emb_1 = model.e_embedding_base.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim), device=device)
        anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim, device=device)
        torch.nn.init.xavier_uniform_(anon_e_emb_2)
        anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            # Calculate loss for each available data type
            if 'ggi' in dataloaders:
                loss_ggi = model.forward_ggi(next(dataloaders['ggi']).to(device).long())
                total_loss += loss_ggi * weights['ggi']
                loss_components['ggi'] = loss_ggi.item()

            if 'abox_ec' in dataloaders:
                loss_abox_ec = model.forward_abox_ec(next(dataloaders['abox_ec']).to(device).long(), anon_e_emb)
                total_loss += loss_abox_ec * weights['abox_ec']
                loss_components['ec'] = loss_abox_ec.item()

            if 'abox_ec_created' in dataloaders:
                loss_abox_ec_created = model.forward_abox_ec_created(next(dataloaders['abox_ec_created']).to(device).long())
                total_loss += loss_abox_ec_created * weights['abox_ec_created']
                loss_components['ec_cr'] = loss_abox_ec_created.item()

            if 'tbox_name' in dataloaders:
                tbox_name_batch = next(dataloaders['tbox_name']).to(device).long()
                loss_tbox_name = model.forward_name(tbox_name_batch, anon_e_emb)
                total_loss += loss_tbox_name * weights['tbox_name']
                loss_components['t_name'] = loss_tbox_name.item()

            if 'tbox_desc' in dataloaders:
                loss_tbox_desc_batch = []
                # Need global dicts for forward call here
                c_dict = global_c_dict
                r_dict = global_r_dict
                for axiom_str in next(dataloaders['tbox_desc']):
                    fs = model.forward(axiom_str, anon_e_emb) # Pass reconstructed string
                    loss_tbox_desc_batch.append(model.get_cc_loss(fs))
                loss_tbox_desc = sum(loss_tbox_desc_batch) / len(loss_tbox_desc_batch) if loss_tbox_desc_batch else torch.tensor(0.0).to(device)
                total_loss += loss_tbox_desc * weights['tbox_desc']
                loss_components['t_desc'] = loss_tbox_desc.item()

            # Average the total loss by total weight
            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0).to(device) # No loss if no data

        # Backpropagation
        if total_weight > 0:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses_log.append(loss.item())
        else:
            losses_log.append(0.0) # Log zero loss if no data

        # Logging loss components (optional)
        if cfg.verbose and (step + 1) % (cfg.valid_interval // 10 + 1) == 0:
             loss_str = ", ".join([f"{k}:{v:.4f}" for k, v in loss_components.items()])
             ranger.set_postfix_str(f"Loss: {loss.item():.4f} ({loss_str})")


        # --- Validation ---
        if (step + 1) % cfg.valid_interval == 0:
            avg_loss = sum(losses_log) / len(losses_log) if losses_log else 0.0
            print(f'\nStep {step+1}/{cfg.max_steps} - Avg Loss: {avg_loss:.4f}', flush=True)
            losses_log = [] # Reset loss log

            if ggi_dataloader_test:
                mh10_raw, mh100_raw, mh10_filter, mh100_filter = ggi_evaluate(
                    model, ggi_dataloader_test, e_dict_len, device, already_ts_dict, already_hs_dict
                )
                print(f'Validation GGI - H@10(R): {mh10_raw}, H@100(R): {mh100_raw}, H@10(F): {mh10_filter}, H@100(F): {mh100_filter}', flush=True)
                results.append([mh10_raw, mh100_raw, mh10_filter, mh100_filter])

                # Early stopping check
                current_metric = mh10_filter # Use H@10 Filter for early stopping
                if current_metric > best_metric:
                    best_metric = current_metric
                    steps_since_best = 0
                    # Optionally save the best model checkpoint here
                    # torch.save(model.state_dict(), "best_model.pt")
                    print(f"  New best validation metric: {best_metric:.4f}", flush=True)
                else:
                    steps_since_best += 1
                    print(f"  Validation metric did not improve ({current_metric:.4f} vs best {best_metric:.4f}). Tolerance left: {cfg.tolerance - steps_since_best}", flush=True)

                if steps_since_best >= cfg.tolerance:
                    print(f"Early stopping triggered after {step + 1} steps.", flush=True)
                    break
            else:
                print("Skipping validation (no test GGI data).", flush=True)
                # If no validation data, disable early stopping or use training loss?
                # For now, just continue training until max_steps.

    # --- Final Results ---
    print("\nTraining finished.", flush=True)
    if results:
        results_tensor = torch.tensor(results)
        # Find the best result based on the chosen metric (H@10 Filter - index 2)
        best_epoch_idx = results_tensor[:, 2].argmax()
        final_results = results_tensor[best_epoch_idx]
        mh10_raw, mh100_raw, mh10_filter, mh100_filter = final_results.tolist()
        print(f'Best Validation Result (at validation step {best_epoch_idx + 1}):', flush=True)
        print(f'  H@10 (Raw): {mh10_raw:.3f}\tH@100 (Raw): {mh100_raw:.3f}', flush=True)
        print(f'  H@10 (Filter): {mh10_filter:.3f}\tH@100 (Filter): {mh100_filter:.3f}', flush=True)
    else:
        print("No validation results recorded.", flush=True)

