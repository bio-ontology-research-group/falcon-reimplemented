from email.policy import default
from multiprocessing.sharedctypes import Value
from tabnanny import verbose
import torch
import pdb
import argparse
import os
import sys # Added import
import pandas as pd
import numpy as np
import random
import tqdm
import re # Keep re
# import pickle # Moved to utils
import torch.utils.checkpoint as checkpoint
from sklearn.metrics import roc_auc_score # Keep for potential direct use
from sklearn.metrics import average_precision_score # Keep for potential direct use
from sklearn.metrics import precision_recall_curve # Keep for potential direct use
import warnings
from pathlib import Path # Use pathlib

# Import shared functions from cfalcon.utils
from cfalcon.utils import (
    load_obj, save_obj, reconstruct_functional_syntax,
    read_list_from_file, read_tbox_test_axioms, concept_replacer,
    tbox_test_neg_generator, get_abox_ec_created, compute_metrics,
    iterator
)

warnings.filterwarnings('ignore')


# Removed load_obj
# Removed save_obj
# Removed reconstruct_functional_syntax
# Removed read_list_from_file
# Removed read_tbox_test_axioms
# Removed concept_replacer
# Removed tbox_test_neg_generator
# Removed get_abox_ec_created
# Removed compute_metrics

def get_data(cfg):
    """Loads data from preprocessed files specified by cfg.data_path."""
    data_path = Path(cfg.data_path)
    # Assume test files are in the *same* directory for simplicity,
    # or user needs to provide separate paths.
    # Let's assume tbox_test_pos.txt and tbox_test_neg.txt are manually created or generated
    # in the cfg.data_path directory.
    tbox_test_pos_file = data_path / "tbox_test_pos.txt"
    tbox_test_neg_file = data_path / "tbox_test_neg.txt"

    print(f"Loading data from: {data_path.resolve()}")

    # Load concepts, relations, entities using utility function
    all_concepts_list = read_list_from_file(data_path / "concepts.txt")
    all_relations_list = read_list_from_file(data_path / "relations.txt")
    all_entities_list = read_list_from_file(data_path / "entities.txt")

    # Add special concepts/relations if needed
    if 'owl:Thing' not in all_concepts_list: all_concepts_list.append('owl:Thing')
    if 'owl:Nothing' not in all_concepts_list: all_concepts_list.append('owl:Nothing')
    if 'subClassOf' not in all_relations_list: all_relations_list.append('subClassOf')

    # Load ABox data
    try:
        abox_ec = pd.read_csv(data_path / "abox_ec.tsv", sep='\t', header=None, names=['h', 't'], keep_default_na=False)
        # Add relation column (assuming 'hasPhenotype' for HPO)
        pheno_relation = '<http://purl.obolibrary.org/obo/IAO_0000219>' # Example: 'has phenotype' IRI
        if pheno_relation not in all_relations_list: all_relations_list.append(pheno_relation)
        abox_ec['r'] = pheno_relation
        abox_ec = abox_ec[['h', 'r', 't']]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: {data_path / 'abox_ec.tsv'} not found or empty. ABox EC will be empty.")
        abox_ec = pd.DataFrame(columns=['h', 'r', 't'])

    try:
        abox_ee = pd.read_csv(data_path / "abox_ee.tsv", sep='\t', header=None, names=['h', 'r', 't'], keep_default_na=False)
         # Filter ABox EE based on cfg.n_e entities (if needed, mimicking original logic)
        if cfg.n_e > 0 and len(all_entities_list) > cfg.n_e:
             print(f"Filtering entities to top {cfg.n_e}")
             entities_to_keep = set(all_entities_list[:cfg.n_e])
             abox_ee = abox_ee[abox_ee['h'].isin(entities_to_keep) & abox_ee['t'].isin(entities_to_keep)]
             abox_ec = abox_ec[abox_ec['h'].isin(entities_to_keep)]
             all_entities_list = all_entities_list[:cfg.n_e] # Keep only the selected entities
        else:
             entities_to_keep = set(all_entities_list) # Keep all

    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: {data_path / 'abox_ee.tsv'} not found or empty. ABox EE will be empty.")
        abox_ee = pd.DataFrame(columns=['h', 'r', 't'])
        entities_to_keep = set(all_entities_list) # Keep all entities even if EE is empty


    # Load and reconstruct TBox data for training using utility function
    tbox_name_list = []
    tbox_desc_train = []
    tbox_train_axioms_reconstructed = [] # Store all reconstructed train axioms for neg generation
    try:
        with open(data_path / "tbox.tsv", 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if not parts: continue
                axiom_type = parts[0]
                axiom_parts = parts[1:]
                reconstructed = reconstruct_functional_syntax(axiom_type, axiom_parts) # Use util func

                if reconstructed:
                    if isinstance(reconstructed, list):
                        tbox_train_axioms_reconstructed.extend(reconstructed)
                        for axiom_str in reconstructed:
                            name_match = re.match(r'ObjectIntersectionOf\((<.*?>|owl:Thing) ObjectComplementOf\((<.*?>|owl:Thing)\)\)', axiom_str, re.M|re.I)
                            if name_match:
                                matched = name_match.groups()
                                tbox_name_list.append([matched[0], 'subClassOf', matched[1]])
                            else:
                                tbox_desc_train.append(axiom_str)
                    else:
                        axiom_str = reconstructed
                        tbox_train_axioms_reconstructed.append(axiom_str)
                        name_match = re.match(r'ObjectIntersectionOf\((<.*?>|owl:Thing) ObjectComplementOf\((<.*?>|owl:Thing)\)\)', axiom_str, re.M|re.I)
                        if name_match:
                            matched = name_match.groups()
                            tbox_name_list.append([matched[0], 'subClassOf', matched[1]])
                        else:
                            tbox_desc_train.append(axiom_str)
    except FileNotFoundError:
        print(f"Warning: {data_path / 'tbox.tsv'} not found. Training TBox will be empty.")

    tbox_name_train = pd.DataFrame(tbox_name_list, columns=['h', 'r', 't'])

    # Load TBox test axioms (positive and negative) using utility function
    tbox_test_pos = read_tbox_test_axioms(tbox_test_pos_file)
    if not tbox_test_pos:
         print(f"Warning: Positive test TBox file '{tbox_test_pos_file}' not found or empty.")

    try:
        tbox_test_neg = read_tbox_test_axioms(tbox_test_neg_file)
        if not tbox_test_neg:
             print(f"Warning: Negative test TBox file '{tbox_test_neg_file}' not found or empty. Attempting generation.")
             raise FileNotFoundError # Trigger generation
    except FileNotFoundError:
        if tbox_train_axioms_reconstructed or tbox_test_pos:
            print("Generating negative TBox test axioms...")
            num_neg_to_generate = len(tbox_test_pos) if tbox_test_pos else 1000 # Generate based on pos count or default
            # Use utility function for generation
            tbox_test_neg = tbox_test_neg_generator(tbox_train_axioms_reconstructed, tbox_test_pos, all_concepts_list, k=num_neg_to_generate)
            # Save generated negatives
            try:
                with open(tbox_test_neg_file, 'w', encoding='utf-8') as f:
                    for axiom in tbox_test_neg:
                        f.write(f"{axiom}\n")
                print(f"Saved generated negative axioms to {tbox_test_neg_file}")
            except IOError as e:
                print(f"Error saving generated negative axioms: {e}")
        else:
             print("Warning: Cannot generate negative test axioms as no positive axioms were loaded.")
             tbox_test_neg = []


    # Generate created entities for ABox EC using utility function
    abox_ec_created = get_abox_ec_created(all_concepts_list, k=cfg.n_abox_ec_created)
    created_entities_list = list(abox_ec_created['h'].unique())

    # Finalize concept/relation/entity lists (ensure all used IRIs are included)
    # (Simplified check - assumes initial load + ABox check is sufficient)
    all_concepts_list = sorted(list(set(all_concepts_list)))
    all_relations_list = sorted(list(set(all_relations_list)))
    all_entities_list = sorted(list(entities_to_keep)) # Use the filtered list if n_e was applied
    created_entities_list = sorted(list(set(created_entities_list)))


    # Create dictionaries
    c_dict = {k: v for v, k in enumerate(all_concepts_list)}
    e_dict = {k: v for v, k in enumerate(all_entities_list)}
    e_dict_created = {k: v + len(e_dict) for v, k in enumerate(created_entities_list)}
    e_dict_more = {**e_dict, **e_dict_created}
    r_dict = {k: v for v, k in enumerate(all_relations_list)}

    # --- Map IRIs to IDs ---
    tbox_name_train['h'] = tbox_name_train['h'].map(c_dict.get)
    tbox_name_train['r'] = tbox_name_train['r'].map(r_dict.get)
    tbox_name_train['t'] = tbox_name_train['t'].map(c_dict.get)
    tbox_name_train.dropna(inplace=True)
    tbox_name_train = tbox_name_train.astype(int)

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
    # Use pickle files if they exist (mimicking original logic), otherwise split loaded data
    abox_ee_train_path = data_path / 'abox_ee_train.pkl'
    abox_ee_test_path = data_path / 'abox_ee_test.pkl'
    try:
        abox_ee_train = load_obj(abox_ee_train_path) # Use util func
        abox_ee_test = load_obj(abox_ee_test_path) # Use util func
        print("Loaded existing GGI train/test split from .pkl files.")
        # Ensure loaded data uses current entity dictionary
        abox_ee_train = abox_ee_train[abox_ee_train['h'].isin(e_dict) & abox_ee_train['t'].isin(e_dict)]
        abox_ee_test = abox_ee_test[abox_ee_test['h'].isin(e_dict) & abox_ee_test['t'].isin(e_dict)]
    except FileNotFoundError:
        print('Generating new GGI train/test split.', flush=True)
        if len(abox_ee) > 0:
            # Simple random split, consider more robust splitting methods
            abox_ee_test = abox_ee.sample(frac=0.2, random_state=42)
            abox_ee_train = abox_ee.drop(abox_ee_test.index)
            try:
                save_obj(abox_ee_train, abox_ee_train_path) # Use util func
                save_obj(abox_ee_test, abox_ee_test_path) # Use util func
                print(f"Saved new GGI split to {abox_ee_train_path} and {abox_ee_test_path}")
            except IOError as e:
                 print(f"Error saving GGI split: {e}")
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
            already_ts_dict[(int(record[0]), int(record[1]))] = [int(x) for x in record[2]]
        for record in already_hs:
            already_hs_dict[(int(record[0]), int(record[1]))] = [int(x) for x in record[2]]

    # Return mapped data, reconstructed descriptions, test axioms, and dictionaries
    return tbox_name_train, tbox_desc_train, tbox_test_pos, tbox_test_neg, \
           abox_ec, abox_ec_created, abox_ee_train, abox_ee_test, \
           c_dict, e_dict, e_dict_more, r_dict, already_ts_dict, already_hs_dict


# --- Datasets --- (Adapted like in el.py)
class GGIDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, e_dict_len, already_ts_dict, already_hs_dict, stage):
        super().__init__()
        self.stage = stage
        self.cfg = cfg
        self.e_dict_len = e_dict_len
        self.data = torch.tensor(data.values)
        self.all_candidate = torch.arange(self.e_dict_len).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng//2, 1, dtype=torch.long)
        self.already_ts_dict = already_ts_dict
        self.already_hs_dict = already_hs_dict

    def sampling(self, pos):
        head, rel, tail = pos
        ts_key = (head.item(), rel.item())
        hs_key = (tail.item(), rel.item())
        already_ts = torch.tensor(self.already_ts_dict.get(ts_key, []), dtype=torch.long)
        already_hs = torch.tensor(self.already_hs_dict.get(hs_key, []), dtype=torch.long)

        neg_pool_t = torch.ones(self.e_dict_len, dtype=torch.bool)
        if len(already_ts) > 0: neg_pool_t[already_ts] = 0
        neg_pool_t = neg_pool_t.nonzero().squeeze()

        neg_pool_h = torch.ones(self.e_dict_len, dtype=torch.bool)
        if len(already_hs) > 0: neg_pool_h[already_hs] = 0
        neg_pool_h = neg_pool_h.nonzero().squeeze()

        num_neg_t = len(neg_pool_t) if neg_pool_t.dim() > 0 else 0
        num_neg_h = len(neg_pool_h) if neg_pool_h.dim() > 0 else 0
        num_needed = self.cfg.num_ng // 2

        neg_t = neg_pool_t[torch.randint(num_neg_t, (num_needed,))].unsqueeze(-1) if num_neg_t >= num_needed else \
                neg_pool_t[torch.randperm(num_neg_t)].unsqueeze(-1) if num_neg_t > 0 else \
                torch.randint(self.e_dict_len, (num_needed, 1), dtype=torch.long) # Fallback

        neg_h = neg_pool_h[torch.randint(num_neg_h, (num_needed,))].unsqueeze(-1) if num_neg_h >= num_needed else \
                neg_pool_h[torch.randperm(num_neg_h)].unsqueeze(-1) if num_neg_h > 0 else \
                torch.randint(self.e_dict_len, (num_needed, 1), dtype=torch.long) # Fallback

        # Ensure correct size if sampling yielded fewer than needed
        if len(neg_t) < num_needed:
             neg_t = torch.cat([neg_t, torch.randint(self.e_dict_len, (num_needed - len(neg_t), 1))], dim=0)
        if len(neg_h) < num_needed:
             neg_h = torch.cat([neg_h, torch.randint(self.e_dict_len, (num_needed - len(neg_h), 1))], dim=0)

        return neg_t[:num_needed], neg_h[:num_needed]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        if self.stage == 'train':
            neg_t, neg_h = self.sampling(pos)
            if neg_t.shape[0] != self.cfg.num_ng // 2 or neg_h.shape[0] != self.cfg.num_ng // 2:
                 neg_t = torch.randint(self.e_dict_len, (self.cfg.num_ng // 2, 1), dtype=torch.long)
                 neg_h = torch.randint(self.e_dict_len, (self.cfg.num_ng // 2, 1), dtype=torch.long)

            pos_h_expanded = pos[0].expand_as(neg_t)
            pos_r_expanded_t = pos[1].expand_as(neg_t)
            pos_r_expanded_h = pos[1].expand_as(neg_h)
            pos_t_expanded = pos[2].expand_as(neg_h)

            replace_tail = torch.cat([pos_h_expanded, pos_r_expanded_t, neg_t], dim=-1)
            replace_head = torch.cat([neg_h, pos_r_expanded_h, pos_t_expanded], dim=-1)
            return torch.cat([pos.unsqueeze(0), replace_tail, replace_head], dim=0)

        elif self.stage == 'test':
            pos_h_expanded = pos[0].expand_as(self.all_candidate)
            pos_r_expanded = pos[1].expand_as(self.all_candidate)
            pos_t_expanded = pos[2].expand_as(self.all_candidate)
            replace_tail = torch.cat([pos_h_expanded, pos_r_expanded, self.all_candidate], dim=-1)
            replace_head = torch.cat([self.all_candidate, pos_r_expanded, pos_t_expanded], dim=-1)
            return torch.cat([replace_head, replace_tail], dim=0), pos
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

class AboxECDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, e_dict_len):
        super().__init__()
        self.cfg = cfg
        self.e_dict_len = e_dict_len
        self.data = torch.tensor(data.values)
        self.all_candidate = torch.arange(self.e_dict_len).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng, 1, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx] # h, r, t
        negs = torch.randint(self.e_dict_len, (self.cfg.num_ng, 1), dtype=torch.long)
        pos_r_expanded = pos[1].expand_as(negs)
        pos_t_expanded = pos[2].expand_as(negs)
        replace_head = torch.cat([negs, pos_r_expanded, pos_t_expanded], dim=-1)
        return torch.cat([pos.unsqueeze(0), replace_head], dim=0)

class AboxECCreatedDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, e_dict_len, c_dict_len):
        super().__init__()
        self.cfg = cfg
        self.e_dict_len = e_dict_len
        self.c_dict_len = c_dict_len
        self.data = torch.tensor(data.values) # h, t (entity, concept)
        self.all_candidate_entities = torch.arange(self.e_dict_len).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng, 1, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx] # h, t
        pos_h = pos[0]
        pos_t = pos[1]
        neg_entities = torch.randint(self.e_dict_len, (self.cfg.num_ng, 1), dtype=torch.long)
        pos_t_expanded = pos_t.expand_as(neg_entities)
        neg_samples = torch.cat([neg_entities, pos_t_expanded], dim=-1)
        return torch.cat([pos.unsqueeze(0), neg_samples], dim=0)

class NaiveDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- Model --- (Adapted like in el.py)
class FALCON(torch.nn.Module):
    def __init__(self, c_dict_len, e_dict_len, r_dict_len, cfg, device):
        super().__init__()
        self.c_dict_len = c_dict_len
        self.e_dict_len = e_dict_len
        self.r_dict_len = r_dict_len
        self.n_entity_base = e_dict_len
        self.anon_e = cfg.anon_e
        self.n_entity_total = self.n_entity_base + self.anon_e
        self.cfg = cfg

        self.c_embedding = torch.nn.Embedding(self.c_dict_len, cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(self.r_dict_len, cfg.emb_dim)
        self.e_embedding_base = torch.nn.Embedding(self.n_entity_base, cfg.emb_dim)
        self.fc_0 = torch.nn.Linear(cfg.emb_dim * 2, 1)

        torch.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.e_embedding_base.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_0.weight.data)

        self.max_measure = cfg.max_measure
        self.t_norm = cfg.t_norm
        self.nothing_fs = torch.zeros(self.n_entity_total).to(device)
        self.residuum = cfg.residuum
        self.device = device

    def _logical_and(self, x, y):
        if self.t_norm == 'product': return x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2); y = y.expand_as(x)
            return torch.cat([x, y], dim=-2).min(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2); y = y.expand_as(x)
            return (((x + y -1) > 0) * (x + y - 1)).squeeze(dim=-2)
        else: raise ValueError
    def _logical_or(self, x, y):
        if self.t_norm == 'product': return x + y - x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2); y = y.expand_as(x)
            return torch.cat([x, y], dim=-2).max(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2); y = y.expand_as(x)
            return 1 - ((((1-x) + (1-y) -1) > 0) * ((1-x) + (1-y) - 1)).squeeze(dim=-2)
        else: raise ValueError
    def _logical_not(self, x): return 1 - x
    def _logical_residuum(self, r_fs, c_fs):
        if self.residuum == 'notCorD':
            c_fs_expanded = c_fs.unsqueeze(-2)
            return self._logical_or(self._logical_not(r_fs), c_fs_expanded)
        else: raise ValueError
    def _logical_exist(self, r_fs, c_fs):
        c_fs_expanded = c_fs.unsqueeze(-2)
        return self._logical_and(r_fs, c_fs_expanded).max(dim=-1)[0]
    def _logical_forall(self, r_fs, c_fs):
        return self._logical_residuum(r_fs, c_fs).min(dim=-1)[0]

    def _get_all_entity_embeddings(self, anon_e_emb):
        if anon_e_emb.shape[0] != self.anon_e or anon_e_emb.shape[1] != self.cfg.emb_dim:
             raise ValueError(f"Incorrect shape for anon_e_emb: {anon_e_emb.shape}")
        if anon_e_emb.device != self.device: anon_e_emb = anon_e_emb.to(self.device)
        return torch.cat([self.e_embedding_base.weight, anon_e_emb], dim=0)
    def _get_c_fs(self, c_emb, all_e_emb):
        c_emb_expanded = c_emb.expand_as(all_e_emb)
        emb = torch.cat([c_emb_expanded, all_e_emb], dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
    def _get_c_fs_batch(self, c_emb_batch, all_e_emb):
        num_entities = all_e_emb.size(0); batch_size = c_emb_batch.size(0)
        all_e_emb_expanded = all_e_emb.unsqueeze(0).expand(batch_size, num_entities, self.cfg.emb_dim)
        c_emb_batch_expanded = c_emb_batch.unsqueeze(1).expand(batch_size, num_entities, self.cfg.emb_dim)
        emb = torch.cat([c_emb_batch_expanded, all_e_emb_expanded], dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
    def _get_r_fs(self, r_emb, all_e_emb):
        num_entities = all_e_emb.size(0)
        e_emb_repeated_rows = all_e_emb.repeat_interleave(num_entities, dim=0)
        e_emb_repeated_cols = all_e_emb.repeat(num_entities, 1)
        l_emb_flat = e_emb_repeated_rows + r_emb.unsqueeze(0)
        r_emb_flat = e_emb_repeated_cols
        emb_flat = torch.cat([l_emb_flat, r_emb_flat], dim=-1)
        fs_flat = torch.sigmoid(self.fc_0(emb_flat)).squeeze(dim=-1)
        return fs_flat.view(num_entities, num_entities)
    def _get_r_fs_batch(self, r_emb_batch, all_e_emb):
        num_entities = all_e_emb.size(0); batch_size = r_emb_batch.size(0)
        e_emb_expanded_l = all_e_emb.unsqueeze(0).unsqueeze(2).expand(batch_size, num_entities, num_entities, self.cfg.emb_dim)
        e_emb_expanded_r = all_e_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, num_entities, num_entities, self.cfg.emb_dim)
        r_emb_expanded = r_emb_batch.unsqueeze(1).unsqueeze(2).expand(batch_size, num_entities, num_entities, self.cfg.emb_dim)
        l_emb = e_emb_expanded_l + r_emb_expanded; r_emb = e_emb_expanded_r
        emb = torch.cat([l_emb, r_emb], dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    # Forward needs global dicts or passed dicts
    def forward(self, axiom_str, anon_e_emb, c_dict, r_dict):
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)
        if axiom_str[0] == '<' or axiom_str == 'owl:Thing':
            try:
                c_id = torch.tensor(c_dict[axiom_str]).to(self.device)
                c_emb = self.c_embedding(c_id)
                # return self._get_c_fs(c_emb, all_e_emb)
                return checkpoint.checkpoint(self._get_c_fs, c_emb, all_e_emb, use_reentrant=False)
            except KeyError: return self.nothing_fs.to(all_e_emb.device)
        elif axiom_str == 'owl:Nothing': return self.nothing_fs.to(all_e_emb.device)
        elif axiom_str.startswith('ObjectIntersectionOf('):
            content = axiom_str[21:-1]; split_index = -1; paren_level = 0
            for i, char in enumerate(content):
                if char == '(': paren_level += 1
                elif char == ')': paren_level -= 1
                elif char == ' ' and paren_level == 0: split_index = i; break
            if split_index == -1: # Handle simple pair like ObjectIntersectionOf(<A> <B>)
                 parts = content.split(' ', 1)
                 if len(parts) == 2: left_str, right_str = parts[0], parts[1]
                 else: raise ValueError(f"Cannot parse ObjectIntersectionOf: {axiom_str}")
            else: left_str = content[:split_index].strip(); right_str = content[split_index:].strip()
            fs_left = self.forward(left_str, anon_e_emb, c_dict, r_dict)
            fs_right = self.forward(right_str, anon_e_emb, c_dict, r_dict)
            return self._logical_and(fs_left, fs_right)
        elif axiom_str.startswith('ObjectUnionOf('):
            content = axiom_str[14:-1]; split_index = -1; paren_level = 0
            for i, char in enumerate(content):
                if char == '(': paren_level += 1
                elif char == ')': paren_level -= 1
                elif char == ' ' and paren_level == 0: split_index = i; break
            if split_index == -1: raise ValueError(f"Cannot parse ObjectUnionOf: {axiom_str}")
            left_str = content[:split_index].strip(); right_str = content[split_index:].strip()
            fs_left = self.forward(left_str, anon_e_emb, c_dict, r_dict)
            fs_right = self.forward(right_str, anon_e_emb, c_dict, r_dict)
            return self._logical_or(fs_left, fs_right)
        elif axiom_str.startswith('ObjectSomeValuesFrom('):
            parts = axiom_str[21:-1].split(' ', 1); relation_iri = parts[0]; concept_str = parts[1]
            try:
                r_id = torch.tensor(r_dict[relation_iri]).to(self.device)
                r_emb = self.r_embedding(r_id)
                # r_fs = self._get_r_fs(r_emb, all_e_emb)
                r_fs = checkpoint.checkpoint(self._get_r_fs, r_emb, all_e_emb, use_reentrant=False)
                c_fs = self.forward(concept_str, anon_e_emb, c_dict, r_dict)
                return self._logical_exist(r_fs, c_fs)
            except KeyError: return self.nothing_fs.to(all_e_emb.device)
        elif axiom_str.startswith('ObjectAllValuesFrom('):
            parts = axiom_str[20:-1].split(' ', 1); relation_iri = parts[0]; concept_str = parts[1]
            try:
                r_id = torch.tensor(r_dict[relation_iri]).to(self.device)
                r_emb = self.r_embedding(r_id)
                # r_fs = self._get_r_fs(r_emb, all_e_emb)
                r_fs = checkpoint.checkpoint(self._get_r_fs, r_emb, all_e_emb, use_reentrant=False)
                c_fs = self.forward(concept_str, anon_e_emb, c_dict, r_dict)
                return self._logical_forall(r_fs, c_fs)
            except KeyError: return self.nothing_fs.to(all_e_emb.device)
        elif axiom_str.startswith('ObjectComplementOf('):
            concept_str = axiom_str[19:-1]
            fs = self.forward(concept_str, anon_e_emb, c_dict, r_dict)
            return self._logical_not(fs)
        else:
            print(f"Error: Unrecognized axiom structure in forward: {axiom_str}")
            return self.nothing_fs.to(all_e_emb.device)

    def forward_name(self, x, anon_e_emb):
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)
        c_emb_left = self.c_embedding(x[:, 0])
        c_emb_right = self.c_embedding(x[:, 2])
        fs_left = self._get_c_fs_batch(c_emb_left, all_e_emb)
        fs_right = self._get_c_fs_batch(c_emb_right, all_e_emb)
        fs_intersection = self._logical_and(fs_left, self._logical_not(fs_right))
        return self.get_cc_loss(fs_intersection).mean()

    def get_cc_loss(self, fs_intersection):
        if self.max_measure == 'max':
            max_val = fs_intersection.max(dim=-1)[0]
            return -torch.log(1 - max_val + 1e-10)
        elif self.max_measure.startswith('pmean'):
            p = int(self.max_measure[-1]); pmean_val = ((fs_intersection ** p).mean(dim=-1))**(1/p)
            return -torch.log(1 - pmean_val + 1e-10)
        else: raise ValueError
    def forward_abox_ec(self, x, anon_e_emb):
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)
        e_emb = self.e_embedding_base(x[:, :, 0])
        r_emb = self.r_embedding(x[:, 0, 1])
        c_emb = self.c_embedding(x[:, 0, 2])
        c_fs_batch = self._get_c_fs_batch(c_emb, all_e_emb)
        r_fs_batch = self._get_r_fs_batch(r_emb, all_e_emb)
        exists_rc_fs = self._logical_exist(r_fs_batch, c_fs_batch)
        entity_ids = x[:, :, 0]
        batch_indices = torch.arange(x.size(0)).unsqueeze(1).expand_as(entity_ids)
        dofm = exists_rc_fs[batch_indices, entity_ids]
        loss_pos = -torch.log(dofm[:, 0] + 1e-10); loss_neg = -torch.log(1 - dofm[:, 1:] + 1e-10)
        return (loss_pos.mean() + loss_neg.mean()) / 2
    def forward_abox_ec_created(self, x):
        entity_ids = x[:, :, 0]; concept_ids = x[:, 0, 1]
        e_emb = self.e_embedding_base(entity_ids)
        c_emb = self.c_embedding(concept_ids)
        c_emb_expanded = c_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        emb = torch.cat([c_emb_expanded, e_emb], dim=-1)
        if self.cfg.loss_type == 'c':
            dofm = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
            loss_pos = -torch.log(dofm[:, 0] + 1e-10); loss_neg = -torch.log(1 - dofm[:, 1:] + 1e-10)
            return (loss_pos.mean() + loss_neg.mean()) / 2
        elif self.cfg.loss_type == 'r':
            scores = self.fc_0(emb).squeeze(dim=-1)
            pos_scores = scores[:, 0]; neg_scores = scores[:, 1:]
            return -torch.nn.functional.logsigmoid(pos_scores.unsqueeze(-1) - neg_scores).mean()
        else: raise ValueError
    def forward_ggi(self, x, stage='train'):
        e1_emb = self.e_embedding_base(x[:, :, 0]); r_emb = self.r_embedding(x[:, :, 1]); e2_emb = self.e_embedding_base(x[:, :, 2])
        emb = torch.cat([e1_emb + r_emb, e2_emb], dim=-1)
        if stage == 'train':
            if self.cfg.loss_type == 'c':
                dofm = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
                loss_pos = -torch.log(dofm[:, 0] + 1e-10); loss_neg = -torch.log(1 - dofm[:, 1:] + 1e-10)
                return (loss_pos.mean() + loss_neg.mean()) / 2
            elif self.cfg.loss_type == 'r':
                scores = self.fc_0(emb).squeeze(dim=-1)
                pos_scores = scores[:, 0]; neg_scores = scores[:, 1:]
                return -torch.nn.functional.logsigmoid(pos_scores.unsqueeze(-1) - neg_scores).mean()
            else: raise ValueError
        elif stage == 'test':
            return torch.sigmoid(self.fc_0(emb)).flatten()
        else: raise ValueError

# --- Evaluation --- (Adapted like in el.py)
def get_ranks(logits, y, already_ts_dict, already_hs_dict, flag):
    logits_sorted_indices = torch.argsort(logits.cpu(), dim=-1, descending=True)
    if flag == 'head':
        true_entity_id = y[0].item(); rank_raw = (logits_sorted_indices == true_entity_id).nonzero(as_tuple=True)[0].item() + 1
        filter_key = (y[2].item(), y[1].item()); already = set(already_hs_dict.get(filter_key, []))
    elif flag == 'tail':
        true_entity_id = y[2].item(); rank_raw = (logits_sorted_indices == true_entity_id).nonzero(as_tuple=True)[0].item() + 1
        filter_key = (y[0].item(), y[1].item()); already = set(already_ts_dict.get(filter_key, []))
    else: raise ValueError("flag must be 'head' or 'tail'")
    rank_filtered = rank_raw
    for i in range(rank_raw - 1):
        higher_ranked_entity_id = logits_sorted_indices[i].item()
        if higher_ranked_entity_id in already and higher_ranked_entity_id != true_entity_id:
            rank_filtered -= 1
    # Return metrics based on filtered rank
    r = float(rank_filtered)
    rr = 1.0 / r if r > 0 else 0.0
    h1 = float(rank_filtered == 1)
    h3 = float(rank_filtered <= 3)
    h10 = float(rank_filtered <= 10)
    return r, rr, h1, h3, h10

def ggi_evaluate(model, loader, e_dict_len, device, already_ts_dict, already_hs_dict):
    model.eval()
    mr, mrr, mh1, mh3, mh10 = 0.0, 0.0, 0.0, 0.0, 0.0
    num_samples = 0
    with torch.no_grad():
        try: loader_iter = tqdm.tqdm(loader, desc="GGI Evaluation")
        except: loader_iter = loader
        for X, y in loader_iter:
            X, y = X.to(device), y.to(device).squeeze(0)
            if model.__class__.__name__ == 'FALCON': logits = model.forward_ggi(X, stage='test')
            # elif model.__class__.__name__ == 'KGCModel': logits = model.forward(X).flatten()
            else: raise TypeError(f"Unsupported model type: {model.__class__.__name__}")
            logits_head, logits_tail = logits[:e_dict_len], logits[e_dict_len:]

            r_h, rr_h, h1_h, h3_h, h10_h = get_ranks(logits_head, y, already_ts_dict, already_hs_dict, flag='head')
            mr += r_h; mrr += rr_h; mh1 += h1_h; mh3 += h3_h; mh10 += h10_h
            r_t, rr_t, h1_t, h3_t, h10_t = get_ranks(logits_tail, y, already_ts_dict, already_hs_dict, flag='tail')
            mr += r_t; mrr += rr_t; mh1 += h1_t; mh3 += h3_t; mh10 += h10_t
            num_samples += 1

    total_predictions = num_samples * 2
    if total_predictions == 0: return 0.0, 0.0, 0.0
    # Return MRR, H@3, H@10
    return round(mrr / total_predictions, 5), round(mh3 / total_predictions, 3), round(mh10 / total_predictions, 3)

# --- Utils ---
# Removed iterator (moved to utils)

# --- Args ---
def parse_args(args=None):
    parser = argparse.ArgumentParser(description="FALCON model training for HPO")
    parser.add_argument('--data_path', default='../../data/HPO/processed', type=str,
                        help='Path to the directory containing preprocessed data files (tsv/txt)')
    # Tunable
    parser.add_argument('--model', default='FALCON', type=str, help='FALCON')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--num_ng', default=8, type=int)
    # parser.add_argument('--n_models', default=2, type=int) # Unused
    parser.add_argument('--loss_type', default='r', type=str, choices=['r', 'c'])
    parser.add_argument('--bs_ee', default=64, type=int) # GGI batch size
    parser.add_argument('--bs_ec', default=64, type=int)
    parser.add_argument('--bs_ec_created', default=64, type=int) # Added specific BS
    parser.add_argument('--bs_tbox_name', default=64, type=int)
    parser.add_argument('--bs_tbox_desc', default=4, type=int)
    parser.add_argument('--anon_e', default=4, type=int)
    parser.add_argument('--n_e', default=1500, type=int, help='Max number of entities to keep (approx)')
    parser.add_argument('--n_abox_ec_created', default=1000, type=int)
    # parser.add_argument('--n_inconsistent', default=0, type=int) # Unused
    parser.add_argument('--t_norm', default='product', type=str, choices=['product', 'minmax', 'Łukasiewicz'])
    parser.add_argument('--residuum', default='notCorD', type=str)
    parser.add_argument('--max_measure', default='max', type=str)
    # Untunable
    # parser.add_argument('--data_root', default='../../data/HPO/', type=str) # Replaced
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--valid_interval', default=1000, type=int)
    parser.add_argument('--tolerance', default=10, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    return parser.parse_args(args)

# --- Main ---
if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg): print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)

    # --- Data ---
    tbox_name_train, tbox_desc_train, tbox_test_pos, tbox_test_neg, \
        abox_ec, abox_ec_created, abox_ee_train, abox_ee_test, \
        c_dict, e_dict, e_dict_more, r_dict, already_ts_dict, already_hs_dict = get_data(cfg)

    c_dict_len = len(c_dict); e_dict_len = len(e_dict); e_dict_more_len = len(e_dict_more); r_dict_len = len(r_dict)
    print(f'Concepts: {c_dict_len}\tEntities(Base): {e_dict_len}\tEntities(Created): {e_dict_more_len - e_dict_len}\tAnon: {cfg.anon_e}\tRelations: {r_dict_len}', flush=True)
    print(f'TBox Name Train: {len(tbox_name_train)}\tTBox Desc Train: {len(tbox_desc_train)}', flush=True)
    print(f'TBox Test Pos: {len(tbox_test_pos)}\tTBox Test Neg: {len(tbox_test_neg)}', flush=True)
    print(f'ABox EC: {len(abox_ec)}\tABox EC Created: {len(abox_ec_created)}', flush=True)
    print(f'ABox EE Train: {len(abox_ee_train)}\tABox EE Test: {len(abox_ee_test)}', flush=True)

    # --- Datasets & Loaders ---
    ggi_dataset_train = GGIDataset(cfg, abox_ee_train, e_dict_len, already_ts_dict, already_hs_dict, stage='train') if not abox_ee_train.empty else None
    ggi_dataset_test = GGIDataset(cfg, abox_ee_test, e_dict_len, already_ts_dict, already_hs_dict, stage='test') if not abox_ee_test.empty else None
    abox_ec_dataset = AboxECDataset(cfg, abox_ec, e_dict_len) if not abox_ec.empty else None
    abox_ec_created_dataset = AboxECCreatedDataset(cfg, abox_ec_created, e_dict_more_len, c_dict_len) if not abox_ec_created.empty else None
    tbox_name_dataset = NaiveDataset(torch.tensor(tbox_name_train.values)) if not tbox_name_train.empty else None
    tbox_desc_dataset = NaiveDataset(tbox_desc_train) if tbox_desc_train else None

    dataloaders = {}
    # Use utility function for iterators
    if ggi_dataset_train: dataloaders['ggi'] = iterator(torch.utils.data.DataLoader(ggi_dataset_train, batch_size=cfg.bs_ee, shuffle=True, drop_last=True))
    if abox_ec_dataset: dataloaders['abox_ec'] = iterator(torch.utils.data.DataLoader(abox_ec_dataset, batch_size=cfg.bs_ec, shuffle=True, drop_last=True))
    if abox_ec_created_dataset: dataloaders['abox_ec_created'] = iterator(torch.utils.data.DataLoader(abox_ec_created_dataset, batch_size=cfg.bs_ec_created, shuffle=True, drop_last=True))
    if tbox_name_dataset: dataloaders['tbox_name'] = iterator(torch.utils.data.DataLoader(tbox_name_dataset, batch_size=cfg.bs_tbox_name, shuffle=True, drop_last=True))
    if tbox_desc_dataset: dataloaders['tbox_desc'] = iterator(torch.utils.data.DataLoader(tbox_desc_dataset, batch_size=cfg.bs_tbox_desc, shuffle=True, drop_last=True))

    ggi_dataloader_test = torch.utils.data.DataLoader(ggi_dataset_test, batch_size=1, shuffle=False) if ggi_dataset_test else None

    if not dataloaders: print("Error: No training data loaded. Exiting."); sys.exit(1)

    # --- Model & Optimizer ---
    device = torch.device(f'cuda:{cfg.gpu}' if cfg.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    model = FALCON(c_dict_len, e_dict_more_len, r_dict_len, cfg, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # --- Training ---
    weights = {'ggi': 5, 'abox_ec': 1, 'abox_ec_created': 1, 'tbox_name': 1, 'tbox_desc': 1}
    total_weight = sum(weights[k] for k in dataloaders.keys())
    if cfg.verbose: ranger = tqdm.tqdm(range(cfg.max_steps), desc="Training Steps")
    else: ranger = range(cfg.max_steps)
    losses_log = []
    results = [] # Store validation results: [mrr, mh3, mh10, mae_pos, auc, aupr, fmax]
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_metric = -1.0 # Use GGI MRR for early stopping
    steps_since_best = 0

    # Make dicts accessible for forward pass
    global_c_dict = c_dict
    global_r_dict = r_dict

    for step in ranger:
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        loss_components = {}

        anon_e_emb_1 = model.e_embedding_base.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim), device=device)
        anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim, device=device); torch.nn.init.xavier_uniform_(anon_e_emb_2)
        anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            if 'ggi' in dataloaders:
                loss_ggi = model.forward_ggi(next(dataloaders['ggi']).to(device).long()); total_loss += loss_ggi * weights['ggi']; loss_components['ggi'] = loss_ggi.item()
            if 'abox_ec' in dataloaders:
                loss_abox_ec = model.forward_abox_ec(next(dataloaders['abox_ec']).to(device).long(), anon_e_emb); total_loss += loss_abox_ec * weights['abox_ec']; loss_components['ec'] = loss_abox_ec.item()
            if 'abox_ec_created' in dataloaders:
                loss_abox_ec_created = model.forward_abox_ec_created(next(dataloaders['abox_ec_created']).to(device).long()); total_loss += loss_abox_ec_created * weights['abox_ec_created']; loss_components['ec_cr'] = loss_abox_ec_created.item()
            if 'tbox_name' in dataloaders:
                loss_tbox_name = model.forward_name(next(dataloaders['tbox_name']).to(device).long(), anon_e_emb); total_loss += loss_tbox_name * weights['tbox_name']; loss_components['t_name'] = loss_tbox_name.item()
            if 'tbox_desc' in dataloaders:
                loss_tbox_desc_batch = []
                c_dict = global_c_dict; r_dict = global_r_dict # Make dicts available
                for axiom_str in next(dataloaders['tbox_desc']):
                    fs = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                    loss_tbox_desc_batch.append(model.get_cc_loss(fs))
                loss_tbox_desc = sum(loss_tbox_desc_batch) / len(loss_tbox_desc_batch) if loss_tbox_desc_batch else torch.tensor(0.0).to(device)
                total_loss += loss_tbox_desc * weights['tbox_desc']; loss_components['t_desc'] = loss_tbox_desc.item()

            loss = total_loss / total_weight if total_weight > 0 else torch.tensor(0.0).to(device)

        if total_weight > 0:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses_log.append(loss.item())
        else: losses_log.append(0.0)

        if cfg.verbose and (step + 1) % (cfg.valid_interval // 10 + 1) == 0:
             loss_str = ", ".join([f"{k}:{v:.4f}" for k, v in loss_components.items()])
             ranger.set_postfix_str(f"Loss: {loss.item():.4f} ({loss_str})")

        # --- Validation ---
        if (step + 1) % cfg.valid_interval == 0:
            avg_loss = sum(losses_log) / len(losses_log) if losses_log else 0.0
            print(f'\nStep {step+1}/{cfg.max_steps} - Avg Loss: {avg_loss:.4f}', flush=True)
            losses_log = []

            # GGI Evaluation
            mrr, mh3, mh10 = 0.0, 0.0, 0.0
            if ggi_dataloader_test:
                mrr, mh3, mh10 = ggi_evaluate(model, ggi_dataloader_test, e_dict_len, device, already_ts_dict, already_hs_dict)
                print(f'#GGI# MRR: {mrr:.3f}, H@3: {mh3:.3f}, H@10: {mh10:.3f}', flush=True)
            else: print("#GGI# Skipping evaluation (no test data).")

            # TBox Evaluation using utility function
            mae_pos, auc, aupr, fmax = 0.0, 0.0, 0.0, 0.0
            if tbox_test_pos or tbox_test_neg:
                model.eval()
                preds = []
                with torch.no_grad():
                    c_dict = global_c_dict; r_dict = global_r_dict # Make dicts available
                    # Use last anon embeddings for eval
                    anon_e_emb_1 = model.e_embedding_base.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim), device=device)
                    anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim, device=device); torch.nn.init.xavier_uniform_(anon_e_emb_2)
                    anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)

                    for axiom_str in tbox_test_pos:
                        fs = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                        # Prediction is 1 - max(intersection), so higher is better for true axioms
                        preds.append(1.0 - fs.max().item())
                    for axiom_str in tbox_test_neg:
                        fs = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                        preds.append(1.0 - fs.max().item())

                mae_pos, auc, aupr, fmax = compute_metrics(preds) # Use util func
                print(f'#TBox# MAE(pos):{mae_pos:.3f}\tAUC:{auc:.3f}\tAUPR:{aupr:.3f}\tFmax:{fmax:.3f}', flush=True)
            else: print("#TBox# Skipping evaluation (no test data).")

            results.append([mrr, mh3, mh10, mae_pos, auc, aupr, fmax])

            # Early stopping check (using GGI MRR)
            current_metric = mrr
            if current_metric > best_metric:
                best_metric = current_metric; steps_since_best = 0
                print(f"  New best validation MRR: {best_metric:.4f}", flush=True)
            else:
                steps_since_best += 1
                print(f"  Validation MRR did not improve ({current_metric:.4f} vs best {best_metric:.4f}). Tolerance left: {cfg.tolerance - steps_since_best}", flush=True)
            if steps_since_best >= cfg.tolerance:
                print(f"Early stopping triggered after {step + 1} steps.", flush=True); break

    # --- Final Results ---
    print("\nTraining finished.", flush=True)
    if results:
        results_tensor = torch.tensor(results)
        best_epoch_idx = results_tensor[:, 0].argmax() # Best based on MRR (index 0)
        final_results = results_tensor[best_epoch_idx].tolist()
        mrr, mh3, mh10, mae_pos, auc, aupr, fmax = final_results
        print(f'Best Validation Result (at validation step {best_epoch_idx + 1}):', flush=True)
        print(f'  #GGI# MRR: {mrr:.3f}\tH@3: {mh3:.3f}\tH@10: {mh10:.3f}', flush=True)
        print(f'  #TBox# MAE(pos):{mae_pos:.3f}\tAUC:{auc:.3f}\tAUPR:{aupr:.3f}\tFmax:{fmax:.3f}', flush=True)
    else: print("No validation results recorded.", flush=True)

