import torch
import pdb
import argparse
import os
import pandas as pd
import numpy as np
import random
# from sklearn.metrics import roc_auc_score # Moved to utils
# from sklearn.metrics import average_precision_score # Moved to utils
# from sklearn.metrics import precision_recall_curve # Moved to utils
import re # Keep re
from pathlib import Path # Use pathlib
import sys # For exit

# Import shared functions from cfalcon.utils
from cfalcon.utils import (
    reconstruct_functional_syntax, read_list_from_file,
    read_tbox_test_axioms, concept_replacer, tbox_test_neg_generator,
    compute_metrics
)

# Removed reconstruct_functional_syntax
# Removed read_list_from_file
# Removed read_tbox_test_axioms
# Removed concept_replacer
# Removed tbox_test_neg_generator
# Removed compute_metrics


class FALCON(torch.nn.Module):
    # Pass lengths instead of dicts
    def __init__(self, c_dict_len, e_dict_len, r_dict_len, cfg):
        super().__init__()
        self.c_dict_len = c_dict_len
        self.e_dict_len = e_dict_len # Base + created
        self.r_dict_len = r_dict_len
        self.n_entity_base = e_dict_len
        self.anon_e = cfg.anon_e
        self.n_entity_total = self.n_entity_base + self.anon_e
        self.cfg = cfg

        self.c_embedding = torch.nn.Embedding(self.c_dict_len, cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(self.r_dict_len, cfg.emb_dim)
        self.e_embedding_base = torch.nn.Embedding(self.n_entity_base, cfg.emb_dim) # Base + created
        self.fc_0 = torch.nn.Linear(cfg.emb_dim * 2, cfg.emb_dim)
        self.fc_1 = torch.nn.Linear(cfg.emb_dim, 1)

        torch.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.e_embedding_base.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_0.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)

        self.max_measure = cfg.max_measure
        self.t_norm = cfg.t_norm
        self.nothing_fs = torch.zeros(self.n_entity_total) # Fuzzy set for owl:Nothing
        self.residuum = cfg.residuum
        # self.device will be set by .to(device)

    def to(self, device):
        """Move model and internal tensors to device."""
        super().to(device)
        self.nothing_fs = self.nothing_fs.to(device)
        self.device = device # Store device
        return self

    # --- Fuzzy Logic Operators --- (No changes needed)
    def _logical_and(self, x, y):
        if self.t_norm == 'product': return x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2); y = y.expand_as(x); return torch.cat([x, y], dim=-2).min(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2); y = y.expand_as(x); return (((x + y -1) > 0) * (x + y - 1)).squeeze(dim=-2)
        else: raise ValueError
    def _logical_or(self, x, y):
        if self.t_norm == 'product': return x + y - x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2); y = y.expand_as(x); return torch.cat([x, y], dim=-2).max(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2); y = y.expand_as(x); return 1 - ((((1-x) + (1-y) -1) > 0) * ((1-x) + (1-y) - 1)).squeeze(dim=-2)
        else: raise ValueError
    def _logical_not(self, x): return 1 - x
    def _logical_residuum(self, r_fs, c_fs):
        if self.residuum == 'notCorD':
            c_fs_expanded = c_fs.unsqueeze(-2); return self._logical_or(self._logical_not(r_fs), c_fs_expanded)
        else: raise ValueError
    def _logical_exist(self, r_fs, c_fs):
        c_fs_expanded = c_fs.unsqueeze(-2); return self._logical_and(r_fs, c_fs_expanded).max(dim=-1)[0]
    def _logical_forall(self, r_fs, c_fs):
        return self._logical_residuum(r_fs, c_fs).min(dim=-1)[0]

    # --- Fuzzy Set Calculation ---
    def _get_all_entity_embeddings(self, anon_e_emb):
        if anon_e_emb.shape[0] != self.anon_e or (self.anon_e > 0 and anon_e_emb.shape[1] != self.cfg.emb_dim):
             # Allow empty tensor if anon_e is 0
             if self.anon_e > 0:
                 raise ValueError(f"Incorrect shape for anon_e_emb: {anon_e_emb.shape}")
        if not hasattr(self, 'device'): raise RuntimeError("Model device not set.")
        if anon_e_emb.device != self.device: anon_e_emb = anon_e_emb.to(self.device)
        return torch.cat([self.e_embedding_base.weight, anon_e_emb], dim=0)
    def _get_c_fs(self, c_emb, all_e_emb):
        c_emb_expanded = c_emb.expand_as(all_e_emb)
        emb = torch.cat([c_emb_expanded, all_e_emb], dim=-1)
        return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
    def _get_r_fs(self, r_emb, all_e_emb):
        num_entities = all_e_emb.size(0)
        e_emb_repeated_rows = all_e_emb.repeat_interleave(num_entities, dim=0)
        e_emb_repeated_cols = all_e_emb.repeat(num_entities, 1)
        l_emb_flat = e_emb_repeated_rows + r_emb.unsqueeze(0)
        r_emb_flat = e_emb_repeated_cols
        emb_flat = torch.cat([l_emb_flat, r_emb_flat], dim=-1)
        fs_flat = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb_flat), negative_slope=0.1))).squeeze(dim=-1)
        return fs_flat.view(num_entities, num_entities)

    # --- Forward Pass for Axioms ---
    # Needs global dicts or passed dicts
    def forward(self, axiom_str, anon_e_emb, c_dict, r_dict):
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)
        if axiom_str[0] == '<' or axiom_str == 'owl:Thing':
            try:
                c_id = torch.tensor(c_dict[axiom_str]).to(self.device)
                c_emb = self.c_embedding(c_id)
                return self._get_c_fs(c_emb, all_e_emb)
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
                r_fs = self._get_r_fs(r_emb, all_e_emb)
                c_fs = self.forward(concept_str, anon_e_emb, c_dict, r_dict)
                return self._logical_exist(r_fs, c_fs)
            except KeyError: return self.nothing_fs.to(all_e_emb.device)
        elif axiom_str.startswith('ObjectAllValuesFrom('):
            parts = axiom_str[20:-1].split(' ', 1); relation_iri = parts[0]; concept_str = parts[1]
            try:
                r_id = torch.tensor(r_dict[relation_iri]).to(self.device)
                r_emb = self.r_embedding(r_id)
                r_fs = self._get_r_fs(r_emb, all_e_emb)
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

    # --- Loss Functions ---
    def get_cc_loss(self, fs_intersection):
        if self.max_measure == 'max':
            max_val = fs_intersection.max(dim=-1)[0]; return -torch.log(1 - max_val + 1e-10)
        elif self.max_measure.startswith('pmean'):
            p = int(self.max_measure[-1]); pmean_val = ((fs_intersection ** p).mean(dim=-1))**(1/p)
            return -torch.log(1 - pmean_val + 1e-10)
        else: raise ValueError
    def get_ec_loss(self, e_id, c_id):
        e_emb = self.e_embedding_base(torch.tensor(e_id).to(self.device))
        c_emb = self.c_embedding(torch.tensor(c_id).to(self.device))
        emb = torch.cat([c_emb, e_emb])
        score = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))
        dofm = torch.sigmoid(score).squeeze(); return -torch.log(dofm + 1e-10)
    def get_ee_loss(self, e1_id, r_id, e2_id, sample_neg=True):
        e1_emb = self.e_embedding_base(torch.tensor(e1_id).to(self.device))
        r_emb = self.r_embedding(torch.tensor(r_id).to(self.device))
        e2_emb = self.e_embedding_base(torch.tensor(e2_id).to(self.device))
        emb_pos = torch.cat([e1_emb + r_emb, e2_emb])
        score_pos = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb_pos), negative_slope=0.1))
        dofm_pos = torch.sigmoid(score_pos).squeeze(); loss = -torch.log(dofm_pos + 1e-10)
        if sample_neg:
            emb_neg = torch.cat([e2_emb + r_emb, e1_emb])
            score_neg = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb_neg), negative_slope=0.1))
            dofm_neg = torch.sigmoid(score_neg).squeeze(); loss += -torch.log(1 - dofm_neg + 1e-10)
        return loss

# --- Data Loading ---
def get_data(cfg):
    """Loads data from preprocessed files specified by cfg.data_path."""
    data_path = Path(cfg.data_path)
    tbox_test_pos_file = data_path / "tbox_test_pos.txt" # Assumed pre-generated
    tbox_test_neg_file = data_path / "tbox_test_neg.txt" # Assumed pre-generated or generated here

    print(f"Loading data from: {data_path.resolve()}")

    # Load concepts, relations, entities using utility function
    all_concepts_list = read_list_from_file(data_path / "concepts.txt")
    all_relations_list = read_list_from_file(data_path / "relations.txt")
    all_entities_list = read_list_from_file(data_path / "entities.txt")

    # Add special concepts/relations if needed
    if 'owl:Thing' not in all_concepts_list: all_concepts_list.append('owl:Thing')
    if 'owl:Nothing' not in all_concepts_list: all_concepts_list.append('owl:Nothing')

    # Load ABox data
    try:
        abox_ec_df = pd.read_csv(data_path / "abox_ec.tsv", sep='\t', header=None, names=['h', 't'], keep_default_na=False)
        # Convert to original string format "entity concept"
        abox_ec = abox_ec_df.apply(lambda row: f"{row['h']} {row['t']}", axis=1).tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: {data_path / 'abox_ec.tsv'} not found or empty. ABox EC will be empty.")
        abox_ec = []

    try:
        abox_ee_df = pd.read_csv(data_path / "abox_ee.tsv", sep='\t', header=None, names=['h', 'r', 't'], keep_default_na=False)
        # Convert to original string format "e1 r e2"
        abox_ee = abox_ee_df.apply(lambda row: f"{row['h']} {row['r']} {row['t']}", axis=1).tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: {data_path / 'abox_ee.tsv'} not found or empty. ABox EE will be empty.")
        abox_ee = []

    # Load and reconstruct TBox data for training using utility function
    tbox_train = []
    try:
        with open(data_path / "tbox.tsv", 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if not parts: continue
                axiom_type = parts[0]; axiom_parts = parts[1:]
                reconstructed = reconstruct_functional_syntax(axiom_type, axiom_parts) # Use util func
                if reconstructed:
                    if isinstance(reconstructed, list): tbox_train.extend(reconstructed)
                    else: tbox_train.append(reconstructed)
    except FileNotFoundError:
        print(f"Warning: {data_path / 'tbox.tsv'} not found. Training TBox will be empty.")

    # Add inconsistent axioms if specified (mimicking original logic)
    # This part is complex as the original axioms were hardcoded.
    # We'll skip adding inconsistent axioms for now, assuming they'd be in tbox.tsv if needed.
    if cfg.n_inconsistent > 0:
         print(f"Warning: n_inconsistent > 0 is set, but adding inconsistent axioms from code is disabled. "
               f"Ensure they are present in {data_path / 'tbox.tsv'} if required.")

    # Load TBox test axioms using utility function
    tbox_test_pos = read_tbox_test_axioms(tbox_test_pos_file)
    if not tbox_test_pos: print(f"Warning: Positive test TBox file '{tbox_test_pos_file}' not found or empty.")

    try:
        tbox_test_neg = read_tbox_test_axioms(tbox_test_neg_file)
        if not tbox_test_neg: raise FileNotFoundError # Trigger generation
    except FileNotFoundError:
        if tbox_train or tbox_test_pos:
            print("Generating negative TBox test axioms...")
            num_neg_to_generate = len(tbox_test_pos) if tbox_test_pos else 500 # Default count
            # Use utility function for generation
            tbox_test_neg = tbox_test_neg_generator(tbox_train, tbox_test_pos, all_concepts_list, k=num_neg_to_generate)
            try: # Save generated negatives
                with open(tbox_test_neg_file, 'w', encoding='utf-8') as f:
                    for axiom in tbox_test_neg: f.write(f"{axiom}\n")
                print(f"Saved generated negative axioms to {tbox_test_neg_file}")
            except IOError as e: print(f"Error saving generated negative axioms: {e}")
        else:
             print("Warning: Cannot generate negative test axioms as no positive axioms were loaded.")
             tbox_test_neg = []

    # Create dictionaries
    all_concepts_list = sorted(list(set(all_concepts_list)))
    all_relations_list = sorted(list(set(all_relations_list)))
    all_entities_list = sorted(list(set(all_entities_list)))
    c_dict = {k: v for v, k in enumerate(all_concepts_list)}
    r_dict = {k: v for v, k in enumerate(all_relations_list)}
    e_dict = {k: v for v, k in enumerate(all_entities_list)} # Base + created (if any)

    print(f"Loaded: {len(c_dict)} concepts, {len(r_dict)} relations, {len(e_dict)} entities.")
    print(f"Loaded: {len(tbox_train)} Train TBox, {len(abox_ec)} ABox EC, {len(abox_ee)} ABox EE.")
    print(f"Loaded: {len(tbox_test_pos)} Test TBox Pos, {len(tbox_test_neg)} Test TBox Neg.")

    return tbox_train, tbox_test_pos, tbox_test_neg, abox_ec, abox_ee, c_dict, e_dict, r_dict

# --- Metrics ---
# Removed compute_metrics (moved to utils)

# --- Args ---
def parse_args(args=None):
    parser = argparse.ArgumentParser(description="FALCON model training for Pizza")
    parser.add_argument('--data_path', default='../../data/Pizza/processed', type=str,
                        help='Path to the directory containing preprocessed data files (tsv/txt)')
    # Tunable
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--emb_dim', default=50, type=int)
    parser.add_argument('--n_models', default=1, type=int) # Default to 1 for simpler run
    parser.add_argument('--bs', default=256, type=int, help='Batch size for sampling axioms')
    parser.add_argument('--anon_e', default=0, type=int) # Pizza usually doesn't use anon entities
    parser.add_argument('--n_inconsistent', default=0, type=int, help='(Currently ignored) Number of inconsistent axioms to add')
    parser.add_argument('--t_norm', default='product', type=str, choices=['product', 'minmax', 'Łukasiewicz'])
    parser.add_argument('--residuum', default='notCorD', type=str)
    parser.add_argument('--max_measure', default='max', type=str)
    # Untunable
    # parser.add_argument('--data_root', default='../../data/Pizza/', type=str) # Replaced
    parser.add_argument('--max_steps', default=10000, type=int)
    parser.add_argument('--valid_interval', default=100, type=int) # Increased interval
    parser.add_argument('--tolerance', default=10, type=int) # Increased tolerance
    parser.add_argument('--save_dir', default='../tmp/pizza_run', type=str, help='Directory to save temporary model checkpoints')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use (-1 for CPU)')
    return parser.parse_args(args)

# --- Main ---
if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:')
    for arg in vars(cfg): print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)

    # --- Data ---
    tbox_train, tbox_test_pos, tbox_test_neg, abox_ec, abox_ee, c_dict, e_dict, r_dict = get_data(cfg)

    # --- Setup ---
    device = torch.device(f'cuda:{cfg.gpu}' if cfg.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    save_root = Path(cfg.save_dir)
    if not save_root.exists(): save_root.mkdir(parents=True)
    print(f"Saving checkpoints to: {save_root.resolve()}", flush=True)

    # --- Multi-model Training & Evaluation ---
    all_model_logits = [] # Store final logits from each model for max aggregation
    all_model_metrics = [] # Store metrics for each model for averaging

    for i in range(cfg.n_models):
        print(f'\n--- Training Model {i+1}/{cfg.n_models} ---', flush=True)
        model = FALCON(c_dict_len=len(c_dict),
                       e_dict_len=len(e_dict),
                       r_dict_len=len(r_dict),
                       cfg=cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

        best_valid_metric = -1.0 # Track best validation AUC for this model
        steps_since_best = 0
        best_step = -1

        # Make dicts accessible
        global_c_dict = c_dict
        global_r_dict = r_dict

        # Training loop for one model
        for step in range(cfg.max_steps):
            model.train()
            optimizer.zero_grad()

            # Generate anonymous embeddings (even if anon_e is 0, create empty tensor)
            if cfg.anon_e > 0:
                 anon_e_emb_1 = model.e_embedding_base.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim), device=device)
                 anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim, device=device); torch.nn.init.xavier_uniform_(anon_e_emb_2)
                 anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)
            else:
                 anon_e_emb = torch.empty((0, cfg.emb_dim), device=device)


            # Sample batches and calculate loss
            loss_cc_list, loss_ec_list, loss_ee_list = [], [], []
            # TBox Loss (sample from tbox_train)
            if tbox_train:
                batch_tbox_train = random.sample(tbox_train, min(cfg.bs, len(tbox_train)))
                c_dict = global_c_dict; r_dict = global_r_dict # Make dicts available
                for axiom_str in batch_tbox_train:
                    fs = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                    loss_cc_list.append(model.get_cc_loss(fs))
            # ABox EC Loss (sample from abox_ec)
            if abox_ec:
                batch_abox_ec = random.sample(abox_ec, min(cfg.bs // 4 if cfg.bs // 4 > 0 else 1, len(abox_ec)))
                for axiom_str in batch_abox_ec:
                    try:
                        e_iri, c_iri = axiom_str.split(' ', 1)
                        loss_ec_list.append(model.get_ec_loss(e_dict[e_iri], c_dict[c_iri]))
                    except (ValueError, KeyError): pass # Skip if parsing/lookup fails
            # ABox EE Loss (sample from abox_ee) - Currently weighted 0
            # if abox_ee:
            #     batch_abox_ee = random.sample(abox_ee, min(cfg.bs // 4 if cfg.bs // 4 > 0 else 1, len(abox_ee)))
            #     for axiom_str in batch_abox_ee:
            #         try:
            #             e1_iri, r_iri, e2_iri = axiom_str.split(' ', 2)
            #             loss_ee_list.append(model.get_ee_loss(e_dict[e1_iri], r_dict[r_iri], e_dict[e2_iri]))
            #         except (ValueError, KeyError): pass

            loss_cc = sum(loss_cc_list) / len(loss_cc_list) if loss_cc_list else torch.tensor(0.0).to(device)
            loss_ec = sum(loss_ec_list) / len(loss_ec_list) if loss_ec_list else torch.tensor(0.0).to(device)
            loss_ee = sum(loss_ee_list) / len(loss_ee_list) if loss_ee_list else torch.tensor(0.0).to(device)
            loss = 0.5 * loss_cc + 0.5 * loss_ec # + 0.0 * loss_ee

            if loss.requires_grad: loss.backward(); optimizer.step()

            # --- Validation ---
            if (step + 1) % cfg.valid_interval == 0:
                model.eval()
                preds = []
                with torch.no_grad():
                    # Use last anon embeddings for eval
                    if cfg.anon_e > 0:
                         anon_e_emb_1 = model.e_embedding_base.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim), device=device)
                         anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim, device=device); torch.nn.init.xavier_uniform_(anon_e_emb_2)
                         anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)
                    else: anon_e_emb = torch.empty((0, cfg.emb_dim), device=device)

                    c_dict = global_c_dict; r_dict = global_r_dict # Make dicts available
                    for axiom_str in tbox_test_pos:
                        fs = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                        preds.append(1.0 - fs.max().item()) # Score = 1 - max(violation)
                    for axiom_str in tbox_test_neg:
                        fs = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                        preds.append(1.0 - fs.max().item())

                mae_pos, auc, aupr, fmax = compute_metrics(preds) # Use util func
                print(f' Step {step+1} - Loss: {loss.item():.4f} | Valid MAE(pos):{mae_pos:.3f} AUC:{auc:.3f} AUPR:{aupr:.3f} Fmax:{fmax:.3f}', flush=True)

                # Early stopping check (using AUC)
                current_metric = auc
                if current_metric >= best_valid_metric: # Use >= to save latest best
                    best_valid_metric = current_metric
                    steps_since_best = 0
                    best_step = step + 1
                    # Save checkpoint of the best model so far for this run
                    torch.save(model.state_dict(), save_root / f"model_{i}_best.pt")
                    print(f"  Saved new best model checkpoint (AUC: {best_valid_metric:.4f})", flush=True)
                else:
                    steps_since_best += 1

                if steps_since_best >= cfg.tolerance:
                    print(f"Early stopping triggered for model {i+1} at step {step + 1}.", flush=True)
                    break

        # --- Post-Training Evaluation for this model ---
        print(f"Loading best checkpoint for model {i+1} from step {best_step}")
        if best_step != -1:
            try:
                 model.load_state_dict(torch.load(save_root / f"model_{i}_best.pt", map_location=device))
            except FileNotFoundError:
                 print("Warning: Best checkpoint not found, using model from last step.")

        model.eval()
        final_logits_this_model = []
        with torch.no_grad():
            # Use last anon embeddings for final eval
            if cfg.anon_e > 0:
                 anon_e_emb_1 = model.e_embedding_base.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim), device=device)
                 anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim, device=device); torch.nn.init.xavier_uniform_(anon_e_emb_2)
                 anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)
            else: anon_e_emb = torch.empty((0, cfg.emb_dim), device=device)

            c_dict = global_c_dict; r_dict = global_r_dict # Make dicts available
            for axiom_str in tbox_test_pos:
                fs = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                final_logits_this_model.append(1.0 - fs.max().item())
            for axiom_str in tbox_test_neg:
                fs = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                final_logits_this_model.append(1.0 - fs.max().item())

        all_model_logits.append(final_logits_this_model)
        mae_pos, auc, aupr, fmax = compute_metrics(final_logits_this_model) # Use util func
        all_model_metrics.append([mae_pos, auc, aupr, fmax])
        print(f"Model {i+1} Final Metrics: MAE(pos):{mae_pos:.3f} AUC:{auc:.3f} AUPR:{aupr:.3f} Fmax:{fmax:.3f}", flush=True)

        # --- Aggregated Results (after each model finishes) ---
        if all_model_metrics:
             avg_metrics = np.mean(all_model_metrics, axis=0)
             print(f"AVG Metrics ({i+1} models): MAE(pos):{avg_metrics[0]:.3f} AUC:{avg_metrics[1]:.3f} AUPR:{avg_metrics[2]:.3f} Fmax:{avg_metrics[3]:.3f}", flush=True)

        if all_model_logits:
             # Max aggregation
             max_agg_preds = torch.tensor(all_model_logits).max(dim=0)[0].numpy().tolist()
             mae_pos_max, auc_max, aupr_max, fmax_max = compute_metrics(max_agg_preds) # Use util func
             print(f"MAX Aggregation ({i+1} models): MAE(pos):{mae_pos_max:.3f} AUC:{auc_max:.3f} AUPR:{aupr_max:.3f} Fmax:{fmax_max:.3f}", flush=True)

    print("\n--- All models finished ---")
    # Final summary is printed after the last model loop iteration.

