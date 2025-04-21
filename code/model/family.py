import torch
import pdb
import argparse
import re
import pandas as pd
import numpy as np
import random
from pathlib import Path # Use pathlib

# Removed get_all_concepts_and_relations - Not needed

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
            return f"ObjectIntersectionOf(ObjectIntersectionOf({parts[0]} {parts[1]}) ObjectComplementOf(<Nothing>))" # Use <Nothing> as in original family.py
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
        # Ignore Domain, Range, SubProperty for TBox loss
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

class FALCON(torch.nn.Module):
    # Pass lengths instead of dicts
    def __init__(self, emb_dim, c_dict_len, e_dict_len, r_dict_len, cfg):
        super().__init__()
        # Store lengths
        self.c_dict_len = c_dict_len
        self.e_dict_len = e_dict_len # Base + created
        self.r_dict_len = r_dict_len

        self.n_entity_base = e_dict_len # Number of entities from file + created
        self.anon_e = cfg.anon_e
        self.n_entity_total = self.n_entity_base + self.anon_e # Total entities including anonymous
        self.cfg = cfg # Store cfg

        # Embeddings sized by dictionary lengths
        self.c_embedding = torch.nn.Embedding(self.c_dict_len, emb_dim)
        self.r_embedding = torch.nn.Embedding(self.r_dict_len, emb_dim)
        # Combined embedding for base + created entities
        self.e_embedding_base = torch.nn.Embedding(self.n_entity_base, emb_dim)

        # Linear layers
        self.fc_0 = torch.nn.Linear(emb_dim * 2, emb_dim)
        self.fc_1 = torch.nn.Linear(emb_dim, 1)

        # Initialization
        torch.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.e_embedding_base.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_0.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)

        # Fuzzy logic parameters
        self.max_measure = cfg.max_measure
        self.t_norm = cfg.t_norm
        # Note: Device needs to be set, assuming default or passed later
        self.nothing_fs = torch.zeros(self.n_entity_total) # Fuzzy set for <Nothing>

        # Store IRI->ID maps if needed for parsing inside forward
        # Avoided by reconstructing strings before calling forward
        # self.c_dict = c_dict
        # self.e_dict = e_dict
        # self.r_dict = r_dict

    def to(self, device):
        """Move model and internal tensors to device."""
        super().to(device)
        self.nothing_fs = self.nothing_fs.to(device)
        self.device = device # Store device
        return self

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

    def _logical_exist(self, r_fs, c_fs):
         # r_fs shape: [N, N], c_fs shape: [N]
         c_fs_expanded = c_fs.unsqueeze(-2) # [1, N]
         # Result shape: [N] (max over second dim)
         return self._logical_and(r_fs, c_fs_expanded).max(dim=-1)[0]

    # --- Fuzzy Set Calculation ---
    def _get_all_entity_embeddings(self, anon_e_emb):
        """Combines base entity embeddings with current anonymous embeddings."""
        if anon_e_emb.shape[0] != self.anon_e or anon_e_emb.shape[1] != self.cfg.emb_dim:
             raise ValueError(f"Incorrect shape for anon_e_emb: {anon_e_emb.shape}")
        if not hasattr(self, 'device'): # Ensure device is set
             raise RuntimeError("Model device not set. Call model.to(device) first.")
        if anon_e_emb.device != self.device:
             anon_e_emb = anon_e_emb.to(self.device)
        return torch.cat([self.e_embedding_base.weight, anon_e_emb], dim=0)

    def _get_c_fs(self, c_emb, all_e_emb):
        """Calculates fuzzy set for a concept C over all entities."""
        c_emb_expanded = c_emb.expand_as(all_e_emb)
        emb = torch.cat([c_emb_expanded, all_e_emb], dim=-1)
        return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)

    def _get_r_fs(self, r_emb, all_e_emb):
        """Calculates fuzzy set for a relation R over all entity pairs (x, y)."""
        num_entities = all_e_emb.size(0)
        # Use flat representation for memory efficiency
        e_emb_repeated_rows = all_e_emb.repeat_interleave(num_entities, dim=0)
        e_emb_repeated_cols = all_e_emb.repeat(num_entities, 1)
        l_emb_flat = e_emb_repeated_rows + r_emb.unsqueeze(0)
        r_emb_flat = e_emb_repeated_cols
        emb_flat = torch.cat([l_emb_flat, r_emb_flat], dim=-1)
        fs_flat = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb_flat), negative_slope=0.1))).squeeze(dim=-1)
        return fs_flat.view(num_entities, num_entities)

    # --- Forward Pass for Axioms ---
    # Relies on reconstructed functional syntax string and global dicts
    def forward(self, axiom_str, anon_e_emb, c_dict, r_dict): # Pass dicts explicitly
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)

        # --- Base Cases ---
        if axiom_str[0] == '<':
            if axiom_str == '<Nothing>': # Handle specific Nothing IRI used here
                return self.nothing_fs.to(all_e_emb.device) # Ensure device match
            else:
                try:
                    c_id = torch.tensor(c_dict[axiom_str]).to(self.device)
                    c_emb = self.c_embedding(c_id)
                    return self._get_c_fs(c_emb, all_e_emb)
                except KeyError:
                    print(f"Error: Concept IRI '{axiom_str}' not found in c_dict during forward.")
                    return self.nothing_fs.to(all_e_emb.device)

        # --- Recursive Steps ---
        elif axiom_str.startswith('ObjectIntersectionOf('):
            content = axiom_str[21:-1]
            split_index = -1
            paren_level = 0
            for i, char in enumerate(content):
                if char == '(': paren_level += 1
                elif char == ')': paren_level -= 1
                elif char == ' ' and paren_level == 0:
                    split_index = i
                    break
            if split_index == -1: # Handle case like ObjectIntersectionOf(<A> <B>)
                 parts = content.split(' ', 1)
                 if len(parts) == 2:
                     left_str, right_str = parts[0], parts[1]
                 else: # Handle case like ObjectIntersectionOf(ObjectIntersectionOf(<A> <B>) <C>) - needs better parsing
                     raise ValueError(f"Cannot parse ObjectIntersectionOf: {axiom_str}")
            else:
                 left_str = content[:split_index].strip()
                 right_str = content[split_index:].strip()

            fs_left = self.forward(left_str, anon_e_emb, c_dict, r_dict)
            fs_right = self.forward(right_str, anon_e_emb, c_dict, r_dict)
            return self._logical_and(fs_left, fs_right)

        elif axiom_str.startswith('ObjectUnionOf('):
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
            fs_left = self.forward(left_str, anon_e_emb, c_dict, r_dict)
            fs_right = self.forward(right_str, anon_e_emb, c_dict, r_dict)
            return self._logical_or(fs_left, fs_right)

        elif axiom_str.startswith('ObjectSomeValuesFrom('):
            parts = axiom_str[21:-1].split(' ', 1)
            relation_iri = parts[0]
            concept_str = parts[1]
            try:
                r_id = torch.tensor(r_dict[relation_iri]).to(self.device)
                r_emb = self.r_embedding(r_id)
                r_fs = self._get_r_fs(r_emb, all_e_emb)
                c_fs = self.forward(concept_str, anon_e_emb, c_dict, r_dict)
                return self._logical_exist(r_fs, c_fs)
            except KeyError:
                 print(f"Error: Relation IRI '{relation_iri}' not found in r_dict during forward.")
                 return self.nothing_fs.to(all_e_emb.device)


        elif axiom_str.startswith('ObjectComplementOf('):
            concept_str = axiom_str[19:-1]
            fs = self.forward(concept_str, anon_e_emb, c_dict, r_dict)
            return self._logical_not(fs)

        else:
            print(f"Error: Unrecognized axiom structure in forward: {axiom_str}")
            return self.nothing_fs.to(all_e_emb.device)

    # --- Loss Functions ---
    def get_cc_loss(self, fs_intersection):
        """ Calculates the loss for concept containment. """
        if self.max_measure == 'max':
            max_val = fs_intersection.max(dim=-1)[0]
            return -torch.log(1 - max_val + 1e-10)
        elif self.max_measure.startswith('pmean'):
            try:
                p = int(self.max_measure[-1])
            except ValueError:
                raise ValueError(f"Invalid pmean value: {self.max_measure}")
            pmean_val = ((fs_intersection ** p).mean(dim=-1))**(1/p)
            return -torch.log(1 - pmean_val + 1e-10)
        else:
            raise ValueError(f"Unknown max_measure: {self.max_measure}")

    def get_ec_loss(self, e_id, c_id):
        """ Calculates loss for ABox EC axiom C(a). """
        # Assumes e_id and c_id are integer indices
        e_emb = self.e_embedding_base(torch.tensor(e_id).to(self.device))
        c_emb = self.c_embedding(torch.tensor(c_id).to(self.device))
        emb = torch.cat([c_emb, e_emb]) # Shape [2*dim]
        # Apply scoring layers
        score = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))
        dofm = torch.sigmoid(score).squeeze() # Get single membership degree
        return -torch.log(dofm + 1e-10)

    def get_ee_loss(self, e1_id, r_id, e2_id, sample_neg=True):
        """ Calculates loss for ABox EE axiom R(a, b). """
        e1_emb = self.e_embedding_base(torch.tensor(e1_id).to(self.device))
        r_emb = self.r_embedding(torch.tensor(r_id).to(self.device))
        e2_emb = self.e_embedding_base(torch.tensor(e2_id).to(self.device))

        # Positive sample score: score(h+r, t)
        emb_pos = torch.cat([e1_emb + r_emb, e2_emb])
        score_pos = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb_pos), negative_slope=0.1))
        dofm_pos = torch.sigmoid(score_pos).squeeze()
        loss = -torch.log(dofm_pos + 1e-10)

        # Negative sample score (corruption): score(t+r, h) - assuming relation is not symmetric
        if sample_neg:
            emb_neg = torch.cat([e2_emb + r_emb, e1_emb])
            score_neg = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb_neg), negative_slope=0.1))
            dofm_neg = torch.sigmoid(score_neg).squeeze()
            loss += -torch.log(1 - dofm_neg + 1e-10) # Minimize membership of corrupted triple

        return loss

    # --- Prediction / Visualization ---
    def predict(self, anon_e_emb, c_dict, e_dict): # Pass dicts
        """ Generates a DataFrame of concept memberships for all entities. """
        cols = ['concept']
        lines = []
        all_e_emb = self._get_all_entity_embeddings(anon_e_emb)
        e_dict_rev = {v: k for k, v in e_dict.items()} # Need reverse map ID->IRI

        with torch.no_grad():
            for concept_iri, concept_id in c_dict.items():
                if concept_iri == '<Nothing>': continue # Skip Nothing concept
                c_emb = self.c_embedding(torch.tensor(concept_id).to(self.device))
                c_fs = self._get_c_fs(c_emb, all_e_emb) # Fuzzy set over all entities [N_total]
                line = [concept_iri]
                # Get memberships for base entities + anonymous entities
                memberships = [round(m.item(), 3) for m in c_fs] # Use 3 decimal places
                lines.append(line + memberships)

        # Create column headers: 'concept', entity_iri_1, ..., entity_iri_N, anon_0, ..., anon_M
        entity_iris = [e_dict_rev[i] for i in range(self.n_entity_base)] # Base entity IRIs
        anon_names = [f'anon_{i}' for i in range(self.anon_e)]
        cols.extend(entity_iris)
        cols.extend(anon_names)

        return pd.DataFrame(lines, columns=cols)


def get_data(cfg):
    """Loads data from preprocessed files specified by cfg.data_path."""
    data_path = Path(cfg.data_path)
    print(f"Loading data from: {data_path.resolve()}")

    # Load concepts, relations, entities
    all_concepts_list = read_list_from_file(data_path / "concepts.txt")
    all_relations_list = read_list_from_file(data_path / "relations.txt")
    all_entities_list = read_list_from_file(data_path / "entities.txt")

    # Add special concept used in this ontology's axioms
    if '<Nothing>' not in all_concepts_list: all_concepts_list.append('<Nothing>')

    # Load ABox data
    try:
        abox_ec_df = pd.read_csv(data_path / "abox_ec.tsv", sep='\t', header=None, names=['h', 't'], keep_default_na=False)
        abox_ec = abox_ec_df.apply(lambda row: f"{row['h']} {row['t']}", axis=1).tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: {data_path / 'abox_ec.tsv'} not found or empty. ABox EC will be empty.")
        abox_ec = []

    try:
        abox_ee_df = pd.read_csv(data_path / "abox_ee.tsv", sep='\t', header=None, names=['h', 'r', 't'], keep_default_na=False)
        abox_ee = abox_ee_df.apply(lambda row: f"{row['h']} {row['r']} {row['t']}", axis=1).tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: {data_path / 'abox_ee.tsv'} not found or empty. ABox EE will be empty.")
        abox_ee = []

    # Load and reconstruct TBox data
    tbox = []
    try:
        with open(data_path / "tbox.tsv", 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if not parts: continue
                axiom_type = parts[0]
                axiom_parts = parts[1:]
                reconstructed = reconstruct_functional_syntax(axiom_type, axiom_parts)
                if reconstructed:
                    if isinstance(reconstructed, list):
                        tbox.extend(reconstructed)
                    else:
                        tbox.append(reconstructed)
    except FileNotFoundError:
        print(f"Warning: {data_path / 'tbox.tsv'} not found. TBox will be empty.")

    # Ensure all concepts/relations/entities from axioms are in lists
    # (Simplified check - assumes preprocessor output is comprehensive)

    # Create dictionaries (ensure stable order)
    all_concepts_list = sorted(list(set(all_concepts_list)))
    all_relations_list = sorted(list(set(all_relations_list)))
    all_entities_list = sorted(list(set(all_entities_list)))

    c_dict = {k: v for v, k in enumerate(all_concepts_list)}
    r_dict = {k: v for v, k in enumerate(all_relations_list)}
    e_dict = {k: v for v, k in enumerate(all_entities_list)} # Includes base + created

    print(f"Loaded: {len(c_dict)} concepts, {len(r_dict)} relations, {len(e_dict)} entities.")
    print(f"Loaded: {len(tbox)} TBox axioms, {len(abox_ec)} ABox EC axioms, {len(abox_ee)} ABox EE axioms.")

    return tbox, abox_ec, abox_ee, c_dict, e_dict, r_dict


def get_one_model(cfg, tbox, abox_ec, abox_ee, c_dict, e_dict, r_dict):
    """Trains one FALCON model instance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Pass lengths to model constructor
    model = FALCON(emb_dim=cfg.emb_dim,
                   c_dict_len=len(c_dict),
                   e_dict_len=len(e_dict), # Pass total length (base+created)
                   r_dict_len=len(r_dict),
                   cfg=cfg)
    model.to(device) # Move model to device

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # Training loop
    for epoch in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()

        # Generate anonymous entity embeddings
        anon_e_emb_1 = model.e_embedding_base.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim), device=device)
        anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim, device=device)
        torch.nn.init.xavier_uniform_(anon_e_emb_2)
        anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)

        # Calculate losses
        loss_cc_list = []
        loss_ec_list = []
        loss_ee_list = []

        # TBox Loss
        if tbox:
            for axiom_str in tbox: # Iterate over reconstructed strings
                # Pass dicts to forward
                fs_intersection = model.forward(axiom_str, anon_e_emb, c_dict, r_dict)
                loss_cc_list.append(model.get_cc_loss(fs_intersection))

        # ABox EC Loss
        if abox_ec:
            for axiom_str in abox_ec:
                try:
                    e_iri, c_iri = axiom_str.split(' ', 1)
                    e_id = e_dict[e_iri]
                    c_id = c_dict[c_iri]
                    loss_ec_list.append(model.get_ec_loss(e_id, c_id))
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping ABox EC axiom '{axiom_str}' due to error: {e}")

        # ABox EE Loss
        if abox_ee:
             for axiom_str in abox_ee:
                try:
                    e1_iri, r_iri, e2_iri = axiom_str.split(' ', 2)
                    e1_id = e_dict[e1_iri]
                    r_id = r_dict[r_iri]
                    e2_id = e_dict[e2_iri]
                    loss_ee_list.append(model.get_ee_loss(e1_id, r_id, e2_id))
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping ABox EE axiom '{axiom_str}' due to error: {e}")


        # Combine losses (handle empty lists)
        loss_cc = sum(loss_cc_list) / len(loss_cc_list) if loss_cc_list else torch.tensor(0.0).to(device)
        loss_ec = sum(loss_ec_list) / len(loss_ec_list) if loss_ec_list else torch.tensor(0.0).to(device)
        loss_ee = sum(loss_ee_list) / len(loss_ee_list) if loss_ee_list else torch.tensor(0.0).to(device)

        # Weighted loss (adjust weights as needed)
        loss = 0.5 * loss_cc + 0.5 * loss_ec + 0.0 * loss_ee # Original weights, EE loss seems ignored

        # Backpropagation
        if loss.requires_grad: # Only backprop if loss is not zero const
            loss.backward()
            optimizer.step()
        else:
             # Manually step if no grad (e.g., only non-trainable params or zero loss)
             # optimizer.step() # This might not be needed if loss is truly zero
             pass


        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{cfg.max_epochs} - Loss: {loss.item():.4f} (CC: {loss_cc.item():.4f}, EC: {loss_ec.item():.4f}, EE: {loss_ee.item():.4f})')
            # Optional: Add evaluation or prediction visualization here

    # After training, evaluate or predict
    model.eval()
    with torch.no_grad():
        # Regenerate anon embeddings for prediction if needed, or use last ones
        anon_e_emb_1 = model.e_embedding_base.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim), device=device)
        anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim, device=device)
        torch.nn.init.xavier_uniform_(anon_e_emb_2)
        anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)

        # Pass dicts to predict
        predictions_df = model.predict(anon_e_emb, c_dict, e_dict)
        print("\n--- Concept Membership Predictions ---")
        print(predictions_df.to_string()) # Print full dataframe

    return predictions_df # Return the prediction results


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Data path argument
    parser.add_argument('--data_path', default='../../data/Family/processed', type=str, # Default to a 'processed' subdirectory
                        help='Path to the directory containing preprocessed data files (tsv/txt)')
    # Tunable hyperparameters
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--emb_dim', default=50, type=int)
    parser.add_argument('--max_epochs', default=500, type=int)
    parser.add_argument('--max_measure', default='max', type=str)
    parser.add_argument('--t_norm', default='product', type=str, help='product, minmax, Łukasiewicz')
    parser.add_argument('--n_models', default=1, type=int) # Default to 1 for simpler run
    parser.add_argument('--anon_e', default=4, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')

    # Load data using the new function
    tbox, abox_ec, abox_ee, c_dict, e_dict, r_dict = get_data(cfg)

    # Train multiple models if specified
    all_results = []
    for i in range(cfg.n_models):
        print(f"\n--- Training Model {i+1}/{cfg.n_models} ---")
        result_df = get_one_model(cfg, tbox, abox_ec, abox_ee, c_dict, e_dict, r_dict)
        all_results.append(result_df)
        print(f"--- Model {i+1} Finished ---")
        # Optional: Save individual model results
        # result_df.to_csv(f'family_model_{i+1}_predictions.csv', index=False)

    # Optional: Aggregate results from multiple models (e.g., max pooling)
    if cfg.n_models > 1 and all_results:
        print("\n--- Aggregating Results (Max Pooling) ---")
        # Assuming all dataframes have the same columns and concepts in the same order
        numeric_cols = all_results[0].columns[1:] # Exclude 'concept' column
        aggregated_data = []
        concepts = all_results[0]['concept'].tolist()

        for concept_idx, concept_iri in enumerate(concepts):
            max_memberships = [-1.0] * len(numeric_cols) # Initialize with -1
            for model_result_df in all_results:
                memberships = model_result_df.iloc[concept_idx, 1:].astype(float).tolist()
                for entity_idx, membership in enumerate(memberships):
                    max_memberships[entity_idx] = max(max_memberships[entity_idx], membership)
            aggregated_data.append([concept_iri] + [round(m, 3) for m in max_memberships])

        aggregated_df = pd.DataFrame(aggregated_data, columns=all_results[0].columns)
        print(aggregated_df.to_string())
        # Optional: Save aggregated results
        # aggregated_df.to_csv('family_aggregated_predictions.csv', index=False)

    print("\n--- Script Finished ---")
