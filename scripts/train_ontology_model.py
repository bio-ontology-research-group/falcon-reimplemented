import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import sys
import random
import time
from tqdm import tqdm # For progress bar
import numpy as np # Import numpy for seeding
import pandas as pd # Import pandas for potential future matrix output
import re # Import re for checking unsupported constructors

# Add project root to path to import cfalcon modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the multi-model and utility functions
from cfalcon.model import FuzzyOntologyModel
from cfalcon.utils import read_list_from_file, extract_vocabulary

# Define known unsupported constructors (add more as needed)
UNSUPPORTED_CONSTRUCTORS = {
    "ObjectMinCardinality",
    "ObjectMaxCardinality",
    "ObjectExactCardinality",
    "DataHasValue", # Add data property related constructors if they appear
    "DataSomeValuesFrom",
    "DataAllValuesFrom",
    "DataMinCardinality",
    "DataMaxCardinality",
    "DataExactCardinality",
    "DifferentIndividuals", # Keep this explicit check too
    # Add others based on potential errors, e.g., related to datatypes
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Fuzzy Ontology Model (multi-model) on TBox and ABox axioms.")
    parser.add_argument('--tbox_path', type=str, required=True, help='Path to the TBox file (functional syntax NNF, one axiom per line).')
    parser.add_argument('--abox_path', type=str, required=True, help='Path to the ABox file (functional syntax NNF, one axiom per line).')
    parser.add_argument('--output_dir', type=str, default='output_ontology_model', help='Directory to save the trained ontology model.')
    parser.add_argument('--num_models', type=int, default=5, help='Number of individual fuzzy models to create.')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Dimension for embeddings in each model.')
    parser.add_argument('--fuzzy_logic', type=str, default='godel', choices=['godel', 'lukasiewicz', 'product'], help='Type of fuzzy logic to use.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of axioms to process before optimizer step.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, or auto).')
    # Removed membership matrix output for now, can be added back if needed for multi-model context

    return parser.parse_args()

def check_unsupported(axiom_str):
    """Checks if an axiom string contains known unsupported constructors."""
    # Simple string checking - might be too broad but avoids full parsing here
    for constructor in UNSUPPORTED_CONSTRUCTORS:
        # Use regex to check for the constructor name followed by '('
        if re.search(r'\b' + re.escape(constructor) + r'\s*\(', axiom_str):
            return True
    return False

def main():
    args = parse_args()

    # --- Setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Data ---
    print("Loading axioms...")
    tbox_axioms = read_list_from_file(args.tbox_path)
    abox_axioms = read_list_from_file(args.abox_path)
    all_axioms = tbox_axioms + abox_axioms
    print(f"Loaded {len(tbox_axioms)} TBox axioms and {len(abox_axioms)} ABox axioms.")
    print(f"Total axioms for training: {len(all_axioms)}")

    if not all_axioms:
        print("Error: No axioms loaded. Exiting.")
        sys.exit(1)

    # --- Identify Unsupported Axioms ---
    print("Scanning for unsupported axioms...")
    unsupported_axioms_set = set()
    found_unsupported_types = set()
    for axiom_str in all_axioms:
        for constructor in UNSUPPORTED_CONSTRUCTORS:
             if re.search(r'\b' + re.escape(constructor) + r'\s*\(', axiom_str):
                 unsupported_axioms_set.add(axiom_str)
                 found_unsupported_types.add(constructor)
                 break # Move to next axiom once one unsupported constructor is found

    if unsupported_axioms_set:
        print(f"Found {len(unsupported_axioms_set)} axioms containing unsupported constructors.")
        print(f"Unsupported constructor types found: {', '.join(sorted(list(found_unsupported_types)))}")
        print("These axioms will be skipped during training.")
    else:
        print("No known unsupported constructors found in the axioms.")


    # --- Extract Vocabulary ---
    # Extract from *all* axioms initially, even unsupported ones,
    # to ensure entities within them are known if they appear elsewhere.
    print("Extracting vocabulary...")
    concepts, roles, individuals = extract_vocabulary(all_axioms)
    print(f"Found {len(concepts)} concepts, {len(roles)} roles, {len(individuals)} individuals.")

    # Create mappings
    concept_list = sorted(list(concepts))
    role_list = sorted(list(roles))
    individual_list = sorted(list(individuals))

    concept_to_idx = {name: i for i, name in enumerate(concept_list)}
    role_to_idx = {name: i for i, name in enumerate(role_list)}
    individual_to_idx = {name: i for i, name in enumerate(individual_list)}

    num_concepts = len(concept_list)
    num_roles = len(role_list)
    num_individuals = len(individual_list) # Pass actual count

    if num_individuals == 0:
        print("Warning: No individuals found in the vocabulary. ABox axioms will be skipped during training.")

    # --- Initialize Model ---
    print(f"Initializing FuzzyOntologyModel with {args.num_models} individual models...")
    # FuzzyOntologyModel handles num_individuals >= 1 internally for FuzzyOWLModel
    model = FuzzyOntologyModel(
        num_models=args.num_models,
        num_concepts=num_concepts,
        num_roles=num_roles,
        num_individuals=num_individuals, # Pass actual count
        embedding_dim=args.embedding_dim,
        fuzzy_logic=args.fuzzy_logic,
        device=device
    )
    model.set_vocab_mappings(concept_to_idx, role_to_idx, individual_to_idx)
    model.to(device)
    print(model)
    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")


    # --- Optimizer and Loss ---
    # Optimize parameters of all underlying models
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Use BCE loss on the *aggregated* truth value (entailment degree)
    criterion = nn.BCELoss()
    # Target tensor (all ones) - create on the correct device
    target = torch.tensor([1.0], device=device)

    # --- Training Loop ---
    print("Starting training...")
    model.train() # Set model to training mode

    other_skipped_axioms = 0 # Count axioms skipped for reasons other than unsupported constructors
    total_processed_axioms = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        processed_in_epoch = 0
        optimizer.zero_grad()
        accumulated_loss = 0.0
        axioms_in_batch = 0

        # Shuffle axioms for each epoch
        random.shuffle(all_axioms)

        progress_bar = tqdm(all_axioms, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for axiom_str in progress_bar:
            # --- Silent Skipping of Pre-identified Unsupported Axioms ---
            if axiom_str in unsupported_axioms_set:
                continue

            # --- Skip ABox assertions if no individuals exist ---
            if num_individuals == 0 and ("ClassAssertion" in axiom_str or "ObjectPropertyAssertion" in axiom_str):
                # This check might be redundant if FuzzyOntologyModel handles it, but keep for clarity
                other_skipped_axioms += 1
                continue

            try:
                # Get the aggregated truth value (entailment degree) from the FuzzyOntologyModel
                entailment_degree = model(axiom_str)

                # Ensure entailment_degree is a scalar tensor for loss calculation
                if entailment_degree.numel() != 1:
                     # This might happen if all sub-models failed for an axiom (e.g., unknown entity)
                     # print(f"Warning: Skipping axiom due to non-scalar aggregated output ({entailment_degree.shape}): {axiom_str[:100]}...")
                     other_skipped_axioms += 1
                     continue

                # Clamp truth value slightly away from 0 and 1 for numerical stability with BCE
                entailment_degree = torch.clamp(entailment_degree, 1e-6, 1.0 - 1e-6)

                # Calculate loss based on the aggregated entailment degree
                loss = criterion(entailment_degree.unsqueeze(0), target.expand(1)) # Match shape for BCE

                # Accumulate loss for batch
                accumulated_loss += loss
                axioms_in_batch += 1
                processed_in_epoch += 1
                total_processed_axioms += 1

                # Perform optimizer step after processing a batch
                if axioms_in_batch >= args.batch_size:
                    # Average loss over the batch
                    batch_loss = accumulated_loss / axioms_in_batch
                    # Backpropagate through the aggregated loss - gradients flow to all sub-models
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += accumulated_loss.item() # Add accumulated loss to epoch total
                    accumulated_loss = 0.0 # Reset accumulator
                    axioms_in_batch = 0    # Reset batch counter

            except (NotImplementedError, ValueError, RuntimeError, IndexError) as e:
                # Catch *other* errors during forward pass (parsing, unknown entities, etc.)
                # These should ideally not be the pre-identified unsupported constructors anymore
                # print(f"Warning: Skipping axiom due to runtime error: {e} | Axiom: {axiom_str[:100]}...")
                other_skipped_axioms += 1
                # Ensure gradients are cleared if an error occurs mid-batch accumulation
                optimizer.zero_grad()
                accumulated_loss = 0.0
                axioms_in_batch = 0
                continue
            except Exception as e:
                print(f"\nError processing axiom: {axiom_str}")
                print(f"Unexpected error: {e}")
                other_skipped_axioms += 1
                # Ensure gradients are cleared
                optimizer.zero_grad()
                accumulated_loss = 0.0
                axioms_in_batch = 0
                continue # Skip this axiom

        # Process any remaining axioms in the last partial batch
        if axioms_in_batch > 0:
            batch_loss = accumulated_loss / axioms_in_batch
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss.item()

        epoch_duration = time.time() - epoch_start_time
        avg_loss = total_loss / processed_in_epoch if processed_in_epoch > 0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.6f} | Duration: {epoch_duration:.2f}s | Processed: {processed_in_epoch}")

    print(f"\nTraining finished.")
    print(f"Total axioms processed: {total_processed_axioms}")
    print(f"Axioms skipped due to unsupported constructors: {len(unsupported_axioms_set)}")
    print(f"Axioms skipped due to other runtime errors: {other_skipped_axioms}")

    # --- Save Model ---
    # Save the entire FuzzyOntologyModel, which includes all sub-models
    model_save_path = output_dir / f"fuzzy_ontology_model_n{args.num_models}_{args.fuzzy_logic}_dim{args.embedding_dim}_epoch{args.epochs}.pth"
    print(f"Saving ontology model to {model_save_path}...")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(), # Saves state of FuzzyOntologyModel and its ModuleList
        'optimizer_state_dict': optimizer.state_dict(),
        'concept_to_idx': concept_to_idx,
        'role_to_idx': role_to_idx,
        'individual_to_idx': individual_to_idx,
        'args': args,
    }, model_save_path)
    print("Ontology model saved.")

    # --- Optional: Evaluate Entailment Degree on Training Data (Example) ---
    # You might want to add a separate evaluation step using test data later
    print("\nExample: Evaluating entailment degree on the first 5 supported training axioms...")
    model.eval() # Set model to evaluation mode
    evaluated_count = 0
    with torch.no_grad():
        for axiom_str in all_axioms:
             if axiom_str in unsupported_axioms_set: # Skip unsupported ones here too
                 continue
             if evaluated_count >= 5:
                 break
             try:
                 entailment_degree = model(axiom_str)
                 print(f"Axiom: {axiom_str[:100]}... -> Entailment Degree: {entailment_degree.item():.4f}")
                 evaluated_count += 1
             except Exception as e:
                 print(f"Could not evaluate axiom: {axiom_str[:100]}... Error: {e}")
                 evaluated_count += 1 # Count as evaluated even if error


if __name__ == '__main__':
    main()
