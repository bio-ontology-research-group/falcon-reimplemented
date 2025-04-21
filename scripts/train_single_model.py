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
import pandas as pd # Import pandas for matrix output

# Add project root to path to import cfalcon modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from cfalcon.single_model import FuzzyOWLModel
from cfalcon.utils import read_list_from_file, extract_vocabulary

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Fuzzy OWL Model on TBox and ABox axioms.")
    parser.add_argument('--tbox_path', type=str, required=True, help='Path to the TBox file (functional syntax NNF, one axiom per line).')
    parser.add_argument('--abox_path', type=str, required=True, help='Path to the ABox file (functional syntax NNF, one axiom per line).')
    parser.add_argument('--output_dir', type=str, default='output_model', help='Directory to save the trained model and membership matrix.')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Dimension for embeddings.')
    parser.add_argument('--fuzzy_logic', type=str, default='godel', choices=['godel', 'lukasiewicz', 'product'], help='Type of fuzzy logic to use.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of axioms to process before optimizer step.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, or auto).')
    parser.add_argument('--output_membership_filename', type=str, default='membership_matrix.csv', help='Filename for the output membership matrix CSV within the output directory.')

    return parser.parse_args()

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

    # --- Extract Vocabulary ---
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
    num_individuals = len(individual_list)

    if num_individuals == 0:
        print("Warning: No individuals found in the vocabulary. Fuzzy set operations might be trivial or lead to errors.")
        # Consider exiting or adding a dummy individual if ABox operations are expected
        # sys.exit("Exiting due to lack of individuals.")

    # --- Initialize Model ---
    print("Initializing model...")
    # Ensure num_individuals is at least 1 for embedding layer creation, even if no individuals were found
    # This might need adjustment based on how the model handles zero individuals internally
    model_num_individuals = max(1, num_individuals)
    model = FuzzyOWLModel(
        num_concepts=num_concepts,
        num_roles=num_roles,
        num_individuals=model_num_individuals, # Use at least 1 for embedding layer
        embedding_dim=args.embedding_dim,
        fuzzy_logic=args.fuzzy_logic,
        device=device
    )
    model.set_vocab_mappings(concept_to_idx, role_to_idx, individual_to_idx)
    model.to(device)
    print(model)

    # --- Optimizer and Loss ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Use BCE loss: target is 1.0 for all axioms, loss is -log(truth_value)
    # Using BCELoss requires model output to be probability-like (0-1)
    criterion = nn.BCELoss()
    # Target tensor (all ones) - create on the correct device
    target = torch.tensor([1.0], device=device)

    # --- Training Loop ---
    print("Starting training...")
    model.train() # Set model to training mode

    skipped_axioms = 0
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
            try:
                # Skip axioms the current parser/model cannot handle
                # TODO: Make this check more robust or improve the parser
                if axiom_str.startswith("DifferentIndividuals"):
                     # print(f"Warning: Skipping unsupported axiom type: {axiom_str[:100]}...")
                     skipped_axioms += 1
                     continue
                # Skip ABox assertions if no individuals exist
                if num_individuals == 0 and ("ClassAssertion" in axiom_str or "ObjectPropertyAssertion" in axiom_str):
                    # print(f"Warning: Skipping ABox axiom due to no individuals: {axiom_str[:100]}...")
                    skipped_axioms +=1
                    continue


                truth_value = model(axiom_str)

                # Ensure truth_value is a scalar tensor for loss calculation
                if truth_value.numel() != 1:
                     # print(f"Warning: Skipping axiom due to non-scalar output ({truth_value.shape}): {axiom_str[:100]}...")
                     skipped_axioms += 1
                     continue

                # Clamp truth value slightly away from 0 and 1 for numerical stability with BCE
                truth_value = torch.clamp(truth_value, 1e-6, 1.0 - 1e-6)

                # Expand target to match the batch size (which is 1 in this case before accumulation)
                loss = criterion(truth_value.unsqueeze(0), target.expand(1)) # Match shape for BCE

                # Accumulate loss for batch
                accumulated_loss += loss
                axioms_in_batch += 1
                processed_in_epoch += 1
                total_processed_axioms += 1

                # Perform optimizer step after processing a batch
                if axioms_in_batch >= args.batch_size:
                    # Average loss over the batch
                    batch_loss = accumulated_loss / axioms_in_batch
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += accumulated_loss.item() # Add accumulated loss to epoch total
                    accumulated_loss = 0.0 # Reset accumulator
                    axioms_in_batch = 0    # Reset batch counter

            except (NotImplementedError, ValueError, RuntimeError, IndexError) as e:
                # Catch errors during forward pass (parsing, unknown entities, etc.)
                # print(f"Warning: Skipping axiom due to error: {e} | Axiom: {axiom_str[:100]}...")
                skipped_axioms += 1
                continue
            except Exception as e:
                print(f"\nError processing axiom: {axiom_str}")
                print(f"Unexpected error: {e}")
                # Decide whether to skip or re-raise
                skipped_axioms += 1
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
    print(f"Total axioms skipped: {skipped_axioms}")

    # --- Save Model ---
    model_save_path = output_dir / f"fuzzy_owl_model_{args.fuzzy_logic}_dim{args.embedding_dim}_epoch{args.epochs}.pth"
    print(f"Saving model to {model_save_path}...")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'concept_to_idx': concept_to_idx,
        'role_to_idx': role_to_idx,
        'individual_to_idx': individual_to_idx,
        'args': args,
    }, model_save_path)
    print("Model saved.")

    # --- Evaluate and Save Membership Matrix ---
    if num_individuals > 0:
        print("\nCalculating membership matrix...")
        model.eval() # Set model to evaluation mode

        membership_data = {}
        with torch.no_grad():
            for concept_name in tqdm(concept_list, desc="Evaluating concepts"):
                try:
                    # Get the fuzzy set for the concept (membership for all individuals)
                    fuzzy_set = model(concept_name) # Shape: [num_individuals]
                    if fuzzy_set.shape == (num_individuals,):
                         membership_data[concept_name] = fuzzy_set.cpu().numpy()
                    else:
                         print(f"Warning: Unexpected output shape {fuzzy_set.shape} for concept '{concept_name}'. Skipping.")
                except (NotImplementedError, ValueError, RuntimeError, IndexError) as e:
                    print(f"Warning: Could not evaluate concept '{concept_name}' due to error: {e}. Skipping.")
                except Exception as e:
                    print(f"Warning: Unexpected error evaluating concept '{concept_name}': {e}. Skipping.")


        if membership_data:
            # Create DataFrame
            membership_df = pd.DataFrame(membership_data, index=individual_list)
            membership_df.index.name = 'Individual'

            # Save to CSV
            membership_save_path = output_dir / args.output_membership_filename
            print(f"Saving membership matrix to {membership_save_path}...")
            try:
                membership_df.to_csv(membership_save_path, float_format='%.4f')
                print("Membership matrix saved.")
                # Optionally print part of the matrix
                print("\nMembership Matrix (sample):")
                print(membership_df.head())

            except Exception as e:
                print(f"Error saving membership matrix: {e}")
        else:
            print("No membership data was generated (possibly due to errors or no concepts).")

    else:
        print("\nSkipping membership matrix generation as no individuals were found in the vocabulary.")


if __name__ == '__main__':
    main()
