import torch
import torch.nn as nn
import re
from typing import Dict, List, Tuple, Optional

# Basic regular expressions for parsing functional syntax (simplified)
# These might need refinement for complex IRIs or nested structures
CONCEPT_IRI_RE = re.compile(r"<[^>]+>")
ROLE_IRI_RE = re.compile(r"<[^>]+>") # Same as concept for now
INDIVIDUAL_IRI_RE = re.compile(r"<[^>]+>") # Same as concept for now
LITERAL_RE = re.compile(r"\"([^\"]*)\"\^\^<[^>]+>") # Basic literal matching

# --- Helper Functions for Parsing (Simplified) ---

def find_matching_paren(text, start_index=0):
    """Finds the index of the matching parenthesis."""
    balance = 0
    for i in range(start_index, len(text)):
        if text[i] == '(':
            balance += 1
        elif text[i] == ')':
            balance -= 1
            if balance == 0:
                return i
    return -1 # Not found

def split_arguments(args_str):
    """Splits arguments within parentheses, respecting nested structures."""
    args = []
    current_arg_start = 0
    balance = 0
    for i, char in enumerate(args_str):
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        elif char == ' ' and balance == 0:
            # Split only if not inside nested parentheses
            args.append(args_str[current_arg_start:i].strip())
            current_arg_start = i + 1
    # Add the last argument
    args.append(args_str[current_arg_start:].strip())
    return [arg for arg in args if arg] # Filter out empty strings

# --- The Fuzzy OWL Model ---

class FuzzyOWLModel(nn.Module):
    """
    A PyTorch module to represent a single fuzzy interpretation of an OWL ontology.
    It evaluates the truth value of OWL axioms (TBox and ABox) provided
    in Functional Syntax NNF.
    """
    def __init__(self, num_concepts: int, num_roles: int, num_individuals: int, embedding_dim: int, fuzzy_logic: str = 'godel', device: torch.device = torch.device('cpu')):
        """
        Initializes the FuzzyOWLModel.

        Args:
            num_concepts: Number of unique concepts (classes) in the vocabulary.
            num_roles: Number of unique roles (object properties) in the vocabulary.
            num_individuals: Number of unique individuals (entities) in the vocabulary.
            embedding_dim: The dimension for concept, role, and individual embeddings.
            fuzzy_logic: The type of fuzzy logic to use ('godel', 'lukasiewicz', 'product').
            device: The torch device to run computations on.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_roles = num_roles
        self.num_individuals = num_individuals
        self.embedding_dim = embedding_dim
        self.fuzzy_logic = fuzzy_logic.lower()
        self.device = device

        # --- Embeddings ---
        # Concepts (Classes): Represented by their fuzzy sets (membership function)
        # We might embed the *parameters* of the membership function, or directly embed concepts
        # For simplicity here, let's assume a direct embedding mapped to a fuzzy set later.
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        # Roles (Object Properties): Represented by fuzzy relations
        self.role_embeddings = nn.Embedding(num_roles, embedding_dim) # Simplified: needs transformation to relation
        # Individuals (Entities): Represented by points in the embedding space
        self.individual_embeddings = nn.Embedding(num_individuals, embedding_dim)

        # Initialize embeddings (optional, but good practice)
        nn.init.xavier_uniform_(self.concept_embeddings.weight.data)
        nn.init.xavier_uniform_(self.role_embeddings.weight.data)
        nn.init.xavier_uniform_(self.individual_embeddings.weight.data)

        # --- Vocabulary Mappings (Assumed to be provided externally) ---
        # These dictionaries map IRIs/names to integer indices used by embeddings
        self.concept_to_idx: Optional[Dict[str, int]] = None
        self.role_to_idx: Optional[Dict[str, int]] = None
        self.individual_to_idx: Optional[Dict[str, int]] = None

        self.to(self.device)

    def set_vocab_mappings(self, concept_to_idx: Dict[str, int], role_to_idx: Dict[str, int], individual_to_idx: Dict[str, int]):
        """Sets the vocabulary mapping dictionaries."""
        self.concept_to_idx = concept_to_idx
        self.role_to_idx = role_to_idx
        self.individual_to_idx = individual_to_idx
        # Basic validation
        assert len(self.concept_to_idx) <= self.num_concepts
        assert len(self.role_to_idx) <= self.num_roles
        assert len(self.individual_to_idx) <= self.num_individuals

    # --- Fuzzy Logic Operators ---
    # These operate on fuzzy sets (tensors of shape [num_individuals]) or
    # fuzzy relations (tensors of shape [num_individuals, num_individuals])
    # or truth values (scalar tensors).

    def _logical_and(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy Intersection (t-norm)."""
        if self.fuzzy_logic == 'godel':
            return torch.minimum(x, y)
        elif self.fuzzy_logic == 'lukasiewicz':
            return torch.clamp(x + y - 1.0, min=0.0)
        elif self.fuzzy_logic == 'product':
            return x * y
        else:
            raise ValueError(f"Unknown fuzzy logic: {self.fuzzy_logic}")

    def _logical_or(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy Union (t-conorm)."""
        if self.fuzzy_logic == 'godel':
            return torch.maximum(x, y)
        elif self.fuzzy_logic == 'lukasiewicz':
            return torch.clamp(x + y, max=1.0)
        elif self.fuzzy_logic == 'product':
            return x + y - (x * y)
        else:
            raise ValueError(f"Unknown fuzzy logic: {self.fuzzy_logic}")

    def _logical_not(self, x: torch.Tensor) -> torch.Tensor:
        """Fuzzy Negation."""
        # Standard negation is often 1 - x regardless of the t-norm/t-conorm pair
        return 1.0 - x

    def _logical_implies(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy Implication."""
        # Implementation depends on the chosen t-norm/t-conorm
        if self.fuzzy_logic == 'godel':
            # Gödel implication: 1 if x <= y, else y
            return torch.where(x <= y, torch.ones_like(x), y)
        elif self.fuzzy_logic == 'lukasiewicz':
            # Łukasiewicz implication: min(1, 1 - x + y)
            return torch.clamp(1.0 - x + y, max=1.0)
        elif self.fuzzy_logic == 'product':
            # Product implication (Gaines): 1 if x <= y, else y/x (handle x=0)
            return torch.where(x <= y, torch.ones_like(x), torch.where(x > 1e-6, y / x, torch.ones_like(x))) # Avoid division by zero
        else:
            raise ValueError(f"Unknown fuzzy logic: {self.fuzzy_logic}")

    def _logical_subsethood(self, c1_fs: torch.Tensor, c2_fs: torch.Tensor) -> torch.Tensor:
        """Calculates the degree of subsethood C1 <= C2. Typically inf_x(C1(x) -> C2(x))."""
        implication_values = self._logical_implies(c1_fs, c2_fs)
        # The 'infimum' is often implemented using the corresponding t-norm (min for Godel)
        # For simplicity and common practice, we use torch.min here.
        subsethood_degree = torch.min(implication_values)
        return subsethood_degree

    def _logical_equivalence(self, c1_fs: torch.Tensor, c2_fs: torch.Tensor) -> torch.Tensor:
        """Calculates the degree of equivalence C1 <=> C2."""
        subset_c1_c2 = self._logical_subsethood(c1_fs, c2_fs)
        subset_c2_c1 = self._logical_subsethood(c2_fs, c1_fs)
        # Equivalence is the conjunction of both subsethoods
        return self._logical_and(subset_c1_c2, subset_c2_c1)

    # --- Fuzzy Set / Relation Computation ---

    def _get_all_individual_embeddings(self) -> torch.Tensor:
        """Returns embeddings for all individuals."""
        # In a simple model, the embeddings *are* the representation.
        # In more complex models (like BoxE), this might involve transformations.
        return self.individual_embeddings.weight

    def _get_concept_fuzzy_set(self, concept_idx: int) -> torch.Tensor:
        """
        Computes the fuzzy set for a given concept index.
        This maps the concept embedding to a membership function over individuals.
        Example: using distance or similarity in the embedding space.
        Placeholder: Assumes a simple transformation or direct interpretation.
        """
        # This is a placeholder. A common approach is to use similarity (e.g., dot product + sigmoid)
        # between the concept embedding and all individual embeddings.
        # C(x) = sigmoid( C_emb @ Ind_emb^T ) - needs adjustment based on model specifics.
        # For now, return a dummy tensor. Replace with actual logic.
        # print(f"Warning: _get_concept_fuzzy_set returning dummy tensor for concept {concept_idx}")
        # return torch.rand(self.num_individuals, device=self.device) # Dummy implementation

        # Example using dot product + sigmoid (adjust scaling/bias as needed)
        c_emb = self.concept_embeddings(torch.tensor([concept_idx], device=self.device)) # [1, dim]
        all_ind_emb = self._get_all_individual_embeddings() # [num_ind, dim]
        similarities = torch.matmul(c_emb, all_ind_emb.t()).squeeze(0) # [num_ind]
        fuzzy_set = torch.sigmoid(similarities) # Map to [0, 1]
        return fuzzy_set


    def _get_role_fuzzy_relation(self, role_idx: int) -> torch.Tensor:
        """
        Computes the fuzzy relation for a given role index.
        This maps the role embedding to a membership function over pairs of individuals.
        Example: R(x, y) = sigmoid( score(Ind_x, R_emb, Ind_y) )
        Placeholder: Assumes a simple transformation or direct interpretation.
        """
        # This is a placeholder. Common approaches involve bilinear maps (TransE style)
        # or more complex neural network scores.
        # R(x,y) = sigmoid( Ind_x^T @ W_r @ Ind_y ) or similar.
        # For now, return a dummy tensor. Replace with actual logic.
        # print(f"Warning: _get_role_fuzzy_relation returning dummy tensor for role {role_idx}")
        # return torch.rand(self.num_individuals, self.num_individuals, device=self.device) # Dummy

        # Example using a simplified TransE-like score (needs refinement)
        r_emb = self.role_embeddings(torch.tensor([role_idx], device=self.device)) # [1, dim]
        all_ind_emb = self._get_all_individual_embeddings() # [num_ind, dim]
        # Calculate score for all pairs (x, y): || emb_x + r_emb - emb_y ||
        # This requires broadcasting: [num_ind, 1, dim] + [1, 1, dim] - [1, num_ind, dim]
        head_emb = all_ind_emb.unsqueeze(1)
        tail_emb = all_ind_emb.unsqueeze(0)
        r_emb_exp = r_emb.unsqueeze(0)

        # Simplified score: Use dot product similarity instead of distance for illustration
        # score(x, r, y) = dot(emb_x + r_emb, emb_y) - adjust as needed
        projected_head = head_emb + r_emb_exp # [num_ind, 1, dim]
        scores = torch.matmul(projected_head, tail_emb.transpose(1, 2)).squeeze(-1) # [num_ind, num_ind]

        fuzzy_relation = torch.sigmoid(scores) # Map scores to [0, 1]
        return fuzzy_relation


    def _get_existential_restriction(self, role_fs: torch.Tensor, concept_fs: torch.Tensor) -> torch.Tensor:
        """Computes fuzzy set for 'exists R.C'. Sup_y ( R(x,y) AND C(y) )."""
        # role_fs: [num_ind, num_ind] = R(x, y)
        # concept_fs: [num_ind] = C(y)
        # We need to compute Sup_y ( t-norm(R(x,y), C(y)) ) for each x
        # Expand concept_fs to match role_fs dimensions for broadcasted AND
        concept_fs_expanded = concept_fs.unsqueeze(0) # [1, num_ind]
        # Apply t-norm element-wise (R(x,y) AND C(y))
        conjunction_values = self._logical_and(role_fs, concept_fs_expanded) # [num_ind, num_ind]
        # Take supremum (max) over y for each x
        existential_fs, _ = torch.max(conjunction_values, dim=1) # [num_ind]
        return existential_fs

    def _get_universal_restriction(self, role_fs: torch.Tensor, concept_fs: torch.Tensor) -> torch.Tensor:
        """Computes fuzzy set for 'forall R.C'. Inf_y ( R(x,y) -> C(y) )."""
        # role_fs: [num_ind, num_ind] = R(x, y)
        # concept_fs: [num_ind] = C(y)
        # We need to compute Inf_y ( Implies(R(x,y), C(y)) ) for each x
        concept_fs_expanded = concept_fs.unsqueeze(0) # [1, num_ind]
        # Apply implication element-wise (R(x,y) -> C(y))
        implication_values = self._logical_implies(role_fs, concept_fs_expanded) # [num_ind, num_ind]
        # Take infimum (min) over y for each x
        universal_fs, _ = torch.min(implication_values, dim=1) # [num_ind]
        return universal_fs

    # --- Recursive Forward Evaluation ---

    def forward(self, axiom_str: str) -> torch.Tensor:
        """
        Recursively evaluates the truth value of an OWL axiom string or
        computes the fuzzy set for a class expression string.

        Args:
            axiom_str: The OWL axiom or class expression in Functional Syntax NNF.

        Returns:
            A scalar tensor representing the truth value [0, 1] if axiom_str is an axiom,
            or a tensor of shape [num_individuals] representing the fuzzy set
            if axiom_str is a class expression.
        """
        axiom_str = axiom_str.strip()
        # print(f"Processing: {axiom_str}") # Debugging

        if self.concept_to_idx is None or self.role_to_idx is None or self.individual_to_idx is None:
            raise RuntimeError("Vocabulary mappings must be set before calling forward.")

        # --- Base Cases: Named Concepts, Roles, Individuals ---
        if axiom_str in self.concept_to_idx:
            concept_idx = self.concept_to_idx[axiom_str]
            return self._get_concept_fuzzy_set(concept_idx)
        if axiom_str == 'owl:Thing':
            return torch.ones(self.num_individuals, device=self.device)
        if axiom_str == 'owl:Nothing':
            return torch.zeros(self.num_individuals, device=self.device)
        # Note: Roles and Individuals are typically not evaluated directly by forward,
        # but appear as arguments within constructors.

        # --- Recursive Cases: Constructors and Axioms ---
        constructor_match = re.match(r"(\w+)\((.*)\)", axiom_str, re.DOTALL)
        if constructor_match:
            constructor = constructor_match.group(1)
            args_str = constructor_match.group(2)
            # print(f"Constructor: {constructor}, Args: {args_str}") # Debugging

            # Split arguments carefully, respecting nested structures
            # This simple splitting might fail on complex cases (e.g., literals with brackets)
            # A proper parser is needed for full robustness.
            args = split_arguments(args_str)
            # print(f"Split Args: {args}") # Debugging

            # --- Class Expressions ---
            if constructor == "ObjectIntersectionOf":
                fs1 = self.forward(args[0])
                fs2 = self.forward(args[1])
                return self._logical_and(fs1, fs2)
            elif constructor == "ObjectUnionOf":
                fs1 = self.forward(args[0])
                fs2 = self.forward(args[1])
                return self._logical_or(fs1, fs2)
            elif constructor == "ObjectComplementOf":
                fs = self.forward(args[0])
                return self._logical_not(fs)
            elif constructor == "ObjectSomeValuesFrom":
                role_name = args[0]
                class_expr = args[1]
                if role_name not in self.role_to_idx: raise ValueError(f"Unknown role: {role_name}")
                role_idx = self.role_to_idx[role_name]
                role_relation = self._get_role_fuzzy_relation(role_idx)
                concept_fs = self.forward(class_expr)
                return self._get_existential_restriction(role_relation, concept_fs)
            elif constructor == "ObjectAllValuesFrom":
                role_name = args[0]
                class_expr = args[1]
                if role_name not in self.role_to_idx: raise ValueError(f"Unknown role: {role_name}")
                role_idx = self.role_to_idx[role_name]
                role_relation = self._get_role_fuzzy_relation(role_idx)
                concept_fs = self.forward(class_expr)
                return self._get_universal_restriction(role_relation, concept_fs)
            # TODO: Add ObjectHasValue, Cardinality Restrictions if needed

            # --- TBox Axioms (return scalar truth value) ---
            elif constructor == "SubClassOf":
                c1_fs = self.forward(args[0])
                c2_fs = self.forward(args[1])
                return self._logical_subsethood(c1_fs, c2_fs)
            elif constructor == "EquivalentClasses":
                c1_fs = self.forward(args[0])
                c2_fs = self.forward(args[1])
                return self._logical_equivalence(c1_fs, c2_fs)
            elif constructor == "DisjointClasses":
                # Disjoint(C, D) <=> (C and D) <= Nothing
                c1_fs = self.forward(args[0])
                c2_fs = self.forward(args[1])
                intersection_fs = self._logical_and(c1_fs, c2_fs)
                nothing_fs = self.forward("owl:Nothing")
                return self._logical_subsethood(intersection_fs, nothing_fs)
            # TODO: Add SubObjectPropertyOf, Domain, Range etc. if needed

            # --- ABox Axioms (return scalar truth value) ---
            elif constructor == "ClassAssertion":
                class_expr = args[0]
                individual_name = args[1]
                if individual_name not in self.individual_to_idx: raise ValueError(f"Unknown individual: {individual_name}")
                ind_idx = self.individual_to_idx[individual_name]
                class_fs = self.forward(class_expr)
                # Truth value is the membership degree of the individual in the class fuzzy set
                return class_fs[ind_idx]
            elif constructor == "ObjectPropertyAssertion":
                role_name = args[0]
                ind1_name = args[1]
                ind2_name = args[2]
                if role_name not in self.role_to_idx: raise ValueError(f"Unknown role: {role_name}")
                if ind1_name not in self.individual_to_idx: raise ValueError(f"Unknown individual: {ind1_name}")
                if ind2_name not in self.individual_to_idx: raise ValueError(f"Unknown individual: {ind2_name}")
                role_idx = self.role_to_idx[role_name]
                ind1_idx = self.individual_to_idx[ind1_name]
                ind2_idx = self.individual_to_idx[ind2_name]
                role_relation = self._get_role_fuzzy_relation(role_idx)
                # Truth value is the membership degree of the pair in the role fuzzy relation
                return role_relation[ind1_idx, ind2_idx]
            # TODO: Add NegativeObjectPropertyAssertion, DifferentIndividuals etc. if needed

            else:
                raise NotImplementedError(f"Unsupported OWL constructor: {constructor}")
        else:
            # Should be an axiom type not fitting the Constructor(...) pattern, or error
            # Example: DifferentIndividuals( a b ) - needs specific handling if required
             raise ValueError(f"Could not parse axiom or class expression: {axiom_str}")


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # 1. Define Vocabulary Mappings (Example)
    concepts = {"<http://example.org/Person>": 0, "<http://example.org/Male>": 1, "owl:Thing": 2, "owl:Nothing": 3}
    roles = {"<http://example.org/hasChild>": 0}
    individuals = {"<http://example.org/John>": 0, "<http://example.org/Peter>": 1, "<http://example.org/Mary>": 2}

    num_concepts = len(concepts) # Adjust if Thing/Nothing handled specially
    num_roles = len(roles)
    num_individuals = len(individuals)
    emb_dim = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Instantiate the Model
    model = FuzzyOWLModel(num_concepts, num_roles, num_individuals, emb_dim, fuzzy_logic='godel', device=device)
    model.set_vocab_mappings(concepts, roles, individuals)
    model.eval() # Set to evaluation mode if not training

    # 3. Define Axioms (NNF Functional Syntax)
    axiom1 = "SubClassOf(<http://example.org/Male> <http://example.org/Person>)"
    axiom2 = "ClassAssertion(<http://example.org/Person> <http://example.org/John>)"
    axiom3 = "ObjectPropertyAssertion(<http://example.org/hasChild> <http://example.org/John> <http://example.org/Peter>)"
    axiom4 = "SubClassOf(<http://example.org/Person> ObjectSomeValuesFrom(<http://example.org/hasChild> <http://example.org/Person>))" # Person <= exists hasChild.Person
    axiom5 = "EquivalentClasses(<http://example.org/Male> ObjectComplementOf(<http://example.org/Female>))" # Assuming Female is concept 4

    # Add Female to vocab for axiom5
    concepts["<http://example.org/Female>"] = 4
    num_concepts = len(concepts)
    # Re-instantiate or resize embeddings if vocab changes dynamically (complex)
    # For simplicity, assume fixed vocab known at init
    # Let's re-init here for demo purposes
    model = FuzzyOWLModel(num_concepts, num_roles, num_individuals, emb_dim, fuzzy_logic='godel', device=device)
    model.set_vocab_mappings(concepts, roles, individuals)
    model.eval()


    # 4. Evaluate Axioms
    with torch.no_grad():
        truth_value1 = model(axiom1)
        truth_value2 = model(axiom2)
        truth_value3 = model(axiom3)
        truth_value4 = model(axiom4)
        # truth_value5 = model(axiom5) # This will fail until Female is handled properly

    print(f"Axiom 1 Truth: {axiom1} -> {truth_value1.item():.4f}")
    print(f"Axiom 2 Truth: {axiom2} -> {truth_value2.item():.4f}")
    print(f"Axiom 3 Truth: {axiom3} -> {truth_value3.item():.4f}")
    print(f"Axiom 4 Truth: {axiom4} -> {truth_value4.item():.4f}")
    # print(f"Axiom 5 Truth: {axiom5} -> {truth_value5.item():.4f}")

    # Example: Evaluate a class expression (returns a fuzzy set)
    class_expr = "ObjectIntersectionOf(<http://example.org/Person> <http://example.org/Male>)"
    with torch.no_grad():
        fuzzy_set = model(class_expr)
    print(f"\nFuzzy set for: {class_expr}")
    print(f"Shape: {fuzzy_set.shape}")
    print(f"Values (first 5): {fuzzy_set[:5].cpu().numpy()}")

