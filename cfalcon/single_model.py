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
    # Trim leading/trailing whitespace from the whole argument string first
    args_str = args_str.strip()
    # Handle empty argument list
    if not args_str:
        return []
    for i, char in enumerate(args_str):
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        # Split only on spaces that are *not* inside nested parentheses
        elif char == ' ' and balance == 0:
            # Check if the space is genuinely separating arguments
            # Avoid splitting inside IRIs or literals if possible (though this is hard without full parsing)
            # A simple heuristic: split only if the next char isn't part of the current arg continuation
            # For now, stick to basic space splitting at balance 0
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
            num_individuals: Number of unique individuals (entities). Should be >= 1.
            embedding_dim: The dimension for concept, role, and individual embeddings.
            fuzzy_logic: The type of fuzzy logic to use ('godel', 'lukasiewicz', 'product').
            device: The torch device to run computations on.
        """
        super().__init__()
        if num_individuals < 1:
             # Embedding layers require at least size 1.
             # If 0 individuals are truly needed, logic needs significant changes.
             # For now, enforce >= 1, potentially representing a dummy individual if none exist.
             print("Warning: num_individuals must be >= 1 for Embedding layers. Setting to 1.")
             num_individuals = 1

        self.num_concepts = num_concepts
        self.num_roles = num_roles
        self.num_individuals = num_individuals
        self.embedding_dim = embedding_dim
        self.fuzzy_logic = fuzzy_logic.lower()
        self.device = device

        # --- Embeddings ---
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        self.role_embeddings = nn.Embedding(num_roles, embedding_dim) # Simplified: needs transformation to relation
        self.individual_embeddings = nn.Embedding(num_individuals, embedding_dim)

        # Initialize embeddings (optional, but good practice)
        nn.init.xavier_uniform_(self.concept_embeddings.weight.data)
        nn.init.xavier_uniform_(self.role_embeddings.weight.data)
        nn.init.xavier_uniform_(self.individual_embeddings.weight.data)

        # --- Vocabulary Mappings (Assumed to be provided externally) ---
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
        # Allow empty dicts, but check against num_ provided if not empty
        if self.concept_to_idx:
            assert len(self.concept_to_idx) <= self.num_concepts
        if self.role_to_idx:
             assert len(self.role_to_idx) <= self.num_roles
        if self.individual_to_idx:
             # The number of individuals in the map might be 0, but self.num_individuals is >= 1
             assert len(self.individual_to_idx) <= self.num_individuals


    # --- Fuzzy Logic Operators ---
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
        return 1.0 - x

    def _logical_implies(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fuzzy Implication."""
        if self.fuzzy_logic == 'godel':
            return torch.where(x <= y, torch.ones_like(x), y)
        elif self.fuzzy_logic == 'lukasiewicz':
            return torch.clamp(1.0 - x + y, max=1.0)
        elif self.fuzzy_logic == 'product':
            # Avoid division by zero or near-zero
            return torch.where(x <= y, torch.ones_like(x), torch.where(x > 1e-9, torch.clamp(y / x, max=1.0), torch.ones_like(x)))
        else:
            raise ValueError(f"Unknown fuzzy logic: {self.fuzzy_logic}")

    def _logical_subsethood(self, c1_fs: torch.Tensor, c2_fs: torch.Tensor) -> torch.Tensor:
        """Calculates the degree of subsethood C1 <= C2. Typically inf_x(C1(x) -> C2(x))."""
        if c1_fs.numel() == 0 or c2_fs.numel() == 0: # Handle case with 0 individuals
             return torch.tensor(1.0, device=self.device) # Vacuously true
        implication_values = self._logical_implies(c1_fs, c2_fs)
        # Use torch.min as the infimum operator (consistent with Godel logic)
        subsethood_degree, _ = torch.min(implication_values, dim=-1) # Aggregate over individuals
        return subsethood_degree

    def _logical_equivalence(self, c1_fs: torch.Tensor, c2_fs: torch.Tensor) -> torch.Tensor:
        """Calculates the degree of equivalence C1 <=> C2."""
        if c1_fs.numel() == 0 or c2_fs.numel() == 0: # Handle case with 0 individuals
             return torch.tensor(1.0, device=self.device) # Vacuously true
        subset_c1_c2 = self._logical_subsethood(c1_fs, c2_fs)
        subset_c2_c1 = self._logical_subsethood(c2_fs, c1_fs)
        return self._logical_and(subset_c1_c2, subset_c2_c1)

    # --- Fuzzy Set / Relation Computation ---

    def _get_all_individual_embeddings(self) -> torch.Tensor:
        """Returns embeddings for all individuals."""
        return self.individual_embeddings.weight # Shape [num_individuals, embedding_dim]

    def _get_concept_fuzzy_set(self, concept_idx: int) -> torch.Tensor:
        """Computes the fuzzy set for a given concept index over all individuals."""
        if self.num_individuals == 0:
             return torch.empty(0, device=self.device) # Return empty tensor if no individuals

        c_emb = self.concept_embeddings(torch.tensor([concept_idx], device=self.device)) # [1, dim]
        all_ind_emb = self._get_all_individual_embeddings() # [num_ind, dim]
        # Example: Dot product similarity + sigmoid
        similarities = torch.matmul(c_emb, all_ind_emb.t()).squeeze(0) # [num_ind]
        fuzzy_set = torch.sigmoid(similarities) # Map to [0, 1]
        return fuzzy_set

    def _get_role_fuzzy_relation(self, role_idx: int) -> torch.Tensor:
        """Computes the fuzzy relation for a given role index over pairs of individuals."""
        if self.num_individuals == 0:
             # Return empty tensor with correct number of dimensions
             return torch.empty((0, 0), device=self.device)

        r_emb = self.role_embeddings(torch.tensor([role_idx], device=self.device)) # [1, dim]
        all_ind_emb = self._get_all_individual_embeddings() # [num_ind, dim]
        # Example: Simplified TransE-like score (adjust as needed)
        head_emb = all_ind_emb.unsqueeze(1) # [num_ind, 1, dim]
        tail_emb = all_ind_emb.unsqueeze(0) # [1, num_ind, dim]
        r_emb_exp = r_emb.unsqueeze(0)      # [1, 1, dim]

        # Score(x, r, y) = sigmoid( dot(emb_x + r_emb, emb_y) ) - adjust as needed
        projected_head = head_emb + r_emb_exp # [num_ind, 1, dim]
        # Use bmm for batch matmul: (num_ind, 1, dim) @ (num_ind, dim, num_ind) -> (num_ind, 1, num_ind)
        # Need tail_emb transposed and expanded: [num_ind, dim, num_ind]
        scores = torch.bmm(projected_head, tail_emb.transpose(1, 2).expand(self.num_individuals, -1, -1))
        scores = scores.squeeze(1) # [num_ind, num_ind]

        fuzzy_relation = torch.sigmoid(scores) # Map scores to [0, 1]
        return fuzzy_relation


    def _get_existential_restriction(self, role_fs: torch.Tensor, concept_fs: torch.Tensor) -> torch.Tensor:
        """Computes fuzzy set for 'exists R.C'. Sup_y ( R(x,y) AND C(y) )."""
        if self.num_individuals == 0: return torch.empty(0, device=self.device)
        # role_fs: [num_ind, num_ind] = R(x, y)
        # concept_fs: [num_ind] = C(y)
        concept_fs_expanded = concept_fs.unsqueeze(0) # [1, num_ind]
        conjunction_values = self._logical_and(role_fs, concept_fs_expanded) # [num_ind, num_ind]
        existential_fs, _ = torch.max(conjunction_values, dim=1) # [num_ind] (Sup over y)
        return existential_fs

    def _get_universal_restriction(self, role_fs: torch.Tensor, concept_fs: torch.Tensor) -> torch.Tensor:
        """Computes fuzzy set for 'forall R.C'. Inf_y ( R(x,y) -> C(y) )."""
        if self.num_individuals == 0: return torch.empty(0, device=self.device)
        # role_fs: [num_ind, num_ind] = R(x, y)
        # concept_fs: [num_ind] = C(y)
        concept_fs_expanded = concept_fs.unsqueeze(0) # [1, num_ind]
        implication_values = self._logical_implies(role_fs, concept_fs_expanded) # [num_ind, num_ind]
        universal_fs, _ = torch.min(implication_values, dim=1) # [num_ind] (Inf over y)
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
            if axiom_str is a class expression. Returns empty tensor if num_individuals is 0
            and the expression evaluates to a fuzzy set. Returns scalar 1.0 if num_individuals is 0
            and the expression evaluates to an axiom truth value (vacuously true).
        """
        axiom_str = axiom_str.strip()

        if self.concept_to_idx is None or self.role_to_idx is None or self.individual_to_idx is None:
            raise RuntimeError("Vocabulary mappings must be set before calling forward.")

        # --- Base Cases: Named Concepts, Roles, Individuals ---
        if axiom_str in self.concept_to_idx:
            concept_idx = self.concept_to_idx[axiom_str]
            return self._get_concept_fuzzy_set(concept_idx)
        if axiom_str == 'owl:Thing':
             if self.num_individuals == 0: return torch.empty(0, device=self.device)
             return torch.ones(self.num_individuals, device=self.device)
        if axiom_str == 'owl:Nothing':
             if self.num_individuals == 0: return torch.empty(0, device=self.device)
             return torch.zeros(self.num_individuals, device=self.device)
        # Roles and Individuals are typically not evaluated directly by forward

        # --- Recursive Cases: Constructors and Axioms ---
        # Use a more robust regex that handles potential spaces after constructor name
        constructor_match = re.match(r"([\w:]+)\s*\((.*)\)", axiom_str, re.DOTALL)
        if constructor_match:
            constructor = constructor_match.group(1)
            args_str = constructor_match.group(2)
            args = split_arguments(args_str)

            # --- Special Handling for NNF Axiom Patterns ---
            # Check if the top-level is ObjectIntersectionOf(arg1, ObjectComplementOf(arg2))
            # This pattern often represents SubClassOf(arg1, arg2) in NNF.
            # Also handle ObjectIntersectionOf(ObjectIntersectionOf(arg1, arg2), ObjectComplementOf(owl:Nothing)) -> Disjoint(arg1, arg2)
            if constructor == "ObjectIntersectionOf" and len(args) == 2:
                # Pattern 1: SubClassOf(arg1, arg2) -> ObjectIntersectionOf(arg1, ObjectComplementOf(arg2))
                complement_match = re.match(r"ObjectComplementOf\s*\((.*)\)", args[1], re.DOTALL)
                if complement_match:
                    arg1_str = args[0]
                    arg2_inner_str = complement_match.group(1).strip() # The argument inside ComplementOf

                    # Check for Disjoint(arg1_inner, arg2_inner) -> SubClassOf(Intersection(arg1,arg2), Nothing)
                    # -> ObjectIntersectionOf(ObjectIntersectionOf(arg1_inner, arg2_inner), ObjectComplementOf(owl:Nothing))
                    if arg2_inner_str == "owl:Nothing":
                         inner_intersection_match = re.match(r"ObjectIntersectionOf\s*\((.*)\)", arg1_str, re.DOTALL)
                         if inner_intersection_match:
                             inner_args_str = inner_intersection_match.group(1)
                             inner_args = split_arguments(inner_args_str)
                             if len(inner_args) == 2:
                                 # This is the DisjointClasses pattern
                                 c1_fs = self.forward(inner_args[0])
                                 c2_fs = self.forward(inner_args[1])
                                 intersection_fs = self._logical_and(c1_fs, c2_fs)
                                 nothing_fs = self.forward("owl:Nothing")
                                 # Return SubClassOf(intersection, Nothing)
                                 return self._logical_subsethood(intersection_fs, nothing_fs) # Returns scalar

                    # If not the Disjoint pattern, assume it's the SubClassOf pattern
                    fs1 = self.forward(arg1_str)
                    fs2 = self.forward(arg2_inner_str)
                    return self._logical_subsethood(fs1, fs2) # Returns scalar

            # --- Standard Class Expressions (Return Fuzzy Set) ---
            if constructor == "ObjectIntersectionOf":
                # If not the special axiom pattern above, treat as standard intersection
                if len(args) != 2: raise ValueError(f"ObjectIntersectionOf expects 2 arguments, got {len(args)} in '{axiom_str}'")
                fs1 = self.forward(args[0])
                fs2 = self.forward(args[1])
                return self._logical_and(fs1, fs2)
            elif constructor == "ObjectUnionOf":
                if len(args) != 2: raise ValueError(f"ObjectUnionOf expects 2 arguments, got {len(args)} in '{axiom_str}'")
                fs1 = self.forward(args[0])
                fs2 = self.forward(args[1])
                return self._logical_or(fs1, fs2)
            elif constructor == "ObjectComplementOf":
                if len(args) != 1: raise ValueError(f"ObjectComplementOf expects 1 argument, got {len(args)} in '{axiom_str}'")
                fs = self.forward(args[0])
                return self._logical_not(fs)
            elif constructor == "ObjectSomeValuesFrom":
                if len(args) != 2: raise ValueError(f"ObjectSomeValuesFrom expects 2 arguments, got {len(args)} in '{axiom_str}'")
                role_name = args[0]
                class_expr = args[1]
                if role_name not in self.role_to_idx: raise ValueError(f"Unknown role: {role_name}")
                role_idx = self.role_to_idx[role_name]
                role_relation = self._get_role_fuzzy_relation(role_idx)
                concept_fs = self.forward(class_expr)
                return self._get_existential_restriction(role_relation, concept_fs)
            elif constructor == "ObjectAllValuesFrom":
                if len(args) != 2: raise ValueError(f"ObjectAllValuesFrom expects 2 arguments, got {len(args)} in '{axiom_str}'")
                role_name = args[0]
                class_expr = args[1]
                if role_name not in self.role_to_idx: raise ValueError(f"Unknown role: {role_name}")
                role_idx = self.role_to_idx[role_name]
                role_relation = self._get_role_fuzzy_relation(role_idx)
                concept_fs = self.forward(class_expr)
                return self._get_universal_restriction(role_relation, concept_fs)
            # TODO: Add ObjectHasValue, Cardinality Restrictions if needed

            # --- Standard Axioms (Return Scalar Truth Value) ---
            elif constructor == "SubClassOf":
                if len(args) != 2: raise ValueError(f"SubClassOf expects 2 arguments, got {len(args)} in '{axiom_str}'")
                c1_fs = self.forward(args[0])
                c2_fs = self.forward(args[1])
                return self._logical_subsethood(c1_fs, c2_fs)
            elif constructor == "EquivalentClasses":
                # Assuming binary equivalence for now
                if len(args) != 2: raise ValueError(f"EquivalentClasses expects 2 arguments, got {len(args)} in '{axiom_str}'")
                c1_fs = self.forward(args[0])
                c2_fs = self.forward(args[1])
                return self._logical_equivalence(c1_fs, c2_fs)
            elif constructor == "DisjointClasses":
                # Disjoint(C, D) <=> (C and D) <= Nothing
                # Assuming binary disjointness for now
                if len(args) != 2: raise ValueError(f"DisjointClasses expects 2 arguments, got {len(args)} in '{axiom_str}'")
                c1_fs = self.forward(args[0])
                c2_fs = self.forward(args[1])
                intersection_fs = self._logical_and(c1_fs, c2_fs)
                nothing_fs = self.forward("owl:Nothing")
                return self._logical_subsethood(intersection_fs, nothing_fs)
            # TODO: Add SubObjectPropertyOf, Domain, Range etc. if needed

            # --- ABox Axioms (Return Scalar Truth Value) ---
            elif constructor == "ClassAssertion":
                if len(args) != 2: raise ValueError(f"ClassAssertion expects 2 arguments, got {len(args)} in '{axiom_str}'")
                class_expr = args[0]
                individual_name = args[1]
                if self.num_individuals == 0: # Cannot evaluate if no individuals
                     print(f"Warning: Cannot evaluate ClassAssertion '{axiom_str}' with zero individuals.")
                     return torch.tensor(0.0, device=self.device) # Or 1.0 if vacuously true? Assume 0.0
                if individual_name not in self.individual_to_idx: raise ValueError(f"Unknown individual: {individual_name}")
                ind_idx = self.individual_to_idx[individual_name]
                class_fs = self.forward(class_expr)
                # Truth value is the membership degree of the individual in the class fuzzy set
                return class_fs[ind_idx]
            elif constructor == "ObjectPropertyAssertion":
                if len(args) != 3: raise ValueError(f"ObjectPropertyAssertion expects 3 arguments, got {len(args)} in '{axiom_str}'")
                role_name = args[0]
                ind1_name = args[1]
                ind2_name = args[2]
                if self.num_individuals == 0: # Cannot evaluate if no individuals
                     print(f"Warning: Cannot evaluate ObjectPropertyAssertion '{axiom_str}' with zero individuals.")
                     return torch.tensor(0.0, device=self.device)
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
                raise NotImplementedError(f"Unsupported OWL constructor: {constructor} in '{axiom_str}'")
        else:
            # Should be an axiom type not fitting the Constructor(...) pattern, or error
             raise ValueError(f"Could not parse axiom or class expression: {axiom_str}")


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # 1. Define Vocabulary Mappings (Example)
    concepts = {"<urn:A>": 0, "<urn:B>": 1, "<urn:C>": 2, "owl:Thing": 3, "owl:Nothing": 4}
    roles = {"<urn:r>": 0}
    individuals = {"<urn:i>": 0, "<urn:j>": 1}

    num_concepts = len(concepts)
    num_roles = len(roles)
    num_individuals = len(individuals)
    emb_dim = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Instantiate the Model
    model = FuzzyOWLModel(num_concepts, num_roles, num_individuals, emb_dim, fuzzy_logic='godel', device=device)
    model.set_vocab_mappings(concepts, roles, individuals)
    model.eval() # Set to evaluation mode if not training

    # 3. Define Axioms (NNF Functional Syntax)
    axiom_subclass = "SubClassOf(<urn:A> <urn:B>)"
    axiom_equiv = "EquivalentClasses(<urn:A> <urn:B>)"
    axiom_disjoint = "DisjointClasses(<urn:A> <urn:B>)"
    axiom_class_assert = "ClassAssertion(<urn:A> <urn:i>)"
    axiom_prop_assert = "ObjectPropertyAssertion(<urn:r> <urn:i> <urn:j>)"
    axiom_some = "SubClassOf(<urn:A> ObjectSomeValuesFrom(<urn:r> <urn:B>))"

    # Axiom representing SubClassOf(<urn:A>, <urn:B>) in NNF
    axiom_nnf_subclass = "ObjectIntersectionOf(<urn:A> ObjectComplementOf(<urn:B>))"
    # Axiom representing DisjointClasses(<urn:A>, <urn:B>) in NNF -> SubClassOf(Intersection(A,B), Nothing)
    axiom_nnf_disjoint = "ObjectIntersectionOf(ObjectIntersectionOf(<urn:A> <urn:B>) ObjectComplementOf(owl:Nothing))"

    # Class Expression
    class_expr_intersect = "ObjectIntersectionOf(<urn:A> <urn:B>)"


    # 4. Evaluate Axioms and Expressions
    with torch.no_grad():
        tv_subclass = model(axiom_subclass)
        tv_equiv = model(axiom_equiv)
        tv_disjoint = model(axiom_disjoint)
        tv_class_assert = model(axiom_class_assert)
        tv_prop_assert = model(axiom_prop_assert)
        tv_some = model(axiom_some)
        tv_nnf_subclass = model(axiom_nnf_subclass) # Should now work
        tv_nnf_disjoint = model(axiom_nnf_disjoint) # Should now work
        fs_intersect = model(class_expr_intersect)   # Should return fuzzy set

    print(f"Axiom: {axiom_subclass} -> Truth: {tv_subclass.item():.4f}")
    print(f"Axiom: {axiom_equiv} -> Truth: {tv_equiv.item():.4f}")
    print(f"Axiom: {axiom_disjoint} -> Truth: {tv_disjoint.item():.4f}")
    print(f"Axiom: {axiom_class_assert} -> Truth: {tv_class_assert.item():.4f}")
    print(f"Axiom: {axiom_prop_assert} -> Truth: {tv_prop_assert.item():.4f}")
    print(f"Axiom: {axiom_some} -> Truth: {tv_some.item():.4f}")
    print(f"Axiom (NNF): {axiom_nnf_subclass} -> Truth: {tv_nnf_subclass.item():.4f}")
    print(f"Axiom (NNF): {axiom_nnf_disjoint} -> Truth: {tv_nnf_disjoint.item():.4f}")
    print(f"Class Expr: {class_expr_intersect} -> Fuzzy Set (shape {fs_intersect.shape}): {fs_intersect.cpu().numpy()}")

