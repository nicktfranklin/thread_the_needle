import torch


class TensorIndexer:
    def __init__(self, tensors=None, device=None):
        """
        Initialize the indexer with optional tensors and device.
        Args:
            tensors: Optional list of tensors, each of shape (d) containing flattened one-hot vectors
            device: Optional torch.device. If None, uses CUDA if available, else CPU
        """
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.reset()

        if tensors is not None:
            for tensor in tensors:
                self.add(tensor)

    def reset(self):
        """Reset the indexer to an empty state"""
        self.reference_tensors = torch.empty(
            (0, 0), device=self.device, dtype=self.dtype
        )
        self.n = 0

    @torch.no_grad()
    def _compute_tensor_hash(self, tensor):
        """
        Compute deterministic hashes for flat vector(s).
        Args:
            tensor: Tensor of shape (d) or (m,d)
        Returns:
            torch.Tensor: Hash value(s) of shape () for (d) input or (m,) for (m,d) input
        """
        tensor = tensor.detach()
        indices = torch.argmax(tensor, dim=-1)  # () or (m,)
        return indices

    @torch.no_grad()
    def contains(self, query_tensor):
        """
        Check if vector(s) exist in the indexer.
        Args:
            query_tensor: Tensor of shape (d) or (m,d)
        Returns:
            bool or torch.Tensor: Boolean for single vector, boolean tensor of shape (m,) for batch
        """
        query_tensor = query_tensor.detach()
        if self.n == 0:
            return (
                False
                if query_tensor.dim() == 1
                else torch.zeros(
                    len(query_tensor), dtype=torch.bool, device=self.device
                )
            )

        if query_tensor.dim() == 1:
            return torch.any(
                torch.all(self.reference_tensors == query_tensor, dim=1)
            ).item()
        else:
            return torch.any(
                torch.all(
                    self.reference_tensors.unsqueeze(1) == query_tensor.unsqueeze(0),
                    dim=2,
                ),
                dim=0,
            )

    @torch.no_grad()
    def add(self, tensor):
        """
        Add new vector(s) to the indexer if they don't exist.
        Args:
            tensor: Tensor of shape (d) or (m,d)
        Returns:
            int or torch.Tensor: Index/indices of the vector(s), -1 for existing ones
        """
        tensor = tensor.detach().to(device=self.device, dtype=self.dtype)

        if tensor.dim() == 1:
            if self.contains(tensor):
                return -1

            if self.n == 0:
                self.reference_tensors = tensor.unsqueeze(0)
                self.n = 1
                return 0

            self.reference_tensors = torch.cat(
                [self.reference_tensors, tensor.unsqueeze(0)], dim=0
            )
            self.n += 1
            return self.n - 1
        else:
            m = len(tensor)
            exists = self.contains(tensor)
            new_indices = torch.full((m,), -1, device=self.device)

            new_tensors = tensor[~exists]
            if len(new_tensors) > 0:
                if self.n == 0:
                    self.reference_tensors = new_tensors
                    self.n = len(new_tensors)
                    new_indices[~exists] = torch.arange(self.n, device=self.device)
                else:
                    start_idx = self.n
                    self.reference_tensors = torch.cat(
                        [self.reference_tensors, new_tensors], dim=0
                    )
                    self.n += len(new_tensors)
                    new_indices[~exists] = torch.arange(
                        start_idx, self.n, device=self.device
                    )

            return new_indices

    @torch.no_grad()
    def __call__(self, query_tensor):
        """
        Get indices for vector(s), adding new ones if they don't exist.
        Args:
            query_tensor: Tensor of shape (d) or (m,d)
        Returns:
            torch.Tensor: Scalar tensor for (d) input or tensor of shape (m,) for (m,d) input
        """
        query_tensor = query_tensor.detach().to(device=self.device, dtype=self.dtype)

        if self.n == 0:
            if query_tensor.dim() == 1:
                self.add(query_tensor)
                return torch.tensor(0, device=self.device)
            else:
                m = len(query_tensor)
                self.reference_tensors = query_tensor
                self.n = m
                return torch.arange(m, device=self.device)

        if query_tensor.dim() == 1:
            matches = torch.all(self.reference_tensors == query_tensor, dim=1)
            if not torch.any(matches):
                new_idx = self.add(query_tensor)
                return torch.tensor(new_idx, device=self.device)
            return torch.where(matches)[0][0]
        else:
            matches = torch.all(
                self.reference_tensors.unsqueeze(1) == query_tensor.unsqueeze(0), dim=2
            )
            existing_matches = torch.any(matches, dim=0)
            indices = torch.full((len(query_tensor),), -1, device=self.device)

            for i in range(len(query_tensor)):
                if existing_matches[i]:
                    indices[i] = torch.where(matches[:, i])[0][0]

            new_tensors = query_tensor[~existing_matches]
            if len(new_tensors) > 0:
                new_indices = self.add(new_tensors)
                indices[~existing_matches] = new_indices[new_indices != -1]

            return indices

    def lookup(self, indices):
        """
        Look up original vectors from their indices.
        Args:
            indices: Integer or tensor of indices
        Returns:
            torch.Tensor: Original vector(s) of shape (d) for single index or (m,d) for multiple indices
        Raises:
            IndexError: If any index is out of range
        """
        if self.n == 0:
            raise IndexError("Indexer is empty")

        if isinstance(indices, int):
            if not 0 <= indices < self.n:
                raise IndexError(f"Index {indices} is out of range [0, {self.n})")
            return self.reference_tensors[indices]

        indices = torch.as_tensor(indices, device=self.device)
        if not torch.all((0 <= indices) & (indices < self.n)):
            raise IndexError(f"Some indices are out of range [0, {self.n})")

        return self.reference_tensors[indices]


# Example usage:
def test_indexer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create flattened one-hot vectors
    d = 6  # b*n = 2*3 = 6
    t1 = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=device)
    t2 = torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.float32, device=device)
    t3 = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.float32, device=device)

    indexer = TensorIndexer(device=device)

    # Test single vectors
    idx1 = indexer(t1)
    idx2 = indexer(t2)
    idx3 = indexer(t3)
    print("Indices:", idx1, idx2, idx3)

    # Test batch
    batch = torch.stack([t1, t2, t3])
    batch_indices = indexer(batch)
    print("Batch indices:", batch_indices)

    # Test lookup
    recovered = indexer.lookup(batch_indices)
    print("All vectors match:", torch.all(batch == recovered))

    # Test new vector
    t4 = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32, device=device)
    idx4 = indexer(t4)
    print("New vector index:", idx4)


if __name__ == "__main__":
    test_indexer()
