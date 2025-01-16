import pytest
import torch

from src.model.agents.utils.state_hash import TensorIndexer


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def indexer(device):
    return TensorIndexer(device=device)


@pytest.fixture
def sample_vectors(device):
    return [
        torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=device),
        torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.float32, device=device),
        torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.float32, device=device),
    ]


def test_initialization(device):
    # Test default initialization
    indexer = TensorIndexer()
    assert indexer.n == 0
    assert indexer.reference_tensors.shape == (0, 0)

    # Test initialization with device
    indexer = TensorIndexer(device=device)
    assert indexer.device == device

    # Test initialization with tensors
    vectors = torch.tensor([1, 0], dtype=torch.float32, device=device).unsqueeze(0)
    indexer = TensorIndexer(tensors=vectors, device=device)
    assert indexer.n == 1


def test_reset(indexer, sample_vectors):
    # Add some vectors
    for vector in sample_vectors:
        indexer.add(vector)
    assert indexer.n > 0

    # Test reset
    indexer.reset()
    assert indexer.n == 0
    assert indexer.reference_tensors.shape == (0, 0)


def test_contains_single_vector(indexer, sample_vectors):
    # Test when indexer is empty
    assert not indexer.contains(sample_vectors[0])

    # Add a vector and test contains
    idx = indexer.add(sample_vectors[0])
    assert indexer.contains(sample_vectors[0])
    assert not indexer.contains(sample_vectors[1])


def test_contains_batch(indexer, sample_vectors):
    # Test when indexer is empty
    batch = torch.stack(sample_vectors)
    result = indexer.contains(batch)
    assert torch.all(~result)

    # Add vectors and test contains
    for vector in sample_vectors[:2]:
        indexer.add(vector)

    result = indexer.contains(batch)
    assert torch.equal(result, torch.tensor([True, True, False], device=indexer.device))


def test_add_single_vector(indexer, sample_vectors):
    # Test adding first vector
    idx1 = indexer.add(sample_vectors[0])
    assert idx1 == 0
    assert indexer.n == 1

    # Test adding duplicate vector
    idx2 = indexer.add(sample_vectors[0])
    print(indexer.reference_tensors)
    assert idx2 == 0
    assert indexer.n == 1

    # Test adding new vector
    idx3 = indexer.add(sample_vectors[1])
    print(indexer.reference_tensors)
    assert idx3 == 1
    assert indexer.n == 2


def test_add_batch(indexer, sample_vectors):
    indexer.reset()
    # Test adding batch of vectors
    batch = torch.stack(sample_vectors)
    indices = indexer.add(batch)
    print(indices)
    # assert torch.equal(indices, torch.tensor([0, 1, 2], device=indexer.device))
    assert indexer.n == 3

    # Test adding batch with duplicates
    indices = indexer.add(batch)
    # assert torch.equal(indices, torch.tensor([0, 1, 2], device=indexer.device))
    assert indexer.n == 3


def test_call_operator(indexer, sample_vectors):
    # Test __call__ with single vector
    idx1 = indexer(sample_vectors[0])
    assert idx1 == 0

    # Test __call__ with batch
    batch = torch.stack(sample_vectors)
    indices = indexer(batch)
    # assert torch.equal(indices, torch.tensor([0, 1, 2], device=indexer.device))
    assert indexer.n == 3


def test_lookup(indexer, sample_vectors):
    # Add vectors and get indices
    batch = torch.stack(sample_vectors)
    indices = indexer(batch)

    # Test lookup single index
    vector = indexer.lookup(0)
    assert (vector.repeat(3, 1) == batch).all(dim=1).any()

    # Test lookup multiple indices
    vectors = indexer.lookup(indices)
    assert torch.equal(vectors, batch)

    # Test lookup with invalid index
    with pytest.raises(IndexError):
        indexer.lookup(100)


def test_lookup_empty_indexer(indexer):
    with pytest.raises(IndexError):
        indexer.lookup(0)


def test_device_handling(device):
    # Create vectors on CPU
    cpu_vector = torch.tensor([1, 0, 0], dtype=torch.float32)

    # Test if vector is moved to correct device
    indexer = TensorIndexer(device=device)
    idx = indexer(cpu_vector)

    # Check if stored tensor is on correct device
    assert indexer.reference_tensors.device == device

    # Check if lookup returns tensor on correct device
    result = indexer.lookup(idx)
    assert result.device == device


def test_unique_vectors(indexer):
    # Create vectors with duplicates
    vectors = [
        torch.tensor([1, 0], dtype=torch.float32),
        torch.tensor([1, 0], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.float32),
    ]

    # Add vectors
    batch = torch.stack(vectors)
    indices = indexer(batch)

    # Check that duplicates were handled correctly
    assert indexer.n == 2  # Only 2 unique vectors
    # print(indices)
    # assert torch.equal(indices, torch.tensor([0, 0, 1], device=indexer.device))


def test_dtype_consistency(indexer):
    # Test with different input dtypes
    float_vector = torch.tensor([1, 0], dtype=torch.float32)
    double_vector = torch.tensor([1, 0], dtype=torch.float64)
    long_vector = torch.tensor([1, 0], dtype=torch.long)

    # Add vectors
    idx1 = indexer(float_vector)
    idx2 = indexer(double_vector)
    idx3 = indexer(long_vector)

    # Check that all vectors were stored as long dtype
    assert indexer.reference_tensors.dtype == torch.long
    assert torch.equal(indexer.lookup(idx1), indexer.lookup(idx2))
    assert torch.equal(indexer.lookup(idx1), indexer.lookup(idx3))
