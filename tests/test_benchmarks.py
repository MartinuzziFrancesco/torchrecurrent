import pytest
import torch

from torchrecurrent.benchmarks import (
    adding_problem,
    copy_memory,
    penn_treebank,
    sequential_cifar10,
    sequential_mnist,
    timit,
)


def test_adding_problem_matches_definition():
    inputs, targets = adding_problem(
        9,
        32,
        return_dataloader=False,
        generator=torch.Generator().manual_seed(0),
        dtype=torch.float64,
    )

    assert inputs.shape == (32, 9, 2)
    assert targets.shape == (32, 1)
    assert inputs.dtype == targets.dtype == torch.float64
    assert torch.all(inputs[:, :4, 1].sum(dim=1) == 1)
    assert torch.all(inputs[:, 4:, 1].sum(dim=1) == 1)
    torch.testing.assert_close(targets[:, 0], (inputs[:, :, 0] * inputs[:, :, 1]).sum(1))


def test_adding_problem_dataloader():
    loader = adding_problem(8, 7, batch_size=4, shuffle=False)
    inputs, targets = next(iter(loader))

    assert inputs.shape == (4, 8, 2)
    assert targets.shape == (4, 1)


@pytest.mark.parametrize("one_hot", [False, True])
def test_copy_memory_matches_definition(one_hot):
    inputs, targets = copy_memory(
        5,
        4,
        memory_length=3,
        return_dataloader=False,
        one_hot=one_hot,
        generator=torch.Generator().manual_seed(0),
    )
    token_inputs = inputs.argmax(-1) if one_hot else inputs

    assert token_inputs.shape == (4, 11)
    assert targets.shape == (4, 11)
    assert torch.all(token_inputs[:, :3] < 8)
    assert torch.all(token_inputs[:, 3:7] == 8)
    assert torch.all(token_inputs[:, 7] == 9)
    assert torch.all(token_inputs[:, 8:] == 8)
    assert torch.all(targets[:, :8] == 8)
    assert torch.equal(targets[:, 8:], token_inputs[:, :3])
    if one_hot:
        assert inputs.shape == (4, 11, 10)
        assert inputs.is_floating_point()


@pytest.mark.parametrize(
    "function,args",
    [
        (adding_problem, (1, 2)),
        (adding_problem, (2, 0)),
        (copy_memory, (0, 2)),
        (copy_memory, (2, 0)),
    ],
)
def test_benchmarks_validate_sizes(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_sequential_mnist_normalizes_and_flattens_images():
    images = (torch.arange(2 * 28 * 28) % 256).to(torch.uint8).reshape(2, 1, 28, 28)
    targets = torch.tensor([3, 7], dtype=torch.int32)

    sequences, result_targets = sequential_mnist(
        images, targets, return_dataloader=False, dtype=torch.float64
    )

    assert sequences.shape == (2, 784, 1)
    assert sequences.dtype == torch.float64
    assert result_targets.dtype == torch.long
    torch.testing.assert_close(sequences[0, :, 0], images[0].flatten().double() / 255)
    assert torch.equal(result_targets, targets.long())


def test_sequential_mnist_applies_one_fixed_permutation():
    images = torch.arange(2 * 28 * 28).reshape(2, 28, 28).float()
    targets = torch.tensor([0, 1])
    permutation = torch.arange(783, -1, -1)

    sequences, _ = sequential_mnist(
        images,
        targets,
        permutation=permutation,
        return_dataloader=False,
        normalize=False,
    )

    assert torch.equal(sequences[:, :, 0], images.flatten(1)[:, permutation])


def test_sequential_mnist_dataloader_is_layer_ready():
    loader = sequential_mnist(
        torch.zeros(5, 28, 28, dtype=torch.uint8),
        torch.arange(5),
        batch_size=3,
        shuffle=False,
    )
    sequences, targets = next(iter(loader))

    assert sequences.shape == (3, 784, 1)
    assert targets.shape == (3,)


@pytest.mark.parametrize(
    "images,targets,permutation",
    [
        (torch.zeros(2, 27, 28), torch.zeros(2), None),
        (torch.zeros(2, 3, 28, 28), torch.zeros(2), None),
        (torch.zeros(2, 28, 28), torch.zeros(3), None),
        (torch.zeros(2, 28, 28), torch.zeros(2), torch.zeros(783)),
        (torch.zeros(2, 28, 28), torch.zeros(2), torch.zeros(784)),
    ],
)
def test_sequential_mnist_validates_inputs(images, targets, permutation):
    with pytest.raises(ValueError):
        sequential_mnist(images, targets, permutation=permutation)


def test_sequential_cifar10_normalizes_and_flattens_images():
    images = (torch.arange(2 * 32 * 32 * 3) % 256).to(torch.uint8).reshape(2, 32, 32, 3)
    targets = torch.tensor([3, 7], dtype=torch.int32)

    sequences, result_targets = sequential_cifar10(
        images, targets, return_dataloader=False, dtype=torch.float64
    )

    assert sequences.shape == (2, 1024, 3)
    assert sequences.dtype == torch.float64
    assert result_targets.dtype == torch.long
    torch.testing.assert_close(sequences[0], images[0].reshape(1024, 3).double() / 255)
    assert torch.equal(result_targets, targets.long())


def test_sequential_cifar10_accepts_channels_first_images():
    images = torch.arange(2 * 3 * 32 * 32).reshape(2, 3, 32, 32).float()
    targets = torch.tensor([0, 1])

    sequences, _ = sequential_cifar10(
        images, targets, return_dataloader=False, normalize=False
    )

    expected = images.permute(0, 2, 3, 1).reshape(2, 1024, 3)
    torch.testing.assert_close(sequences, expected)


def test_sequential_cifar10_applies_one_fixed_permutation():
    images = torch.arange(2 * 32 * 32 * 3).reshape(2, 32, 32, 3).float()
    targets = torch.tensor([0, 1])
    permutation = torch.arange(1023, -1, -1)

    sequences, _ = sequential_cifar10(
        images,
        targets,
        permutation=permutation,
        return_dataloader=False,
        normalize=False,
    )

    expected = images.reshape(2, 1024, 3)[:, permutation]
    torch.testing.assert_close(sequences, expected)


def test_sequential_cifar10_dataloader_is_layer_ready():
    loader = sequential_cifar10(
        torch.zeros(5, 32, 32, 3, dtype=torch.uint8),
        torch.arange(5),
        batch_size=3,
        shuffle=False,
    )
    sequences, targets = next(iter(loader))

    assert sequences.shape == (3, 1024, 3)
    assert targets.shape == (3,)


@pytest.mark.parametrize(
    "images,targets,permutation",
    [
        (torch.zeros(2, 31, 32, 3), torch.zeros(2), None),
        (torch.zeros(2, 32, 32, 4), torch.zeros(2), None),
        (torch.zeros(2, 32, 32, 3), torch.zeros(3), None),
        (torch.zeros(2, 32, 32, 3), torch.zeros(2), torch.zeros(1023)),
        (torch.zeros(2, 32, 32, 3), torch.zeros(2), torch.zeros(1024)),
    ],
)
def test_sequential_cifar10_validates_inputs(images, targets, permutation):
    with pytest.raises(ValueError):
        sequential_cifar10(images, targets, permutation=permutation)


def test_penn_treebank_builds_vocabulary_and_shifted_streams(tmp_path):
    train = tmp_path / "ptb.train.txt"
    validation = tmp_path / "ptb.valid.txt"
    test = tmp_path / "ptb.test.txt"
    train.write_text("the cat sat\nthe dog sat\n", encoding="utf-8")
    validation.write_text("the fox sat\n", encoding="utf-8")
    test.write_text("the cat ran\n", encoding="utf-8")

    corpus = penn_treebank(train, validation, test, batch_size=2, sequence_length=2)

    assert corpus.vocabulary["<unk>"] == 0
    assert "<eos>" in corpus.vocabulary
    inputs, targets = next(iter(corpus.train))
    assert inputs.shape == targets.shape == (2, 2)
    stream = corpus.train.dataset.tokens
    assert torch.equal(inputs, stream[:, :2])
    assert torch.equal(targets, stream[:, 1:3])

    assert corpus.vocabulary["<unk>"] in corpus.validation.dataset.tokens


def test_penn_treebank_caps_vocabulary_deterministically(tmp_path):
    files = []
    for name in ("train", "valid", "test"):
        path = tmp_path / name
        path.write_text("z b a b a z c\n", encoding="utf-8")
        files.append(path)

    corpus = penn_treebank(*files, batch_size=1, max_vocab_size=4)

    assert list(corpus.vocabulary) == ["<unk>", "<eos>", "a", "b"]


def test_penn_treebank_rejects_too_short_splits(tmp_path):
    files = []
    for name in ("train", "valid", "test"):
        path = tmp_path / name
        path.write_text("token\n", encoding="utf-8")
        files.append(path)

    with pytest.raises(ValueError, match="more tokens"):
        penn_treebank(*files, batch_size=2)


def test_timit_pads_variable_length_utterances():
    features = [torch.ones(3, 120), torch.full((5, 120), 2.0)]
    targets = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])]

    batch = next(iter(timit(features, targets, batch_size=2, shuffle=False)))

    assert batch.features.shape == (2, 5, 120)
    assert batch.targets.shape == (2, 5)
    assert torch.equal(batch.lengths, torch.tensor([3, 5]))
    assert torch.all(batch.features[0, 3:] == 0)
    assert torch.all(batch.targets[0, 3:] == -100)
    assert torch.equal(batch.targets[1], targets[1])


def test_timit_supports_alternative_feature_and_class_counts():
    loader = timit(
        [torch.ones(2, 4)],
        [torch.tensor([0, 2])],
        feature_size=4,
        num_classes=3,
        batch_size=1,
    )

    batch = next(iter(loader))
    assert batch.features.shape == (1, 2, 4)


@pytest.mark.parametrize(
    "features,targets,error",
    [
        ([], [], ValueError),
        ([torch.ones(2, 119)], [torch.zeros(2)], ValueError),
        ([torch.ones(2, 120)], [torch.zeros(3)], ValueError),
        ([torch.ones(0, 120)], [torch.zeros(0)], ValueError),
        ([torch.ones(2, 120)], [torch.tensor([0, 180])], ValueError),
        ([torch.ones(2, 120, dtype=torch.long)], [torch.zeros(2)], TypeError),
    ],
)
def test_timit_validates_aligned_sequences(features, targets, error):
    with pytest.raises(error):
        timit(features, targets)
