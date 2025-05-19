import pytest
from models.baseline_model.example_dataclass import Example
from models.errors import InvalidDimensionException


@pytest.mark.parametrize(
    "tokens,labels,confidences",
    [
        pytest.param([], [1], [0.01], id="empty tokens"),
        pytest.param(["A"], [], [0.01], id="empty labels"),
        pytest.param(["A"], [1], [], id="empty confidences"),
    ],
)
def test_cannot_construct_empty_example(tokens, labels, confidences):
    with pytest.raises(ValueError) as exception:
        Example(tokens, labels, confidences)

    assert exception.type is ValueError


@pytest.mark.parametrize(
    "tokens,labels,confidences",
    [
        pytest.param(
            ["Yoda", "is", "a" "jedi", "."],
            [1, 0, 0, 3, 0, 999],
            [0.92, 0.99, 0.99, 0.77, 0.99],
            id="invalid dimension at tokens",
        ),
        pytest.param(
            ["Yoda", "is", "a" "jedi", ".", "<PAD>", "<PAD>"],
            [1, 0, 0, 3, 0],
            [0.92, 0.99, 0.99, 0.77, 0.99],
            id="invalid dimension at labels",
        ),
        pytest.param(
            ["Yoda", "is", "a" "jedi", "."],
            [1, 0, 0, 3, 0],
            [0.92],
            id="invalid dimension at confidences",
        ),
    ],
)
def test_cannot_construct_bad_dimensions_example(tokens, labels, confidences):
    with pytest.raises(InvalidDimensionException) as exception:
        Example(tokens, labels, confidences)

    assert "Dimension" in str(exception.value)


@pytest.mark.parametrize(
    "tokens,labels,confidences",
    [
        pytest.param(
            ["Yoda", "is", "a", "jedi", "."],
            [1, 0, 0, 3, 0],
            [0.93, 0.99, 0.99, 0.88, 0.99],
        ),
        pytest.param(
            ["Darth", "Vader", "is", "a", "Sith", "."],
            [1, 2, 0, 0, 3, 0],
            [0.89, 0.93, 0.99, 0.99, 0.90, 0.99],
        ),
    ],
)
def test_initialize_example(tokens, labels, confidences):
    Example(tokens, labels, confidences)


@pytest.mark.parametrize(
    "tokens,labels,confidences,expected_labels",
    [
        pytest.param(
            ["Yoda", "is", "a", "jedi", "."],
            [1, 0, 0, 3, 0],
            [0.88, 0.99, 0.99, 0.91, 0.99],
            [-100, 0, 0, 3, 0],
            id="mask uncertain NER entities",
        )
    ],
)
def test_example_mask(tokens, labels, confidences, expected_labels):
    example = Example(tokens, labels, confidences)

    example.mask()

    assert example.labels == expected_labels


@pytest.mark.parametrize(
    "tokens,labels,confidences,expected",
    [
        pytest.param(
            ["Yo", "##da", "is", "a", "je", "##di", "##goat", "."],
            [1, 1, 0, 0, 3, 3, 3, 0],
            [0.90, 0.91, 0.99, 0.99, 0.89, 0.91, 0.91, 0.99],
            [
                ["Yoda", "is", "a", "jedigoat", "."],
                [1, 0, 0, 3, 0],
                [0.90, 0.99, 0.99, 0.89, 0.99],
            ],
            id="merge subwords into whole words",
        )
    ],
)
def test_merge_subwords(tokens, labels, confidences, expected):
    expected_tokens, expected_labels, expected_confidences = (
        expected[0],
        expected[1],
        expected[2],
    )
    example = Example(tokens, labels, confidences)

    example.merge_subwords()

    assert example.tokens == expected_tokens
    assert example.labels == expected_labels
    assert example.confidences == expected_confidences

