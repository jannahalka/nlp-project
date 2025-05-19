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


def test_prepare_for_testing_example():
    pass
