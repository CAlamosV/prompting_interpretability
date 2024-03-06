import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def construct_polynomial(
    degree,
    perturbation=0.1,
    root_offsets=None,
):
    """
    Constructs and scales a polynomial with roots adjusted relative to the positions of Chebyshev polynomial roots.
    Raises an error if the shift is too large, causing roots to go outside [-1, 1] or cross over adjacent roots.

    Parameters:
    - degree (int): Degree of the polynomial.
    - perturbation (float): Maximum perturbation applied if offsets are not specified.
    - root_offsets (np.ndarray, optional): Array of offsets to apply to the Chebyshev roots.

    Returns:
    - roots (np.ndarray): Array of roots of the polynomial.
    - scale (float): Scaling factor for the polynomial.
    """
    k = np.arange(1, degree + 1)
    chebyshev_roots = np.cos((2 * k - 1) * np.pi / (2 * degree))
    chebyshev_roots = np.sort(chebyshev_roots)

    if root_offsets is not None and len(root_offsets) > 0:
        if len(root_offsets) > degree:
            raise ValueError(
                "Number of root offsets cannot exceed the degree of the polynomial."
            )

        adjusted_roots = chebyshev_roots[: len(root_offsets)] + root_offsets

        if np.any(adjusted_roots < -1) or np.any(adjusted_roots > 1):
            raise ValueError("Root offset causes root to go outside the [-1, 1] range.")

        for i in range(1, len(adjusted_roots)):
            root, prev_root = adjusted_roots[i], adjusted_roots[i - 1]
            if root < prev_root:
                root, prev_root = np.round(root, 5), np.round(prev_root, 5)
                raise ValueError(
                    f"Root offset causes root {i} : {root} to cross over root {i - 1} : {prev_root}."
                )

        if len(root_offsets) < degree:
            additional_roots = chebyshev_roots[len(root_offsets) :] + np.random.uniform(
                -perturbation, perturbation, degree - len(root_offsets)
            )
            additional_roots[-1] = min(additional_roots[-1], 1)
            assert np.all(additional_roots >= -1) and np.all(
                additional_roots <= 1
            ), "Perturbation causes root to go outside the [-1, 1] range."
            roots_to_use = np.concatenate((adjusted_roots, additional_roots))
        else:
            roots_to_use = adjusted_roots
    else:
        roots_to_use = chebyshev_roots + np.random.uniform(
            -perturbation, perturbation, degree
        )
        roots_to_use[0] = max(roots_to_use[0], -1)
        roots_to_use[-1] = min(roots_to_use[-1], 1)
        assert np.all(roots_to_use >= -1) and np.all(
            roots_to_use <= 1
        ), "Perturbation causes root to go outside the [-1, 1] range."

    # Scale the polynomial to have a maximum absolute value of 1.
    xs = np.linspace(-1, 1, 100)
    ys = np.polyval(np.poly(roots_to_use), xs)
    scale = np.random.choice([1, -1]) / np.max(np.abs(ys))

    return roots_to_use, scale

def generate_synth_data(
    degree,
    length=100,
    samples=1000,
    perturbation=0.1,
    root_offsets=None,
):
    """
    Generate interleaved sequence data for training a GPT model, where the model predicts
    the next value in a sequence of [x1, f(x1), x2, f(x2), ..., xn] as [f(x1), x2, f(x2), ..., f(xn)],
    for a given number of samples, each with a specified sequence length.

    Parameters:
    - degree (int): Degree of the polynomial used for data generation.
    - length (int): Number of x values (and corresponding f(x) values) per sample.
    - samples (int): Total number of samples to generate.
    - perturbation (float): Perturbation applied to polynomial roots.
    - root_offsets (list or np.ndarray, optional): Specific offsets to apply to Chebyshev roots.

    Returns:
    - torch.Tensor: Input sequences tensor of shape (samples, 2*length-1).
    - torch.Tensor: Target sequences tensor of shape (samples, 2*length-1).
    """
    assert length % 2 == 0, "Length must be an even number."

    inputs_list = []
    targets_list = []

    for _ in range(samples):
        roots, scale = construct_polynomial(degree, perturbation, root_offsets)

        # uniform random sample from [-1, 1]
        x_values = np.random.uniform(-1, 1, (length // 2) + 1)
        y_values = np.polyval(np.poly(roots), x_values) * scale

        # shuffle x, y pairs
        indices = np.random.permutation((length // 2) + 1)
        x_values = x_values[indices]
        y_values = y_values[indices]

        interleaved = np.empty(length + 1)
        interleaved[0::2] = x_values  # x values at even indices
        interleaved[1::2] = y_values[
            :-1
        ]  # f(x) values at odd indices, except for the last f(x)xs

        inputs = interleaved[:-1]  # All except the last value
        targets = interleaved[1:]  # All except the first value

        inputs_list.append(inputs)
        targets_list.append(targets)

    # Convert lists to tensors with shape (samples, 2*length-1)
    inputs_tensor = torch.tensor(inputs_list, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_list, dtype=torch.float32)

    return inputs_tensor, targets_tensor

def plot_polynomial(roots, scale, precision=100):
    """
    Plots the polynomial with the specified roots.
    Parameters:
    - roots (np.ndarray): Array of roots of the polynomial.
    - scale (float): Scaling factor for the polynomial.
    - precision (int): Number of points to use for plotting the polynomial.
    Returns:
    - None
    """
    x = np.linspace(-1, 1, precision)
    y = np.polyval(np.poly(roots), x) * scale

    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x, y=y)
    if roots is not None:
        plt.scatter(roots, np.zeros_like(roots), marker="o", color="black")
        for i, root in enumerate(roots):
            plt.text(root, 0.1, f"{root:.2f}", fontsize=12, ha="center")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random Polynomial of Degree 5")
    plt.ylim(-1, 1)
    plt.show()

# Example usage
# degree = 5
# root_offsets = np.array([0, 0.1])
# roots, scale = construct_polynomial(degree, perturbation=0.2, root_offsets=root_offsets)
# plot_polynomial(roots, scale)

# degree = 5
# length = 20
# samples = 100
# inputs, targets = generate_synth_data(degree, length, samples, perturbation=0.08)