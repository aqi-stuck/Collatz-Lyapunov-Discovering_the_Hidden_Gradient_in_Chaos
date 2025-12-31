import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# Basic Collatz function for arrays
def collatz(n):
    n = n.astype(np.int64)
    even = n % 2 == 0
    result = np.empty_like(n)
    result[even] = n[even] // 2  # if even, divide by 2
    result[~even] = 3 * n[~even] + 1  # if odd, 3n+1
    return result


# Do k Collatz steps
def collatz_steps(n, k=1):
    out = n.copy()
    for _ in range(k):
        out = collatz(out)
    return out


# Count how many times we can divide by 2 (trailing zeros in binary)
def count_trailing_zeros(n):
    n = n.copy().astype(np.int64)
    zeros = np.zeros_like(n, dtype=np.int32)
    can_divide = (n % 2 == 0) & (n > 0)

    while can_divide.any():
        zeros[can_divide] += 1
        n[can_divide] //= 2
        can_divide = (n % 2 == 0) & (n > 0)

    zeros = np.minimum(zeros, 30)  # cap it
    return zeros


# Turn numbers into features the AI can understand
def make_features(nums, max_n):
    nums = nums.astype(np.float64)

    # Normalize log to 0-1 range
    log_n = np.log(nums + 1.0) / np.log(max_n + 1.0)

    # Even/odd and modular patterns
    even_odd = (nums % 2).astype(np.float32)
    mod4 = (nums % 4).astype(np.float32) / 4.0
    mod8 = (nums % 8).astype(np.float32) / 8.0

    # Trailing zeros feature
    tz = count_trailing_zeros(nums).astype(np.float32) / 10.0

    # Stack all features together
    features = np.stack([log_n.astype(np.float32), even_odd, mod4, mod8, tz], axis=1)
    return features.astype(np.float32)


# Make training data
def make_data(min_n, max_n, k_steps=2, samples=None):
    if samples is None:
        nums = np.arange(min_n, max_n + 1, dtype=np.int64)
    else:
        nums = np.random.randint(min_n, max_n + 1, size=samples, dtype=np.int64)

    # Get 1-step and k-step Collatz results
    next1 = collatz_steps(nums, k=1)
    nextk = collatz_steps(nums, k=k_steps)

    # Convert to features
    X_n = make_features(nums, max_n=max_n)
    X_next1 = make_features(next1, max_n=max_n)
    X_nextk = make_features(nextk, max_n=max_n)

    return nums, X_n, X_next1, X_nextk


# Build simple neural network
def build_net(input_size=5, hidden=64):
    model = models.Sequential(
        [
            layers.Input(shape=(input_size,), dtype=tf.float32),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(1, activation="linear"),  # output V(n)
        ]
    )
    return model


# Training settings
EVEN_MARGIN = 0.25  # bigger margin for even numbers
ODD_MARGIN = 0.03  # smaller margin for odd numbers
MULTI_MARGIN = 0.05  # margin for multi-step
VAR_WEIGHT = 0.01  # keep V(n) from being constant
GRAD_WEIGHT = 1e-4  # smoothness penalty
LR = 1e-3  # learning rate


# One training step
def train_step(model, opt, X_n, X_next1, X_nextk, max_n):
    # Convert to tensors
    Xn = tf.convert_to_tensor(X_n, dtype=tf.float32)
    Xf1 = tf.convert_to_tensor(X_next1, dtype=tf.float32)
    Xfk = tf.convert_to_tensor(X_nextk, dtype=tf.float32)

    # First tape: get input gradients for smoothness
    with tf.GradientTape() as tape1:
        tape1.watch(Xn)
        Vn_grad = model(Xn, training=True)

    input_grads = tape1.gradient(Vn_grad, Xn)

    if input_grads is None:
        grad_loss = tf.constant(0.0, dtype=tf.float32)
    else:
        grad_loss = tf.reduce_mean(tf.square(input_grads))
        grad_loss = tf.clip_by_value(grad_loss, 0.0, 10.0)  # cap it

    # Second tape: main loss and parameter gradients
    with tf.GradientTape() as tape2:
        Vn = model(Xn, training=True)
        Vf1 = model(Xf1, training=True)
        Vfk = model(Xfk, training=True)

        # Different margins for even/odd numbers
        even_odd = tf.expand_dims(Xn[:, 1], -1)  # parity feature
        margin = tf.where(tf.equal(even_odd, 0.0), EVEN_MARGIN, ODD_MARGIN)

        # Loss: we want V(f(n)) < V(n) - margin
        loss1 = tf.reduce_mean(tf.maximum(Vf1 - Vn + margin, 0.0))
        loss_multi = tf.reduce_mean(tf.maximum(Vfk - Vn + MULTI_MARGIN, 0.0))

        # Don't let V(n) be constant
        var_loss = tf.math.reduce_variance(Vn)

        total_loss = (
            loss1 + 0.5 * loss_multi + VAR_WEIGHT * var_loss + GRAD_WEIGHT * grad_loss
        )

    # Update model
    grads = tape2.gradient(total_loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)  # clip gradients
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return (
        float(total_loss.numpy()),
        float(loss1.numpy()),
        float(loss_multi.numpy()),
        float(grad_loss.numpy()),
    )


# Train the model
def train(min_n=2, max_n=20000, epochs=40, batch_size=512, k_steps=3, samples=20000):
    print(f"Making training data with {samples} samples...")
    nums, Xn, Xf1, Xfk = make_data(min_n, max_n, k_steps=k_steps, samples=samples)

    print("Building neural network...")
    model = build_net(input_size=Xn.shape[1], hidden=128)
    opt = optimizers.Adam(learning_rate=LR)

    n_samples = Xn.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Shuffle data
        perm = np.random.permutation(n_samples)
        Xn_shuffled = Xn[perm]
        Xf1_shuffled = Xf1[perm]
        Xfk_shuffled = Xfk[perm]

        epoch_loss = 0.0

        for batch in range(n_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, n_samples)

            batch_Xn = Xn_shuffled[start:end]
            batch_Xf1 = Xf1_shuffled[start:end]
            batch_Xfk = Xfk_shuffled[start:end]

            total_loss, l1, lm, gl = train_step(
                model, opt, batch_Xn, batch_Xf1, batch_Xfk, max_n
            )
            epoch_loss += total_loss

        epoch_loss /= n_batches

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d}: loss={epoch_loss:.6f} | single={l1:.6f} | multi={lm:.6f} | smooth={gl:.6f}"
            )

    return model


# Test how well the model works
def test_model(model, min_n, max_n):
    nums = np.arange(min_n, max_n + 1, dtype=np.int64)
    next_nums = collatz_steps(nums, k=1)

    Xn = make_features(nums, max_n=max_n)
    Xf1 = make_features(next_nums, max_n=max_n)

    Vn = model(Xn).numpy().flatten()
    Vf1 = model(Xf1).numpy().flatten()

    diffs = Vf1 - Vn
    success_rate = float(np.mean(diffs < 0))

    good_diffs = diffs[diffs < 0]
    bad_diffs = diffs[diffs >= 0]

    avg_decrease = float(np.mean(-good_diffs) if len(good_diffs) > 0 else 0.0)
    avg_increase = float(np.mean(bad_diffs) if len(bad_diffs) > 0 else 0.0)
    overall_avg = float(np.mean(diffs))

    return {
        "success_rate": success_rate,
        "avg_decrease": avg_decrease,
        "avg_increase": avg_increase,
        "overall_avg": overall_avg,
    }


# Check if Collatz steps look like gradient descent
def gradient_step(model, n, lr=0.01, max_n=20000):
    x = make_features(np.array([n]), max_n=max_n)
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        v = model(x_tensor)

    grad = tape.gradient(v, x_tensor).numpy()[0, 0]
    return n - lr * grad


# Run everything
if __name__ == "__main__":
    print("Training AI to understand Collatz conjecture...")

    model = train(
        min_n=2, max_n=20000, epochs=40, batch_size=512, k_steps=3, samples=20000
    )

    print("\n Testing on small numbers (2-2000):")
    result1 = test_model(model, 2, 2000)
    print(f"Success rate: {result1['success_rate']:.1%}")
    print(f"Avg decrease when working: {result1['avg_decrease']:.4f}")
    print(f"Avg increase when failing: {result1['avg_increase']:.4f}")

    print("\n Testing on big numbers (10k-50k):")
    result2 = test_model(model, 10000, 50000)
    print(f"Success rate: {result2['success_rate']:.1%}")
    print(f"Overall average change: {result2['overall_avg']:.4f}")

    print("\n Gradient descent vs Collatz comparison:")
    test_nums = [3, 7, 15, 27, 31, 63, 127]

    for n in test_nums:
        gd_step = gradient_step(model, n, lr=0.01, max_n=20000)
        collatz_step = collatz_steps(np.array([n]), k=1)[0]
        print(
            f"n={n:4d} | Gradient descent ≈ {gd_step:7.2f} | Collatz = {collatz_step}"
        )
