import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# The famous Collatz function - if even divide by 2, if odd multiply by 3 and add 1
def collatz(n):
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1

# Make training data with fancy features
def make_data(samples=10000, max_n=10000):
    """Make dataset with features that might help the AI understand Collatz"""
    
    nums = []
    next_nums = []
    
    # Generate random numbers and their Collatz steps
    for _ in range(samples):
        n = np.random.randint(2, max_n)
        f_n = collatz(n)
        nums.append(n)
        next_nums.append(f_n)
    
    nums = np.array(nums, dtype=np.float32)
    next_nums = np.array(next_nums, dtype=np.float32)
    
    # Turn numbers into features the AI can understand better
    def make_features(arr):
        features = []
        for val in arr:
            n = int(val)
            
            # Log helps with big vs small numbers
            log_val = np.log(val + 1e-8)
            
            # These modular features are key for Collatz pattern
            mod2 = n % 2  # even or odd
            mod4 = n % 4  # more detailed pattern
            mod8 = n % 8  # even more detailed
            
            # How many times can we divide by 2? (trailing zeros in binary)
            zeros = 0
            temp = n
            while temp % 2 == 0 and temp > 0:
                zeros += 1
                temp //= 2
            zeros = min(zeros, 10)  # cap it
            
            features.append([log_val, mod2, mod4, mod8, zeros])
        
        return np.array(features, dtype=np.float32)
    
    n_feat = make_features(nums)
    f_feat = make_features(next_nums)
    
    return n_feat, f_feat, nums, next_nums

# Build the neural network
def build_model(input_size=5):
    """Build a neural net to learn the Lyapunov function V(n)"""
    
    model = models.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),  # first hidden layer
        layers.Dropout(0.1),  # prevent overfitting
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),  # second hidden layer  
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),  # smaller layer
        layers.Dense(1, activation='softplus', name='V_output')  # output V(n)
    ])
    
    return model

# Turn a single number into features
def num_to_features(n):
    """Convert one number to feature vector"""
    n_float = float(n)
    n_int = int(n)
    
    log_val = np.log(n_float + 1e-8)
    mod2 = n_int % 2
    mod4 = n_int % 4
    mod8 = n_int % 8
    
    # Count trailing zeros
    zeros = 0
    temp = n_int
    while temp % 2 == 0 and temp > 0:
        zeros += 1
        temp //= 2
    zeros = min(zeros, 10)
    
    return np.array([log_val, mod2, mod4, mod8, zeros], dtype=np.float32)

# Train the AI to learn V(n) such that V(collatz(n)) < V(n)
def train_model(model, n_feat, f_feat, epochs=100, margin=0.1):
    """Train the model to satisfy V(f(n)) < V(n) - margin"""
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Convert to tensors once
    n_tensor = tf.convert_to_tensor(n_feat, dtype=tf.float32)
    f_tensor = tf.convert_to_tensor(f_feat, dtype=tf.float32)
    
    # For gradient penalty (use small subset for speed)
    n_var = tf.Variable(n_tensor[:100], trainable=False)
    
    loss_hist = []
    sat_hist = []
    
    batch_size = 256
    n_batches = int(np.ceil(len(n_feat) / batch_size))
    
    print("Starting training...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_sat = 0
        
        # Shuffle data each epoch
        idx = np.random.permutation(len(n_feat))
        n_shuffled = n_feat[idx]
        f_shuffled = f_feat[idx]
        
        for batch in range(n_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, len(n_feat))
            
            batch_n = tf.convert_to_tensor(n_shuffled[start:end], dtype=tf.float32)
            batch_f = tf.convert_to_tensor(f_shuffled[start:end], dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                # Get V(n) and V(f(n))
                V_n = model(batch_n, training=True)
                V_fn = model(batch_f, training=True)
                
                # We want V(f(n)) < V(n) - margin
                # Penalize when this is violated
                violation = tf.maximum(V_fn - V_n + margin, 0)
                constraint_loss = tf.reduce_mean(violation)
                
                # Don't let V(n) be constant (boring solution)
                variance_penalty = 0.01 * tf.math.reduce_variance(V_n)
                
                # Keep V(n) smooth
                with tf.GradientTape() as grad_tape:
                    grad_tape.watch(n_var)
                    V_subset = model(n_var, training=True)
                grads = grad_tape.gradient(V_subset, n_var)
                grad_norm = tf.reduce_mean(tf.square(grads))
                smooth_penalty = 0.001 * grad_norm
                
                total_loss = constraint_loss + variance_penalty + smooth_penalty
                
                # How often do we satisfy the constraint?
                satisfaction = tf.reduce_mean(tf.cast(V_fn < V_n, tf.float32))
            
            # Update model
            grads = tape.gradient(total_loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            
            epoch_loss += total_loss.numpy()
            epoch_sat += satisfaction.numpy()
        
        # Average over batches
        avg_loss = epoch_loss / n_batches
        avg_sat = epoch_sat / n_batches
        
        loss_hist.append(avg_loss)
        sat_hist.append(avg_sat)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Satisfaction = {avg_sat:.3f}")
    
    return loss_hist, sat_hist

# Test how well our V(n) works
def test_model(model, test_max=1000):
    """Test if V(f(n)) < V(n) on a bunch of numbers"""
    
    test_nums = np.arange(2, test_max, dtype=np.float32)
    
    # Get features for test numbers
    test_feat = []
    for n in test_nums:
        test_feat.append(num_to_features(n))
    test_feat = np.array(test_feat, dtype=np.float32)
    
    # Get V(n)
    V_n = model.predict(test_feat, verbose=0)
    
    # Get V(f(n))
    f_nums = np.array([collatz(int(n)) for n in test_nums], dtype=np.float32)
    f_feat = []
    for n in f_nums:
        f_feat.append(num_to_features(n))
    f_feat = np.array(f_feat, dtype=np.float32)
    
    V_fn = model.predict(f_feat, verbose=0)
    
    # Check how often V(f(n)) < V(n)
    diffs = V_fn.flatten() - V_n.flatten()
    sat_rate = np.mean(diffs < 0)
    
    good_diffs = diffs[diffs < 0]
    bad_diffs = diffs[diffs >= 0]
    
    avg_decrease = np.mean(-good_diffs) if len(good_diffs) > 0 else 0
    avg_increase = np.mean(bad_diffs) if len(bad_diffs) > 0 else 0
    
    print(f"\nTesting on numbers 2 to {test_max-1}:")
    print(f"Success rate (V(f(n)) < V(n)): {sat_rate:.1%}")
    print(f"Average decrease when it works: {avg_decrease:.4f}")
    print(f"Average increase when it fails: {avg_increase:.4f}")
    print(f"Overall average change: {np.mean(diffs):.4f}")
    
    # Test full trajectories
    print("\nTesting full Collatz sequences:")
    test_starts = [6, 7, 15, 27, 54, 97]
    
    for start in test_starts:
        seq = [start]
        curr = start
        
        # Follow Collatz sequence
        for _ in range(20):
            curr = collatz(curr)
            seq.append(curr)
            if curr == 1:
                break
        
        # Get V values along sequence
        V_vals = []
        for val in seq:
            feat = num_to_features(val)
            V_val = model.predict(feat.reshape(1, -1), verbose=0)[0, 0]
            V_vals.append(V_val)
        
        # Check if V decreases along sequence
        decreasing_steps = sum(1 for i in range(len(V_vals)-1) if V_vals[i] > V_vals[i+1])
        total_steps = len(V_vals) - 1
        is_monotonic = all(V_vals[i] > V_vals[i+1] for i in range(len(V_vals)-1))
        
        print(f"Start={start:3d}: length={len(seq):2d}, "
              f"decreasing={decreasing_steps}/{total_steps}, "
              f"monotonic={is_monotonic}, final_V={V_vals[-1]:.3f}")
    
    return diffs, sat_rate

# Check if Collatz steps look like gradient descent
def check_gradient_descent(model):
    """See if Collatz steps resemble gradient descent on V(n)"""
    
    print("\n" + "="*50)
    print("GRADIENT DESCENT ANALYSIS")
    print("="*50)
    
    test_nums = [3, 7, 15, 31, 63, 127]
    
    for n in test_nums:
        # Estimate gradient using finite differences
        eps = 0.01
        n_plus = num_to_features(n + eps)
        n_minus = num_to_features(n - eps)
        
        V_plus = model.predict(n_plus.reshape(1, -1), verbose=0)[0, 0]
        V_minus = model.predict(n_minus.reshape(1, -1), verbose=0)[0, 0]
        
        grad_V = (V_plus - V_minus) / (2 * eps)
        
        # What does Collatz give us?
        collatz_next = collatz(n)
        
        # What would gradient descent give us? Try different step sizes
        best_lr = None
        best_diff = float('inf')
        
        for lr in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
            gd_step = n - lr * grad_V
            diff = abs(gd_step - collatz_next)
            if diff < best_diff:
                best_diff = diff
                best_lr = lr
        
        gd_next = n - best_lr * grad_V
        
        # Compare V values
        feat_n = num_to_features(n)
        feat_collatz = num_to_features(collatz_next)
        feat_gd = num_to_features(gd_next)
        
        V_n = model.predict(feat_n.reshape(1, -1), verbose=0)[0, 0]
        V_collatz = model.predict(feat_collatz.reshape(1, -1), verbose=0)[0, 0]
        V_gd = model.predict(feat_gd.reshape(1, -1), verbose=0)[0, 0]
        
        print(f"n={n:4d}: grad={grad_V:8.3f}, best_lr={best_lr:.2f}, "
              f"GD_next={gd_next:8.1f}, Collatz_next={collatz_next:4d}, "
              f"diff={abs(gd_next - collatz_next):6.1f}, "
              f"V_n={V_n:7.3f}, ΔV_collatz={V_collatz - V_n:7.3f}")
              
    # Compare efficiency of descent
    print("\n" + "="*50)
    print("EFFICIENCY RACE: COLLATZ vs GRADIENT DESCENT")
    print("="*50)
    
    start_n = 27
    print(f"Racing from n={start_n} to 1...")
    
    # 1. Collatz
    curr = start_n
    c_steps = 0
    while curr != 1 and c_steps < 200:
        curr = collatz(curr)
        c_steps += 1
    print(f"Collatz Algorithm: {c_steps} steps")
    
    # 2. Gradient Descent on V(n)
    curr_n = float(start_n)
    gd_steps = 0
    lr = 20.0  # Needs high learning rate to traverse integers
    
    for _ in range(1000):
        # Calc gradient
        eps = 0.01
        n_p = num_to_features(curr_n + eps)
        n_m = num_to_features(curr_n - eps)
        V_p = model.predict(n_p.reshape(1, -1), verbose=0)[0, 0]
        V_m = model.predict(n_m.reshape(1, -1), verbose=0)[0, 0]
        grad = (V_p - V_m) / (2 * eps)
        
        curr_n = curr_n - lr * grad
        gd_steps += 1
        if curr_n <= 1.1:
            break
            
    print(f"Gradient Descent:  {gd_steps} steps (Final n={curr_n:.2f})")
    print("Note: GD struggles because modulo features have zero gradient!")

# Main function - run everything
def main():
    print("🤖 Let's teach an AI about the Collatz conjecture! 🤖")
    print("We'll try to find a function V(n) that always decreases...")
    
    print("\nMaking training data...")
    n_feat, f_feat, n_vals, f_vals = make_data(samples=20000, max_n=10000)
    print(f"Got {len(n_feat)} training examples with {n_feat.shape[1]} features each")
    
    print("\nBuilding the neural network...")
    model = build_model(input_size=5)
    model.summary()
    
    print("\nTraining the model to learn V(n)...")
    losses, satisfaction = train_model(model, n_feat, f_feat, epochs=100, margin=0.05)
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(satisfaction)
    plt.title('How Often V(f(n)) < V(n)')
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    plt.show()
    
    # Test the model
    print("\n" + "="*50)
    print("TESTING THE MODEL")
    print("="*50)
    
    diffs, sat_rate = test_model(model, test_max=2000)
    
    # Test on bigger numbers
    print("\n" + "="*50)
    print("TESTING ON BIG NUMBERS")
    print("="*50)
    
    big_nums = np.random.randint(10000, 100000, size=100).astype(np.float32)
    
    big_feat = []
    for n in big_nums:
        big_feat.append(num_to_features(n))
    big_feat = np.array(big_feat)
    
    V_n_big = model.predict(big_feat, verbose=0)
    
    f_n_big = np.array([collatz(int(n)) for n in big_nums], dtype=np.float32)
    f_feat_big = []
    for n in f_n_big:
        f_feat_big.append(num_to_features(n))
    f_feat_big = np.array(f_feat_big)
    
    V_fn_big = model.predict(f_feat_big, verbose=0)
    
    big_diffs = V_fn_big.flatten() - V_n_big.flatten()
    big_sat = np.mean(big_diffs < 0)
    
    print(f"Success rate on numbers 10,000-100,000: {big_sat:.1%}")
    print(f"Average change in V: {np.mean(big_diffs):.4f}")
    
    # Gradient analysis
    check_gradient_descent(model)
    
    # Visualize V(n)
    print("\n" + "="*50)
    print("VISUALIZING V(n)")
    print("="*50)
    
    vis_nums = np.arange(1, 200, dtype=np.float32)
    vis_feat = np.array([num_to_features(n) for n in vis_nums])
    V_vis = model.predict(vis_feat, verbose=0)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(vis_nums, V_vis, alpha=0.6, s=20)
    plt.xlabel('n')
    plt.ylabel('V(n)')
    plt.title('The Learned Function V(n)')
    plt.grid(True, alpha=0.3)
    plt.savefig('V_function.png', dpi=150)
    plt.show()
    
    # V(n) vs log(n)
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log(vis_nums + 1e-8), V_vis, alpha=0.6, s=20)
    plt.xlabel('log(n)')
    plt.ylabel('V(n)')
    plt.title('V(n) vs log(n) - Is There a Pattern?')
    plt.grid(True, alpha=0.3)
    plt.savefig('V_vs_log.png', dpi=150)
    plt.show()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("We trained an AI to find a function V(n) that decreases along Collatz sequences!")
    print(f"Final success rate: {sat_rate:.1%}")
    print("\nWhat we learned:")
    print("✓ The AI found a V(n) that usually decreases with Collatz steps")
    print("✓ This suggests Collatz sequences are 'going downhill' in some sense")
    print("✗ But it's not perfect - some steps still go 'uphill'")
    print("\nFor the Collatz conjecture:")
    print("- If we could find a PERFECT V(n) that ALWAYS decreases,")
    print("  and V(n) has a lower bound, that would prove the conjecture!")
    print("- The challenge is making it work for ALL numbers, not just most")

if __name__ == "__main__":
    main()