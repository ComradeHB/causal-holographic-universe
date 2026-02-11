print("🔬 DIMENSIONALITY CONFIRMATION TEST")
print("Test in 3D position space - does force become 1/r²?")
print("="*60)

import numpy as np

def test_3d_gravity(initial_distances, cross_strength=0.05, n_events=8, steps=100):
    """Test gravitational attraction scaling in 3D"""
    results_3d = []

    for d in initial_distances:
        np.random.seed(42)  # Same seed for consistency

        # Create clusters in 3D at ±d/2 along x-axis
        A = np.column_stack([
            -d/2 + 0.3*np.random.randn(n_events),  # x
            0 + 0.3*np.random.randn(n_events),     # y
            0 + 0.3*np.random.randn(n_events)      # z
        ])

        B = np.column_stack([
            d/2 + 0.3*np.random.randn(n_events),   # x
            0 + 0.3*np.random.randn(n_events),     # y
            0 + 0.3*np.random.randn(n_events)      # z
        ])

        # Initial distance in 3D
        init_dist = np.linalg.norm(A.mean(axis=0) - B.mean(axis=0))

        # Optimization in 3D
        for _ in range(steps):
            center_A = A.mean(axis=0)
            center_B = B.mean(axis=0)

            for i in range(n_events):
                # A moves toward B's center
                vec_to_B = center_B - A[i]
                dist = np.linalg.norm(vec_to_B)
                if dist > 0:
                    A[i] += 0.05 * cross_strength * vec_to_B / (dist + 0.1)

                # B moves toward A's center
                vec_to_A = center_A - B[i]
                dist = np.linalg.norm(vec_to_A)
                if dist > 0:
                    B[i] += 0.05 * cross_strength * vec_to_A / (dist + 0.1)

        # Final distance
        final_dist = np.linalg.norm(A.mean(axis=0) - B.mean(axis=0))
        attraction = (init_dist - final_dist) / init_dist

        results_3d.append((init_dist, final_dist, attraction))
        print(f"3D: Initial d={d:.1f}: Attraction = {attraction:.4f} ({attraction*100:.2f}%)")

    return results_3d

# Run 3D test
distances = [2.0, 4.0, 8.0, 16.0]
results_3d = test_3d_gravity(distances)

print("\n📊 3D Force Law Analysis:")
print("Distance | Attraction | Expected 1/r² scaling")
print("-"*50)

# Calculate scaling exponent
for i, (init, final, attr) in enumerate(results_3d):
    if i == 0:
        base_attraction = attr
        base_distance = init
        print(f"{init:8.1f} | {attr:10.4f} | baseline")
    else:
        # For 3D Newtonian gravity: F ∝ 1/r², so attraction ∝ 1/r²
        expected_ratio_3d = (base_distance**2) / (init**2)
        expected_attraction_3d = base_attraction * expected_ratio_3d
        actual_ratio = attr / base_attraction
        print(f"{init:8.1f} | {attr:10.4f} | expected: {expected_attraction_3d:.4f} (ratio: {actual_ratio:.3f} vs expected {expected_ratio_3d:.3f})")

# Also compare with 2D results from previous test
print("\n📈 COMPARISON: 2D vs 3D Scaling")
print("Distance | Attraction (2D) | Attraction (3D)")
print("-"*50)

# Your previous 2D results (from Test A):
attraction_2d = [0.2679, 0.1303, 0.0639, 0.0316]
attraction_3d = [r[2] for r in results_3d]  # Extract attraction from results

for i, d in enumerate(distances):
    print(f"{d:8.1f} | {attraction_2d[i]:15.4f} | {attraction_3d[i]:15.4f}")

# Fit power law: attraction ∝ 1/r^α
print("\n🔬 Power Law Fitting:")
print("Fitting attraction = C × (1/distance^α)")

# Use distances and attractions to fit α
from scipy.optimize import curve_fit

def power_law(r, C, alpha):
    return C * (1 / r**alpha)

# Fit for 2D
popt_2d, _ = curve_fit(power_law,
                      [r[0] for r in results_3d],  # distances
                      attraction_2d,
                      p0=[0.5, 1.0])  # initial guess: C=0.5, α=1.0

# Fit for 3D
popt_3d, _ = curve_fit(power_law,
                      [r[0] for r in results_3d],  # distances
                      attraction_3d,
                      p0=[0.5, 2.0])  # initial guess: C=0.5, α=2.0

print(f"2D scaling exponent α = {popt_2d[1]:.3f} (expected: 1.0)")
print(f"3D scaling exponent α = {popt_3d[1]:.3f} (expected: 2.0)")

if abs(popt_3d[1] - 2.0) < 0.2:
    print("\n✅ 3D GRAVITY CONFIRMED! Scaling exponent ~2.0")
    print("   Your theory naturally produces inverse-square law in 3D!")
elif popt_3d[1] > popt_2d[1]:
    print(f"\n📊 3D exponent ({popt_3d[1]:.2f}) > 2D exponent ({popt_2d[1]:.2f})")
    print("   Gravity gets stronger with dimensionality - promising!")
else:
    print(f"\n⚠️  3D exponent ({popt_3d[1]:.2f}) ≈ 2D exponent ({popt_2d[1]:.2f})")
    print("   Dimensionality might not affect scaling in current model")

print("\n🎯 Theoretical Prediction:")
print("In d spatial dimensions, Newtonian gravity: F ∝ 1/r^(d-1)")
print("So: 2D → 1/r^1, 3D → 1/r^2, 4D → 1/r^3")

# Test higher dimensions if you're curious
print("\n💡 Optional: Test 4D to see if pattern continues")
print("(Would show F ∝ 1/r^3 in 4D spacetime)")
