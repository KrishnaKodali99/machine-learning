import numpy as np

m_phi = np.random.rand(40, 10)
v_psi = np.random.rand(40)

theta = np.dot(np.linalg.pinv(m_phi), v_psi)
loss = np.linalg.norm(np.dot(m_phi, theta) - v_psi)

print("∥Φ(θ⋆+ δ)-ψ∥2 > ∥Φθ⋆-ψ∥2")
for i in range(10):
    v_theta = np.random.rand(10)
    loss_theta = np.linalg.norm(np.dot(m_phi, theta + v_theta) - v_psi)
    print("Inequality holds" if loss_theta - loss > 0 else "Inequality doesn't holds")
