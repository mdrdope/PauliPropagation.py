def _random_su4(seed=None):
    """Return a random 4¡Á4 special-unitary matrix (det = 1)."""
    if seed is not None:
        np.random.seed(seed)
    z = (np.random.randn(4, 4) + 1j*np.random.randn(4, 4)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q *= np.diag(r) / np.abs(np.diag(r))           # fix phases
    q /= np.linalg.det(q) ** (1/4)                 # det ¡ú 1
    return q                                       # ¡Ê SU(4)

def _staircase_edges(nx, ny):
    """Return the ordered list of 1-based index pairs of the staircase walk."""
    next_inds, temp_inds, edges = [1], [], []
    while next_inds:
        for ind in next_inds:
            if ind % nx != 0:                      # step right
                nxt = ind + 1
                edges.append((ind, nxt)); temp_inds.append(nxt)
            if ((ind-1)//nx + 1) < ny:             # step down
                nxt = ind + nx
                edges.append((ind, nxt)); temp_inds.append(nxt)
        next_inds, temp_inds = temp_inds, []
    seen, uniq = set(), []
    for e in edges:                                # preserve order, dedup
        if e not in seen:
            seen.add(e); uniq.append(e)
    return uniq

def staircasetopology2d_qc(nx, ny, seed=None):
    """
    Build a QuantumCircuit that places a fresh SU(4) gate on every edge
    of the 2-D staircase topology for an nx ¡Á ny grid (row-major indexing).
    """
    nqubits = nx * ny
    qc = QuantumCircuit(nqubits)
    for k, (q1, q2) in enumerate(_staircase_edges(nx, ny)):  # 1-based
        mat = _random_su4(seed=seed + k if seed is not None else None)
        qc.append(UnitaryGate(mat, label=f"SU4_{k}"), [q1-1, q2-1])  # 0-based
    return qc

# Example usage:
nx, ny = 3, 3
qc_2d = staircasetopology2d_qc(nx, ny)

fig = qc_2d.draw(output="mpl", fold=-1)   # no line-wrapping
plt.show()