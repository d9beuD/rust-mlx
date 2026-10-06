// Preserve the separate native BF16 multiply/add boundaries; no reassociation.
uint i = thread_position_in_grid.x;
if (i >= ROWS * HC * HIDDEN) return;
uint row = i / (HC * HIDDEN);
uint stream = (i / HIDDEN) % HC;
uint col = i % HIDDEN;
uint branch = row * HIDDEN + col;
T weighted = T(float(shared[branch]) * float(factor[row]));
T combined = T(float(routed[branch]) + float(weighted));
T injected = T(float(combined) * float(gate[row * HC + stream]));
out[i] = T(float(residual[i]) + float(injected));
