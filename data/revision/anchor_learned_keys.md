# Non-synthetic anchor: learned-projection keys

Address codes are the rows of a token autoencoder embedding (m=256, dk=32, bottleneck forces a *learned*, non-Gaussian geometry), not Gaussian draws. We re-check the two structural claims.

| code | max residual=|1-g| err | locality slope L1 | L1/mu_j |
|---|---|---|---|
| Gaussian (synthetic) | 2.00e-15 | 2.1213 | 4.4247 |
| learned (autoencoder) | 3.55e-15 | 2.1046 | 4.1944 |

**Reading.** (1) The residual identity holds to machine precision under the learned code exactly as under Gaussian --- it is an algebraic property of the outer-product update, independent of how the code was drawn. (2) The locality slope L1 (hence the V-shape) is present under both codes; its magnitude shifts by the state-dependent factor mu_j * r_min(j) (Theorem lb), i.e. the *law* is code-agnostic even though the slope constant is not. This is the non-synthetic anchor R3 asked for; a full pretrained-LM-embedding anchor is deferred to the GPU run.