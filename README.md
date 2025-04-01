# Distribution System State Estimation using JAX

With the growing integration of active control devices in distribution systems, Distribution System State Estimation (DSSE) is increasingly critical for ensuring effective operational control and preventing outages. This repo presents a novel DSSE approach that combines OpenDSS as a power flow solver with Python for state estimation, utilizing smart meter data accessed via the py-dss-interface. A significant contribution of this work is the application of JAX (Just After eXecution), a machine learning framework, to accelerate the gradient descent solution of the Weighted Least Squares (WLS) objective function, removing the need for explicitly computing the Jacobian matrix. The proposed method is validated on unbalanced IEEE 13, 37, and 123 radial bus distribution systems, showing substantial improvements in computational efficiency and accuracy.

Please go through the paper for detailed understanding: [DSSE using JAX](https://doi.org/10.1109/TPEC63981.2025.10906977)

