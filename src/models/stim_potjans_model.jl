
for layer in layers
Currents[pop[r]] = 0.3512*pA*bg_layer[r]

# Layer-specific background input
bg_layer_specific = array([1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100])

# Layer-independent background input
bg_layer_independent = array([2000, 1850 ,2000, 1850, 2000, 1850, 2000, 1850])