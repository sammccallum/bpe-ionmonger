# Bayesian Parameter Estimation (BPE) for characterising perovskite solar cells

A BPE method utilising the perovskite solar cell device model, IonMonger, to derive posterior distributions over physical perovskite parameters associated with a J-V measurement of a perovskite device.

## Example use
There are only four parameters that require input by the user: \n
`n_iter` - number of iterations in each Markov chain \n
`n_chains` - number of Markov chains \n
`prior_ranges` - numpy array containing prior range of each device parameter \n
`y` - experimental J-V characteristics at each scan-rate (0.1V/s, 1.0V/s) \n
