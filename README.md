# Bayesian Parameter Estimation (BPE) for characterising perovskite solar cells

A BPE method utilising the perovskite solar cell device model, IonMonger, to derive posterior distributions over physical perovskite parameters associated with a J-V measurement of a perovskite device.

## Example use
There are only four parameters that require input by the user: \
`n_iter` - number of iterations in each Markov chain \
`n_chains` - number of Markov chains \
`prior_ranges` - numpy array containing prior range of each device parameter \
`y` - experimental J-V characteristics at each scan-rate (0.1V/s, 1.0V/s) \

The parameters `n_iter` and `n_chains` are specified in the main python script. The parameters `prior_ranges` and `y` are found in the `run_single_chain` function as global variables, this is required to allow access by each Markov chain.
