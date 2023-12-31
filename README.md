# Bayesian Parameter Estimation (BPE) for characterising perovskite solar cells

A BPE method utilising the perovskite solar cell device model, IonMonger, to derive posterior distributions over physical perovskite parameters associated with a J-V measurement of a perovskite device. This is the supporting code for the article 'Bayesian parameter estimation for characterising
mobile ion vacancies in perovskite solar cells' that may be found at https://doi.org/10.1088/2515-7655/ad0a38.

## Example use
There are four parameters that require input by the user: \
`n_iter` - number of iterations in each Markov chain \
`n_chains` - number of Markov chains \
`prior_ranges` - numpy array containing prior uniform range of each device parameter in the format (lower_bound, upper_bound) \
`y` - experimental J-V characteristics at each scan-rate (0.1V/s, 1.0V/s)

The parameters `n_iter` and `n_chains` are specified in the main python script. The parameters `prior_ranges` and `y` are found in the `run_single_chain` function as global variables, this is required to allow access by each Markov chain.

Note that all device parameters are input and used throughout the simulation in $\log_{10}$ form.

The simulation saves a `.npy` file (numpy array) of size `(n_chains,)` where each entry contains an array of size `(n_iter, 17)` that contains the `n_iter` samples for the 17 device parameters.
