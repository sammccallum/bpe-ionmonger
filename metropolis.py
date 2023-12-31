import numpy as np
import matlab.engine as matlab
import outputs as out
import multiprocessing
import matplotlib.pyplot as plt
import time

# Define the function that runs the IonMonger simulation and returns the outputs
def run_simulation(eng, inputs):
    # Convert numpy array into python list (matlab doesn't accept numpy arrays)
    params_list = []
    for i in range(len(inputs)):
        params_list.append(10**float(inputs[i]))

    # Call the master function in MATLAB using the MATLAB engine and calculate outputs
    # Scan rate of 0.1V/s
    sol_slow = eng.master_slow(params_list)
    outputs_slow = out.calculate_outputs(sol_slow['J'], sol_slow['V'])

    # Scan rate of 1.0V/s
    sol_fast = eng.master_fast(params_list)
    outputs_fast = out.calculate_outputs(sol_fast['J'], sol_fast['V'])

    # return np.asarray(outputs_fast)

    return np.concatenate((np.asarray(outputs_slow), np.asarray(outputs_fast)))

# Define the Metropolis Hastings algorithm
def metropolis_hastings(eng, initial_state, num_samples):
    # Initialize the current state and likelihood
    current_state = initial_state
    current_log_posterior = log_posterior(eng, current_state)
    
    # Initialize the samples and acceptance rate
    samples = [current_state]
    acceptance_rate = 0.0
    
    # Loop over the desired number of samples
    for i in range(num_samples):
        # Propose a new state using the proposal distribution
        proposed_state = proposal_distribution(current_state)
        while log_prior(proposed_state) == -np.inf:
            proposed_state = proposal_distribution(current_state)
        
        # Calculate the posterior of the proposed state
        proposed_log_posterior = log_posterior(eng, proposed_state)
        
        # Debugging print
        # print(f"Proposed state: {proposed_state}")
        print(f"Posterior: {np.exp(proposed_log_posterior)}")

        # Calculate the acceptance ratio
        acceptance_ratio = min(1, np.exp(proposed_log_posterior - current_log_posterior))
        
        # Accept or reject the proposed state
        if np.random.uniform() < acceptance_ratio:
            current_state = proposed_state
            current_log_posterior = proposed_log_posterior
            acceptance_rate += 1.0
        
        # Add the current state to the samples
        samples.append(current_state)
    
    # Return the samples and acceptance rate
    return samples, acceptance_rate / num_samples

# Scale the outputs so they are same order of magnitude
def scale_outputs(outputs):
    for i in range(10):
        outputs[i] = outputs[i] / y[i]
    return outputs

# Randomly sample inputs from uniform priors
def initial_sample():
    size = len(prior_ranges)
    initial_inputs = np.zeros(size)
    for i in range(size):
        initial_inputs[i] = np.random.uniform(prior_ranges[i, 0], prior_ranges[i, 1])
    return initial_inputs

# Define the log-prior distribution, which is a uniform distribution over the prior ranges
def log_prior(inputs):
    for i in range(len(inputs)):
        if (inputs[i] < prior_ranges[i][0]) or (inputs[i] > prior_ranges[i][1]):
            return -np.inf
    return 0.0

# Define the log-likelihood function using the run_simulation and likelihood functions
def log_likelihood(eng, inputs):
    outputs = run_simulation(eng, inputs)
    outputs = scale_outputs(outputs)

    # Calculate the mean squared error between the logged outputs and experimental logged outputs y
    mse = np.mean((outputs - np.ones((10,)))**2)
    # Return the log_likelihood, which is proportional to -0.5 * mse for a normal distribution
    return (-0.5 * mse/0.05)

# Log posterior is just the log prior + log likelihood
def log_posterior(eng, inputs):
    return log_prior(inputs) + log_likelihood(eng, inputs)

# Define the proposal distribution, which is a normal distribution centered on the current state
def proposal_distribution(current_state):
    return np.random.normal(current_state, jump_dist_sigmas*np.abs(current_state))

def run_single_chain(n_iter):
    global prior_ranges, jump_dist_sigmas, y
    prior_ranges = np.asarray([[-6.82, -6.22],    # perovskite layer width
                               [-10, -9.45],      # perovskite permittivity
                               [6.60, 7.60],      # absorption coefficient
                               [-5.77, -3.27],    # electron diffusion coefficient
                               [-5.77, -3.27],    # hole diffusion coefficient
                               [22, 26],          # mobile ion vacancy density
                               [-17, -12],        # mobile ion difffusion coefficient
                               [23.85, 24.30],    # ETL doping density
                               [-7.40, -6.60],    # ETL layer width
                               [-10.21, -9.51],   # ETL permittivity
                               [-8.0, -4.24],     # ETL electron diffusion coefficient
                               [21.90, 24.30],    # HTL doping density
                               [-7.40, -6.60],    # HTL layer width
                               [-10.75, -10.35],  # HTL permittivity
                               [-8, -5]])         # HTL hole diffusion coefficient
    
    y = np.asarray([22.3037,    # Jsc (0.1V/s)
                    1.0509,     # Voc reverse scan (0.1V/s)
                    0.9616,     # Voc forward scan (0.1V/s)
                    20.3622,    # MPP reverse scan (0.1V/s)
                    15.5894,    # MPP forward scan (0.1V/s)
                    22.3014,    # Jsc (1.0V/s)
                    1.0499,     # Voc reverse scan (1.0V/s)
                    1.0305,     # Voc forward scan (1.0V/s)
                    20.3547,    # MPP reverse scan (1.0V/s)
                    19.5014])   # MPP forward scan (1.0V/s)
    
    jump_dist_sigmas = np.asarray([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    
    np.random.seed()
    initial_inputs = initial_sample()
    # Start the MATLAB engine
    try:
        eng = matlab.start_matlab()
        eng.cd('IonMonger/')

        # Run the Metropolis Hastings algorithm to sample from the likelihood distribution
        samples, acceptance_rate = metropolis_hastings(eng, initial_inputs, num_samples=n_iter)

        # Print the acceptance rate and the mean and standard deviation of the samples
        print(f"Acceptance rate: {acceptance_rate:.2f}")
        print(f"Mean: {np.mean(samples, axis=0)}")
        print(f"Standard deviation: {np.std(samples, axis=0)}")

        # Stop the MATLAB engine
        eng.quit()

        return np.asarray(samples)
    except:
        print('matlab engine error')

def get_experimental(inputs):
    eng = matlab.start_matlab()
    eng.cd('IonMonger/')
     # Convert numpy array into python list (matlab doesn't accept numpy arrays)
    params_list = []
    for i in range(len(inputs)):
        params_list.append(10**float(inputs[i]))

    # Call the master function in MATLAB using the MATLAB engine and calculate outputs
    # Scan rate of 0.1V/s
    sol_slow = eng.master_slow(params_list)
    outputs_slow = out.calculate_outputs(sol_slow['J'], sol_slow['V'])

    # Scan rate of 1.0V/s
    sol_fast = eng.master_fast(params_list)
    outputs_fast = out.calculate_outputs(sol_fast['J'], sol_fast['V'])

    eng.quit()

    outputs = np.concatenate((np.asarray(outputs_slow), np.asarray(outputs_fast)))
    print(f"Experimental inputs: {inputs}")
    print(f"Experimental outputs: {outputs}")
    
    return outputs

def plot_J_V(J, V):
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 18,
    "lines.markersize": 10})

    V_r, V_f, J_r, J_f = out.get_reverse_and_forward(J, V) 

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(V_r, J_r, ls='-', color='k')
    ax.plot(V_f, J_f, ls='--', color='k')

    ax.set_ylim([0, 22.5])
    ax.set_xlim([0, 1.2])

    ax.set_xlabel(r'Voltage, V (V)')
    ax.set_ylabel(r'Current Density, J (mAcm$^{-2}$)')

    plt.savefig('J_V.png', dpi=300)

if __name__ == "__main__":
    n_iter = 500
    n_chains = 5

    multiprocessing.set_start_method("spawn")
    iterations = [n_iter] * n_chains
    pool = multiprocessing.Pool(processes=n_chains)
    pool_result = pool.map_async(run_single_chain, iterations)

    timeout = n_iter * 60
    pool_result.wait(timeout=timeout)
    if pool_result.ready():
        traces = pool_result.get()
    traces = np.array(traces, dtype='object')
    np.save('trace.npy', traces)
