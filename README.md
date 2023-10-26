# wfmoments

Tools to compute second moments and heterozygosity under arbitrary deme models under the Wright-Fisher diffusion.


This package is meant to be used as an API.  Install the package and then import wfmoments and use the functions
to build various spatial models and compute their equilibrium and non-equilibrium
second moments and heterozygosities.

### Installation

Install using `pip`.  `wfmoments` requires `numpy` and `scipy`.  They will be automatically installed and updated
if you have not yet installed them.  To install, simply type:

`pip install git+https://github.com/jeffspence/wfmoments`


### Usage

The code itself is documented (better documentation to come).

A simple workflow showing off some of the features:

```
import wfmoments


# Get the ODE coefficients for a 1D spatial model
# with scaled mutation rate 1e-4, scaled migration
# rate 10, and 100 demes.
# This ODE descibes how the second moments of the
# allele frequencies across all pairs of demes
# changes through time.
m, v = wfmoments.build_1d_spatial(
    theta=1e-4,
    migration_rate=10,
    num_demes=100
)

# solve for the model at equilibrium
# to get the equilibrium second moments
eq = wfmoments.compute_equilibrium(
    moment_mat=m,
    const_vec=v
)

# Get out the pi across all 100 demes
species_pi = wfmoments.compute_pi(
    curr_moments=eq,
    demes=list(range(100))
)

# Get out the pi for the 50th deme:
deme_50_pi = wfmoments.compute_pi(
    curr_moments=eq,
    demes=[50]
)

# Get out the pi for the 50th-100th deme:
deme_50_or_greater_pi = wfmoments.compute_pi(
    curr_moments=eq,
    demes=list(range(50, 100))
)

# Get the second moments for a subsample 50 demes
new_moments = wfmoments.get_moments(
    curr_moments=eq,
    demes=list(range(50, 100))
)

# Set up a new migration model on just these 50 demes
m, v = wfmoments.build_1d_spatial(
    theta=1e-4,
    migration_rate=10,
    num_demes=50
)

# See what happens if we evolve this new model forward a bit
evolved_moments = wfmoments.evolve_forward(
    moment_mat=m,
    const_vec=v,
    curr_moments=new_moments,
    time=0.25
)
```
