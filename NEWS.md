# Bayesian-HMM 0.0.3

* README expanded to include more usage (plotting of posteriors from mcmc results)
* Multiprocessing pool now closed after parallel chain resampling
* MCMC steps reordered
* Latent states only updated at the start of each MCMC step, not within the count update step
* Improve code hygeine, moving non-class code to separate module
* Store aggregate `None` state within the parameter dictionaries


# Bayesian-HMM 0.0.2

* All `HDPHMM` parameters and parameter priors now stored in dict objects `HDPHMM.parameters` and `HDPHMM.priors`
* `HDPHMM.mcmc` now returns a dict object


# Bayesian-HMM 0.0.1

* Initial release

