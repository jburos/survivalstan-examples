data {
    
    // global dimensions
    int<lower=1> S; // number of subjects
    int<lower=1> T; // number of timepoints
    
    // information about each timepoint
    vector<lower=0>[T] t_obs;
    vector<lower=0>[T] t_dur;
    
    // data for terminal & recurrent event submodels
    int N; // number of observations in event submodels
    int M_t; // number of covariates for terminal-event submodel
    int M_r; // number of covariates for recurrent-event submodel
    int<lower=1, upper=S> s[N];
    int<lower=1, upper=T> t[N];
    int<lower=0, upper=1> event_t[N]; // indicator (1:y, 0:n) for terminal event
    int<lower=0, upper=1> event_r[N]; // indicator for recurrent event
    matrix[N, M_t] x_t;
    matrix[N, M_r] x_r;
    
    // data for longitudinal biomarker submodel (times are observation times)
    int N_l;
    int M_l;
    int<lower=1, upper=S> subject_l[N_l];
    vector<lower=0>[N_l] time_l;
    real y_l[N_l];
    matrix[N_l, M_l] x_l;
}
transformed data {
    vector[T] log_t_dur;  // log-duration for each timepoint
    log_t_dur = log(t_obs);
}
parameters {
    // longitudinal submodel 
    vector[M_l] B_l; // means for betas
    real Bt_l;     // mean for beta(time)
    real B0_l;
    vector[S] b0; // random effects b0 and b1
    matrix[S, 1] b1;
    real<lower=0> biomarker_sigma;
    
    // recurrent-event submodel
    vector[S] vi;     // subject-level frailty
    vector[M_r] B_r;    // betas for recurrent-event
    real eta0_r;       // weight on longitudinal-model component
    real eta1_r;
    vector[T] log_baseline_r_raw; 
    real<lower=0> baseline_r_sigma;
    real log_baseline_r_mu;
    
    // terminal-event submodel
    real alpha;       // weight on subject-frailty
    vector[M_t] B_t;    // betas for terminal-event
    real eta0_t;       // weight on longitudinal-model component
    real eta1_t;
    vector[T] log_baseline_t_raw; 
    real<lower=0> baseline_t_sigma;
    real log_baseline_t_mu;
}
transformed parameters {
    //vector[N_l] biomarker_value;
    vector[T] log_baseline_r;
    vector[T] log_baseline_t;
    vector[N] log_hazard_r;
    vector[N] log_hazard_t;
    
    //biomarker_value = B0_l + Bt_l*time_l + x_l*B_l + b0[subject_l] + b1[subject_l]*time_l;
    
    log_baseline_r = log_baseline_r_mu + log_baseline_r_raw + log_t_dur;
    log_baseline_t = log_baseline_t_mu + log_baseline_t_raw + log_t_dur;
    
    log_hazard_r = log_baseline_r[t] + vi[s] + x_r*B_r + eta0_r*b0[s] + eta1_r*to_vector(b1[s]);
    log_hazard_t = log_baseline_t[t] + vi[s] + x_t*B_t + eta0_t*b0[s] + eta1_t*to_vector(b1[s]);
}
model {
    // longitudinal submodel priors
    //biomarker_sigma ~ cauchy(0, 5);
    
    // recurrent event submodel priors
    log_baseline_r_mu ~ normal(0, 1);
    baseline_r_sigma ~ normal(0, 1);
    log_baseline_r_raw ~ normal(0, baseline_r_sigma);
    
    // terminal event submodel priors
    log_baseline_t_mu ~ normal(0, 1);
    baseline_t_sigma ~ normal(0, 1);
    log_baseline_t_raw ~ normal(0, baseline_t_sigma);
    
    // models
    //y_l ~ normal(biomarker_value, biomarker_sigma);
    event_t ~ poisson_log(log_hazard_t);
    event_r ~ poisson_log(log_hazard_r);
}
