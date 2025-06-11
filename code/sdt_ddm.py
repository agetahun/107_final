"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# create a folder for the graphs to be saved in
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT Summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    This version adds covariates for stimulus type and trial difficulty.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    # Extract stimulus type and difficulty from condition
    data = data.copy()
    data['stimulus_type'] = data['condition'] % 2  # 0=simple, 1=complex
    data['difficulty'] = data['condition'] // 2    # 0=easy, 1=hard

    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    N = len(data)

    # Convert to arrays for indexing
    pnum_idx = data['pnum'].values - 1  # Convert to 0-based indexing
    condition_idx = data['condition'].values
    stimulus_type = data['stimulus_type'].values
    difficulty = data['difficulty'].values
    
    # Define the hierarchical model
    with pm.Model() as sdt_model:
        # Group-level baseline parameters
        mean_d_prime_baseline = pm.Normal('mean_d_prime_baseline', mu=1.0, sigma=1.0)
        mean_criterion_baseline = pm.Normal('mean_criterion_baseline', mu=0.0, sigma=1.0)
        
        # ASSIGNMENT REQUIREMENT: Effects of Stimulus Type and Trial Difficulty
        # Effect of stimulus type (simple vs complex)
        d_prime_stimulus_effect = pm.Normal('d_prime_stimulus_effect', mu=0.0, sigma=0.5)
        criterion_stimulus_effect = pm.Normal('criterion_stimulus_effect', mu=0.0, sigma=0.5)
        
        # Effect of difficulty (easy vs hard)
        d_prime_difficulty_effect = pm.Normal('d_prime_difficulty_effect', mu=0.0, sigma=0.5)
        criterion_difficulty_effect = pm.Normal('criterion_difficulty_effect', mu=0.0, sigma=0.5)
        
        # Interaction effect
        d_prime_interaction = pm.Normal('d_prime_interaction', mu=0.0, sigma=0.5)
        criterion_interaction = pm.Normal('criterion_interaction', mu=0.0, sigma=0.5)
        
        # Individual-level variability
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=0.5)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=0.5)
        
        # Build linear predictors for each observation
        d_prime_mu = (mean_d_prime_baseline + 
                     d_prime_stimulus_effect * stimulus_type + 
                     d_prime_difficulty_effect * difficulty +
                     d_prime_interaction * stimulus_type * difficulty)
        
        criterion_mu = (mean_criterion_baseline + 
                       criterion_stimulus_effect * stimulus_type + 
                       criterion_difficulty_effect * difficulty +
                       criterion_interaction * stimulus_type * difficulty)
        
        # Individual-level parameters
        d_prime = pm.Normal('d_prime', mu=d_prime_mu, sigma=stdev_d_prime, shape=N)
        criterion = pm.Normal('criterion', mu=criterion_mu, sigma=stdev_criterion, shape=N)
        
        # Calculate hit and false alarm rates using SDT
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
                
        # Likelihood for signal trials
        pm.Binomial('hit_obs', 
                   n=data['nSignal'].values, 
                   p=hit_rate, 
                   observed=data['hits'].values)
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'].values, 
                   p=false_alarm_rate, 
                   observed=data['false_alarms'].values)
    
    return sdt_model

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # # Create output directory
    # OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            plt.tight_layout()
            
    # Save the figure
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')

def run_analysis(data_file_path):
    """
    Main analysis function for the assignment.
    Quantifies effects of Stimulus Type and Trial Difficulty on performance.
    """
    
    print("="*80)
    print("SDT AND DELTA PLOT ANALYSIS - ASSIGNMENT SOLUTION")
    print("="*80)
    
    # Step 1: Load and examine the data
    print("\n1. Loading SDT data...")
    sdt_data = read_data(data_file_path, prepare_for='sdt', display=True)
    
    print("\n2. Loading delta plot data...")
    dp_data = read_data(data_file_path, prepare_for='delta plots', display=True)
    
    # Step 2: Fit the hierarchical SDT model with covariates
    print("\n3. Fitting hierarchical SDT model with Stimulus Type and Difficulty effects...")
    sdt_model = apply_hierarchical_sdt_model(sdt_data)
    
    # Sample from the model
    print("   Sampling from posterior (this may take a few minutes)...")
    with sdt_model:
        trace = pm.sample(2000, tune=1000, chains=4, cores=4, 
                         target_accept=0.95, return_inferencedata=True)
    
    # Step 3: Check convergence
    print("\n4. Checking convergence...")
    try:
        # Get full summary statistics
        conv_stats = az.summary(trace)
        
        # Extract convergence metrics
        max_rhat = conv_stats['r_hat'].max()
        min_ess = conv_stats[['ess_bulk', 'ess_tail']].min().min()
        
        print(f"Max R-hat: {max_rhat:.4f}")
        print(f"Min Effective Sample Size: {min_ess:.0f}")
        
        if max_rhat > 1.1:
            print("WARNING: Convergence issues detected (R-hat > 1.1)")
            print("Problematic parameters:")
            print(conv_stats[conv_stats['r_hat'] > 1.1][['r_hat']])
        else:
            print("Model converged successfully")
            
        # Optional: Show full diagnostics for top parameters
        print("\nTop parameters by R-hat:")
        print(conv_stats.sort_values('r_hat', ascending=False).head(10))
        
    except Exception as e:
        print(f"Error in convergence check: {str(e)}")
        print("\nFallback - Showing trace summary:")
        print(az.summary(trace))
    
    # Step 4: Display posterior distributions
    print("\n5. Posterior distributions of key parameters:")
    
    # Create summary table
    summary = az.summary(trace, var_names=['d_prime_stimulus_effect', 'd_prime_difficulty_effect',
                                          'criterion_stimulus_effect', 'criterion_difficulty_effect',
                                          'd_prime_interaction', 'criterion_interaction'])
    print(summary)
    
    # Plot posterior distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Posterior Distributions of Effect Parameters', fontsize=16)
    
    effects = ['d_prime_stimulus_effect', 'd_prime_difficulty_effect', 'd_prime_interaction',
               'criterion_stimulus_effect', 'criterion_difficulty_effect', 'criterion_interaction']
    titles = ['d\' Stimulus Effect', 'd\' Difficulty Effect', 'd\' Interaction',
              'Criterion Stimulus Effect', 'Criterion Difficulty Effect', 'Criterion Interaction']
    
    for i, (effect, title) in enumerate(zip(effects, titles)):
        row, col = i // 3, i % 3
        az.plot_posterior(trace, var_names=[effect], ax=axes[row, col])
        axes[row, col].set_title(title)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'posterior_distributions.png')
    plt.close()
    
    # Step 5: Generate delta plots for first participant
    print("\n6. Generating delta plots...")
    first_participant = dp_data['pnum'].iloc[0]
    draw_delta_plots(dp_data, first_participant)
    
    # Step 6: Compare effects
    print("\n7. COMPARISON OF STIMULUS TYPE vs DIFFICULTY EFFECTS:")
    print("="*60)
    
    # Extract effect sizes
    stimulus_d_effect = trace.posterior['d_prime_stimulus_effect'].mean().values
    difficulty_d_effect = trace.posterior['d_prime_difficulty_effect'].mean().values
    stimulus_c_effect = trace.posterior['criterion_stimulus_effect'].mean().values  
    difficulty_c_effect = trace.posterior['criterion_difficulty_effect'].mean().values
    
    print(f"d' Effects:")
    print(f"  Stimulus Type Effect: {stimulus_d_effect:.3f}")
    print(f"  Difficulty Effect: {difficulty_d_effect:.3f}")
    print(f"  → {'Stimulus type' if abs(stimulus_d_effect) > abs(difficulty_d_effect) else 'Difficulty'} has larger effect on sensitivity")
    
    print(f"\nCriterion Effects:")
    print(f"  Stimulus Type Effect: {stimulus_c_effect:.3f}")
    print(f"  Difficulty Effect: {difficulty_c_effect:.3f}")
    print(f"  → {'Stimulus type' if abs(stimulus_c_effect) > abs(difficulty_c_effect) else 'Difficulty'} has larger effect on response bias")
    
    print(f"\nInterpretation:")
    print(f"- Check delta plots for RT patterns:")
    print(f"  * Flat slopes = bias effects")
    print(f"  * Positive slopes = drift rate effects")
    print(f"  * Different slopes for correct/error = selective influence")
    
    return trace, summary, sdt_data, dp_data

# Main execution
if __name__ == "__main__":
    DATA_FILE = "../data/data.csv"
    
    # Run the complete analysis
    try:
        trace, summary, sdt_data, dp_data = run_analysis(DATA_FILE)
        print(f"\nAnalysis complete! Results saved and displayed above.")
    except FileNotFoundError:
        print(f"Data file '{DATA_FILE}' not found.")
        print("Please update the DATA_FILE variable with the correct path.")
    except Exception as e:
        print(f"Error during analysis: {e}")