# ==================================================
# Evaluation of Factor Models using genetic algorithms
# ==================================================
# Kushal Kattel (Group 1)
# 
# Description:
#   This script uses a Genetic Algorithm (GA) to search for the best combination 
#   of factors (independent variables) to explain a target variable 
#   using regression (R² as the fitness metric).
#
#   Main Steps:
#     1. Load and preprocess dataset (Excel file: either query(new).xlsx or 1000Lines_Clusters_Mappings.xlsm)
#     2. Run a GA with selectable population size, chromosome size, and generations
#     3. Collect best results, track factor importance across runs
#     4. Create comprehensive dashboard visualizations for analysis
#     5. Compare results across different configurations
#
# Recommendation:
#   1) description.pdf includes the technical explanation
#   2) README.md includes practical guide
# ==================================================

import pandas as pd
import numpy as np
import random
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Optional imports for interactive dashboards
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==================================================
# -------------------- USER PARAMETERS --------------------
# ===============================================ß===
CHROM_SIZE = 5          # Number of factors (genes) per chromosome
USE_QUERY_DATA = False  # Choose between datasets: True for query(new).xlsx, False for 1000Lines dataset

# ==================================================
# -------------------- DATA LOADING --------------------
# ==================================================
def load_data():
    """
    Loads the dataset depending on USE_QUERY_DATA parameter.
    Cleans column names, selects numeric factors, defines target column.
    
    Returns:
        df_data     : full DataFrame with original data
        X_all       : factor matrix (numeric predictors)
        y_all       : target variable (dependent)
        usable      : list of usable factor names
        target_col  : column name of target
    """
    if USE_QUERY_DATA:
        # Option 1: Query dataset
        df_data = pd.read_excel("../data/query(new).xlsx", engine='openpyxl')
        df_data.columns = df_data.columns.str.strip().str.upper()

        target_name = 'RET_EXC_LEAD1M'.upper()
        if target_name not in df_data.columns:
            raise ValueError(f"No target column '{target_name}' found in query data")

        target_col = target_name
        numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()

        # Use numeric columns starting from row 39 until 'QMJ_SAFETY'
        usable = []
        for col in numeric_cols[39:]:
            usable.append(col.upper())
            if col.upper() == 'QMJ_SAFETY':
                break

        # Drop missing rows in target
        df_data = df_data[df_data[target_col].notna()]
        X_all = df_data[usable]
        y_all = df_data[target_col]

        # Ensure target is not in usable factors
        usable = [col for col in usable if col != target_col]

    else:
        # Option 2: 1000Lines dataset
        file_path = '../data/1000Lines_ Clusters_Mappings.xlsm'
        df_data = pd.read_excel(file_path, sheet_name='1000Lines')
        df_groupings = pd.read_excel(file_path, sheet_name='Groupings')

        # Extract factor names starting from row 39
        factor_start_row = 39
        groupings_subset = df_groupings.iloc[factor_start_row:, [0, 1]]

        # Filter: keep only rows where column A and B are not null/empty
        valid_rows = groupings_subset[
            groupings_subset.iloc[:, 0].notna() & 
            groupings_subset.iloc[:, 1].notna() & 
            (groupings_subset.iloc[:, 1].astype(str).str.strip() != '')
        ]

        # Extract factor names from valid rows
        factor_list = [f.strip().upper() for f in valid_rows.iloc[:, 0].tolist()]

        # Clean column names to match
        df_data.columns = df_data.columns.str.strip().str.upper()

        # Find available factors that exist in the dataframe
        available_factors = [f for f in factor_list if f in df_data.columns]
        usable = available_factors

        # Target variable
        target_col = 'EXCESS RETURN IN USD IN MONTH T+1 (RET_EXC_LEAD1M)'.upper()
        if target_col not in df_data.columns:
            raise ValueError(f"Target '{target_col}' not found!")

        # Drop missing rows in target
        df_data = df_data[df_data[target_col].notna()]
        X_all = df_data[usable].select_dtypes(include=[np.number])
        y_all = df_data[target_col]

        # Ensure target is not in usable factors
        usable = [col for col in usable if col != target_col]
        print(f"Number of factors in usable (final): {len(usable)}")

    return df_data, X_all, y_all, usable, target_col

# ==================================================
# -------------------- GA COMPONENTS --------------------
# ==================================================
# The GA evolves a population of candidate solutions (chromosomes),
# where each chromosome represents a set of CHROM_SIZE factors.

# Cache fitness results for repeated chromosomes to improve efficiency
fitness_cache = {}

def get_fitness_with_pvals(chrom, X_all, y_all, target_col):
    """
    Evaluate a chromosome (factor set) using OLS regression.
    
    Args:
        chrom: list of factor names (chromosome)
        X_all: DataFrame with all factors
        y_all: target variable series
        target_col: name of target column
        
    Returns:
        score: R² value (float, -999 if invalid)
        pvals: dict of factor -> p-value
    """
    key = tuple(sorted(chrom))
    if key in fitness_cache:
        return fitness_cache[key]

    # Clean data subset
    subdata = X_all[list(chrom)].replace([np.inf, -np.inf], np.nan).join(y_all)
    subdata = subdata.dropna()

    if len(subdata) < 25:
        # Too few rows for reliable regression
        fitness_cache[key] = (-999, {f: np.nan for f in chrom})
        return -999, {f: np.nan for f in chrom}

    # Fit OLS regression
    Xc = sm.add_constant(subdata[list(chrom)])
    ols = sm.OLS(subdata[target_col], Xc).fit()
    score = ols.rsquared
    pvals = {f: ols.pvalues[f] for f in chrom}

    fitness_cache[key] = (score, pvals)
    return score, pvals

def create_unique_population(genes, pop_size):
    """
    Initialize population with unique random chromosomes.
    Each chromosome is a random CHROM_SIZE subset of factors.
    
    Args:
        genes: list of all available factor names
        pop_size: size of population to create
        
    Returns:
        pop: list of chromosomes (each chromosome is a list of factors)
    """
    pop, seen_sets = [], set()
    while len(pop) < pop_size:
        chrom = tuple(random.sample(genes, CHROM_SIZE))
        chrom_set = frozenset(chrom)
        if chrom_set not in seen_sets:
            seen_sets.add(chrom_set)
            pop.append(list(chrom))
    return pop

def roulette_selection(pop, X_all, y_all, target_col):
    """
    Roulette-wheel selection: probability of selection proportional to fitness.
    Handles negative fitness by shifting all values to be non-negative.
    
    Args:
        pop: population of chromosomes
        X_all, y_all, target_col: data for fitness evaluation
        
    Returns:
        selected chromosome
    """
    fitnesses = [get_fitness_with_pvals(c, X_all, y_all, target_col)[0] for c in pop]
    fitnesses = [f if f != -999 else 0 for f in fitnesses]

    min_fit = min(fitnesses)
    adjusted = [f - min_fit + 1e-6 for f in fitnesses]  # Ensure non-negativity
    total = sum(adjusted)

    if total <= 0:  # All invalid chromosomes
        return random.choice(pop)

    probs = [f / total for f in adjusted]
    return pop[np.random.choice(len(pop), p=probs)]

def uniform_crossover(p1, p2):
    """
    Uniform crossover: child inherits genes randomly from each parent.
    Avoids duplicate factors within the child chromosome.
    
    Args:
        p1, p2: parent chromosomes
        
    Returns:
        child: offspring chromosome
    """
    child, pool = [], set(p1 + p2)
    for i in range(CHROM_SIZE):
        if random.random() < 0.5 and p1[i] in pool:
            child.append(p1[i])
            pool.remove(p1[i])
        elif p2[i] in pool:
            child.append(p2[i])
            pool.remove(p2[i])
    
    # Fill remaining slots randomly from available factors
    while len(child) < CHROM_SIZE and pool:
        child.append(pool.pop())
    return child

def mutate(chrom, usable, rate=0.2):
    """
    Mutation: randomly replace one factor in chromosome.
    
    Args:
        chrom: chromosome to mutate
        usable: list of all available factors
        rate: mutation probability
        
    Returns:
        mutated chromosome
    """
    if random.random() < rate:
        i = random.randrange(CHROM_SIZE)
        choices = list(set(usable) - set(chrom))
        if choices:
            chrom[i] = random.choice(choices)
    return chrom

def is_duplicate(chrom, others):
    """Check if chromosome already exists in the list."""
    return any(set(chrom) == set(o) for o in others)

# ==================================================
# -------------------- GA EXECUTION --------------------
# ==================================================
def run_ga(X_all, y_all, usable, target_col, pop_sizes, generations, runs_per_config):
    """
    Run GA for multiple population sizes and generations.
    
    Args:
        X_all, y_all: data matrices
        usable: list of usable factor names
        target_col: target column name
        pop_sizes: list of population sizes to test
        generations: list of generation counts to test
        runs_per_config: number of independent runs per configuration
    
    Returns:
        all_summaries: list of summary dicts for each config
        all_factor_data: dict with detailed results for each config
    """
    all_summaries, all_factor_data = [], {}

    # Test all combinations of population sizes and generation counts
    for POP_SIZE in pop_sizes:
        for GEN in generations:
            print("\n" + "=" * 80)
            print(f"▶ POP_SIZE={POP_SIZE}, GENERATIONS={GEN}")
            print("-" * 80)

            # Store results for this configuration
            run_best_scores, run_best_factors, per_gen_best = [], [], []
            per_factor_scores, per_factor_pvals = {}, {}

            # Run multiple independent GA experiments for this config
            for run_idx in range(1, runs_per_config + 1):
                print(f"\n--- Run {run_idx}/{runs_per_config} ---")

                # Initialize Population
                pop = create_unique_population(usable, POP_SIZE)
                best_score, last_impr = -np.inf, 0
                start = time.time()
                gen_best_scores = []

                # Evolution Loop
                for g in range(GEN):
                    # Evaluate and rank all chromosomes in population by fitness
                    scored = sorted(
                        [(get_fitness_with_pvals(c, X_all, y_all, target_col), c) for c in pop],
                        key=lambda x: x[0][0],  # Sort by R² value
                        reverse=True
                    )
                    pop = [c for _, c in scored]  # Keep chromosomes only (sorted)
                    best = pop[0]  # Best chromosome in this generation
                    score, pvals = get_fitness_with_pvals(best, X_all, y_all, target_col)
                    gen_best_scores.append(score)

                    # Print progress for early generations and every 25 generations
                    if g < 10 or g % 25 == 0:
                        print(f"Gen {g+1:03d} | R²: {score:.4f} | {best}")

                    # Update best score if this generation improved it
                    if score > best_score:
                        best_score, last_impr = score, g + 1

                    # Early stopping if solution is essentially perfect
                    if score > 0.999:
                        print(f"✅ Early stop at gen {g+1}")
                        break

                    # Elitism: keep best chromosomes
                    elite_ct = max(1, POP_SIZE // 10)  # 10% elites
                    elites = []
                    for c in pop:
                        if not is_duplicate(c, elites):
                            elites.append(c)
                        if len(elites) == elite_ct:
                            break

                    # Generate Next Generation
                    next_gen = elites.copy()
                    while len(next_gen) < POP_SIZE:
                        # Select parents using fitness-proportionate selection
                        p1 = roulette_selection(pop, X_all, y_all, target_col)
                        p2 = roulette_selection(pop, X_all, y_all, target_col)

                        # Crossover to produce child
                        child = uniform_crossover(p1, p2)

                        # Apply mutation
                        child = mutate(child, usable)

                        # Add child if unique
                        if not is_duplicate(child, next_gen):
                            next_gen.append(child)

                    # Replace population with new generation
                    pop = next_gen

                # Collect Run Results
                elapsed = time.time() - start
                run_best_scores.append(float(best_score))
                run_best_factors.append(best)
                per_gen_best.append(gen_best_scores)

                # Record factor-level statistics from best chromosome
                for f in best:
                    if f not in per_factor_scores:
                        per_factor_scores[f], per_factor_pvals[f] = [], []
                    if score != -999:
                        per_factor_scores[f].append(score)
                        per_factor_pvals[f].append(pvals[f])

                print(f"Run {run_idx} best R²: {best_score:.4f} | Time: {elapsed:.2f}s | Last improvement gen: {last_impr}")

            # Configuration Summary
            avg_score = float(np.mean(run_best_scores))
            std_score = float(np.std(run_best_scores))
            best_idx = np.argmax(run_best_scores)

            print("\n📊 Summary for this config:")
            print(f"Average R²: {avg_score:.4f} | Std R²: {std_score:.4f}")
            print(f"Best factors across runs: {run_best_factors[best_idx]}")

            # Save summary for this configuration
            all_summaries.append({
                "Population Size": POP_SIZE,
                "Generations": GEN,
                "Average R²": avg_score,
                "Std R²": std_score,
                "Best Factors": run_best_factors[best_idx],
            })

            # Save detailed factor-level data for this configuration
            key = f"POP{POP_SIZE}_GEN{GEN}"
            all_factor_data[key] = {
                "per_gen_best": per_gen_best,
                "factor_scores": per_factor_scores,
                "factor_pvals": per_factor_pvals
            }

    return all_summaries, all_factor_data

# ==================================================
# -------------------- DASHBOARD VISUALIZATIONS --------------------
# ==================================================

def truncate_factor_name(name, max_length=8):
    """
    Truncate factor names for better readability
    """
    if len(name) <= max_length:
        return name
    
    # Remove common prefixes/suffixes
    name = name.replace('_LEAD1M', '').replace('_T1', '').replace('_M1', '')
    name = name.replace('EXCESS_', '').replace('RETURN_', '').replace('RET_', '')
    
    if len(name) <= max_length:
        return name
    
    # If still too long, take first part and add ellipsis
    return name[:max_length-2] + '..'

def create_individual_config_dashboard(key, data, X_all, y_all, target_col):
    """
    Create comprehensive dashboard for individual configuration.
    Split into two separate dashboards for better readability.
    """
    POP_SIZE = int(key.split("_")[0][3:])
    GEN = int(key.split("_")[1][3:])
    
    print(f"\n🎯 Creating Dashboard for POP_SIZE={POP_SIZE}, GENERATIONS={GEN}")
    
    # DASHBOARD 1: Performance Analysis (4 plots)
    fig1 = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 2, figure=fig1, hspace=0.5, wspace=0.4, 
                   left=0.08, right=0.95, top=0.90, bottom=0.12)
    
    # 1. Convergence Plot
    ax1 = fig1.add_subplot(gs1[0, 0])
    colors = plt.cm.Set3(np.linspace(0, 1, len(data["per_gen_best"])))
    valid_y_min = float('inf')
    valid_y_max = float('-inf')
    
    for run_idx, run_scores in enumerate(data["per_gen_best"]):
        run_scores = np.array(run_scores)
        
        # Filter out invalid scores (-999, -1000, etc.)
        valid_scores = run_scores[run_scores > -100]  # Threshold to exclude invalid values
        
        if len(valid_scores) == 0:
            continue  # Skip this run if no valid scores
            
        # Find valid score indices in original array
        valid_indices = np.where(run_scores > -100)[0]
        
        # Plot only valid scores with their correct generation indices
        ax1.plot(valid_indices, valid_scores, 
                label=f"Run {run_idx+1}", color=colors[run_idx], linewidth=2, marker='o', markersize=2)
        
        # Track valid y-axis range for proper scaling
        if len(valid_scores) > 0:
            valid_y_min = min(valid_y_min, np.min(valid_scores))
            valid_y_max = max(valid_y_max, np.max(valid_scores))
    
    ax1.set_xlabel("Generation", fontsize=11)
    ax1.set_ylabel("R²", fontsize=11)
    ax1.set_title(f"Convergence (POP={POP_SIZE}, GEN={GEN})", fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    
    # Set reasonable y-axis limits based on valid data only
    if valid_y_min != float('inf') and valid_y_max != float('-inf'):
        y_range = valid_y_max - valid_y_min
        ax1.set_ylim(max(0, valid_y_min - 0.1 * y_range), valid_y_max + 0.1 * y_range)
    
    # 2. Performance Statistics
    ax2 = fig1.add_subplot(gs1[0, 1])
    # Filter out invalid final scores
    final_scores = [scores[-1] for scores in data["per_gen_best"] if len(scores) > 0 and scores[-1] > -100]
    
    if len(final_scores) > 0:
        ax2.hist(final_scores, bins=max(3, len(final_scores)//2), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(final_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(final_scores):.3f}')
        ax2.axvline(np.median(final_scores), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(final_scores):.3f}')
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No valid final scores', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_xlabel("Final R²", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Final R² Distribution", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    
    # 3. Learning Curve Analysis
    ax3 = fig1.add_subplot(gs1[1, 0])
    
    # Filter valid runs first
    valid_runs = []
    for scores in data["per_gen_best"]:
        if len(scores) > 0:
            # Filter out invalid scores within each run
            valid_scores_in_run = [score for score in scores if score > -100]
            if len(valid_scores_in_run) > 0:
                valid_runs.append(valid_scores_in_run)
    
    if len(valid_runs) > 0:
        # Handle different run lengths due to early stopping and filtering
        max_gen = max(len(scores) for scores in valid_runs)
        
        # Pad shorter runs with their final values
        padded_scores = []
        for scores in valid_runs:
            if len(scores) < max_gen:
                extended_scores = list(scores) + [scores[-1]] * (max_gen - len(scores))
                padded_scores.append(extended_scores)
            else:
                padded_scores.append(scores)
        
        if len(padded_scores) > 0:
            generations = range(max_gen)
            mean_scores = np.mean(padded_scores, axis=0)
            std_scores = np.std(padded_scores, axis=0)
            
            ax3.plot(generations, mean_scores, 'b-', linewidth=3, label='Mean R²')
            ax3.fill_between(generations, mean_scores - std_scores, mean_scores + std_scores, 
                            alpha=0.3, color='blue', label='±1 Std Dev')
            ax3.legend(fontsize=9)
            
            # Set reasonable y-axis limits
            y_min = np.min(mean_scores - std_scores)
            y_max = np.max(mean_scores + std_scores)
            y_range = y_max - y_min
            ax3.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)
        else:
            ax3.text(0.5, 0.5, 'No valid learning curve data', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No valid runs for learning curve', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_xlabel("Generation", fontsize=11)
    ax3.set_ylabel("R²", fontsize=11)
    ax3.set_title("Learning Curve", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=9)
    
    # 4. Configuration Summary
    ax4 = fig1.add_subplot(gs1[1, 1])
    ax4.axis('off')
    
    # Calculate improvement rates for summary (only from valid runs)
    improvement_rates = []
    for run_scores in data["per_gen_best"]:
        if len(run_scores) > 1:
            # Filter valid scores first
            valid_scores = [score for score in run_scores if score > -100]
            if len(valid_scores) > 1:
                improvements = [valid_scores[i] - valid_scores[i-1] for i in range(1, len(valid_scores)) 
                               if valid_scores[i] > valid_scores[i-1]]
                improvement_rates.append(len(improvements))
    
    # Calculate metrics only from valid final scores
    valid_final_scores = [scores[-1] for scores in data["per_gen_best"] 
                         if len(scores) > 0 and scores[-1] > -100]
    
    if len(valid_final_scores) > 0:
        best_r2 = max(valid_final_scores)
        avg_r2 = np.mean(valid_final_scores)
        std_r2 = np.std(valid_final_scores)
    else:
        best_r2 = avg_r2 = std_r2 = 0.0
    
    summary_text = f"""CONFIG SUMMARY

Pop Size: {POP_SIZE}
Generations: {GEN}
Valid Runs: {len(valid_final_scores)}/{len(data["per_gen_best"])}

METRICS
Best R²: {best_r2:.4f}
Avg R²: {avg_r2:.4f}
Std R²: {std_r2:.4f}

EFFICIENCY
Avg Improvements/Run: {np.mean(improvement_rates) if improvement_rates else 0:.1f}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle(f'Performance Dashboard 1/2 - POP:{POP_SIZE} | GEN:{GEN}', 
                fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout()
    plt.show()
    
    # DASHBOARD 2: Factor Analysis (4 plots)
    fig2 = plt.figure(figsize=(16, 10))
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.5, wspace=0.4,
                   left=0.08, right=0.95, top=0.90, bottom=0.12)
    
    # Prepare factor data - also filter out invalid scores for factors
    all_factor_counts = {}
    all_factor_rsq = {}
    for f, scores in data["factor_scores"].items():
        # Filter out invalid scores for factors too
        valid_scores = [score for score in scores if score > -100]
        if len(valid_scores) > 0:
            all_factor_counts[f] = len(valid_scores)
            all_factor_rsq[f] = valid_scores
    
    # Get top factors 
    sorted_factors = sorted(all_factor_counts.items(), key=lambda x: x[1], reverse=True)
    top_factors = [f for f, _ in sorted_factors[:10]]
    
    # 1. Factor Importance Analysis 
    ax5 = fig2.add_subplot(gs2[0, 0])
    if len(top_factors) > 0:
        factor_frequencies = [all_factor_counts[f] for f in top_factors]
        
        bars = ax5.bar(range(len(top_factors)), factor_frequencies, color='lightcoral', alpha=0.8)
        ax5.set_xticks(range(len(top_factors)))
        
        factor_labels = [truncate_factor_name(f, 10) for f in top_factors]
        
        ax5.set_xticklabels(factor_labels, rotation=45, ha='right', fontsize=9)
        ax5.set_ylabel("Frequency", fontsize=11)
        ax5.set_title("Top Factors Selection", fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        ax5.tick_params(axis='both', which='major', labelsize=9)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        # Add hover-like info in title or legend for full names
        full_names = [f[:30] + ('...' if len(f) > 30 else '') for f in top_factors[:3]]
        ax5.text(0.02, 0.98, f"Top 3 Full Names:\n" + "\n".join([f"• {name}" for name in full_names]), 
                transform=ax5.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        ax5.text(0.5, 0.5, 'No valid factor data', ha='center', va='center', transform=ax5.transAxes)
    
    # 2. Factor Performance Boxplot
    ax6 = fig2.add_subplot(gs2[0, 1])
    if len(top_factors) >= 4:
        top_6_factors = top_factors[:6]  # Use top 6 for better visibility
        top_factor_scores = [all_factor_rsq[f] for f in top_6_factors]
        
        box_plot = ax6.boxplot(top_factor_scores, patch_artist=True, notch=True)
        # Labels for boxplot - use truncate function
        box_labels = [truncate_factor_name(f, 8) for f in top_6_factors]
        
        ax6.set_xticklabels(box_labels, rotation=45, ha='right', fontsize=9)
        
        # Color the boxes
        colors_box = plt.cm.viridis(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.set_ylabel("R²", fontsize=11)
        ax6.set_title("Top Factor Performance", fontsize=12, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        ax6.tick_params(axis='both', which='major', labelsize=9)
    else:
        ax6.text(0.5, 0.5, 'Insufficient factor data for boxplot', ha='center', va='center', transform=ax6.transAxes)
    
    # 3. Factor P-value Analysis
    ax7 = fig2.add_subplot(gs2[1, 0])
    if len(top_factors):
        top_8_factors = top_factors[:8]
        significant_factors = []
        avg_pvals = []
        
        for f in top_8_factors:
            if f in data["factor_pvals"] and len(data["factor_pvals"][f]) > 0:
                pvals = [p for p in data["factor_pvals"][f] if not np.isnan(p)]
                if pvals:
                    avg_pval = np.mean(pvals)
                    # Use truncate function
                    significant_factors.append(truncate_factor_name(f, 8))
                    avg_pvals.append(avg_pval)
        
        if significant_factors:
            colors_pval = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in avg_pvals]
            bars_pval = ax7.bar(range(len(significant_factors)), avg_pvals, color=colors_pval, alpha=0.8)
            ax7.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05', linewidth=1)
            ax7.axhline(0.1, color='orange', linestyle='--', alpha=0.7, label='α=0.10', linewidth=1)
            ax7.set_xticks(range(len(significant_factors)))
            ax7.set_xticklabels(significant_factors, rotation=45, ha='right', fontsize=9)
            ax7.set_ylabel("Avg P-value", fontsize=11)
            ax7.set_title("Top Factor Significance", fontsize=12, fontweight='bold')
            ax7.legend(fontsize=8)
            ax7.grid(axis='y', alpha=0.3)
            ax7.tick_params(axis='both', which='major', labelsize=9)
        else:
            ax7.text(0.5, 0.5, 'No valid p-value data', ha='center', va='center', transform=ax7.transAxes)
    else:
        ax7.text(0.5, 0.5, 'Insufficient factors for p-value analysis', ha='center', va='center', transform=ax7.transAxes)
    
    # 4. Optimization Efficiency
    ax8 = fig2.add_subplot(gs2[1, 1])
    
    if improvement_rates:
        bars_imp = ax8.bar(range(len(improvement_rates)), improvement_rates, color='green', alpha=0.7)
        ax8.set_xlabel("Run", fontsize=11)
        ax8.set_ylabel("Improvements", fontsize=11)
        ax8.set_title("Efficiency per Run", fontsize=12, fontweight='bold')
        ax8.set_xticks(range(len(improvement_rates)))
        ax8.set_xticklabels([f"R{i+1}" for i in range(len(improvement_rates))], fontsize=9)
        ax8.grid(axis='y', alpha=0.3)
        ax8.tick_params(axis='both', which='major', labelsize=9)
        
        # Add value labels on bars
        for i, bar in enumerate(bars_imp):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'No improvement data', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=11)
    
    plt.suptitle(f'Factor Analysis Dashboard 2/2 - POP:{POP_SIZE} | GEN:{GEN}', 
                fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout()
    plt.show()
    
    # Create OLS Analysis for this configuration using top factors
    if len(top_factors) > 0:
        create_ols_analysis_dashboard(top_factors, X_all, y_all, POP_SIZE, GEN)
    else:
        print(f"⚠️ No valid factors found for OLS analysis in POP:{POP_SIZE} | GEN:{GEN}")

def create_ols_analysis_dashboard(top_factors, X_all, y_all, POP_SIZE, GEN):
    """
    Create detailed OLS regression analysis dashboard
    """
    print(f"📈 Creating OLS Analysis Dashboard for POP_SIZE={POP_SIZE}, GENERATIONS={GEN}")
    
    # Prepare data
    X_top = X_all[top_factors].replace([np.inf, -np.inf], np.nan).dropna()
    y_top = y_all.loc[X_top.index]
    
    if X_top.shape[0] == 0:
        print("⚠️ Warning: No data available for OLS analysis")
        return
    
    # Fit OLS model
    Xc_top = sm.add_constant(X_top)
    ols_top = sm.OLS(y_top, Xc_top).fit()
    
    # Create dashboard 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # 1. Coefficients plot with capped values and extreme indicators
    coefs = ols_top.params.drop('const', errors='ignore')
    pvals = ols_top.pvalues.drop('const', errors='ignore')
    colors_coef = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in pvals]
    
    # Truncate factor names for better readability
    coef_labels = [truncate_factor_name(f, 15) for f in coefs.index]
    
    # Cap coefficients at -1 and +1 for visualization, but track original values
    original_coefs = coefs.values.copy()
    capped_coefs = np.clip(coefs.values, -1, 1)
    
    # Create bars with capped values
    bars = ax1.barh(range(len(coefs)), capped_coefs, color=colors_coef, alpha=0.7)
    ax1.set_yticks(range(len(coefs)))
    ax1.set_yticklabels(coef_labels, fontsize=10)
    ax1.set_xlabel("Coefficient Value (capped at ±1)", fontsize=12)
    ax1.set_title("Regression Coefficients", fontsize=12, fontweight='bold')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax1.axvline(1, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(-1, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_xlim(-1.2, 1.2)  # Set fixed limits
    
    # Add significance markers and extreme value indicators
    for i, (bar, pval, orig_coef, capped_coef) in enumerate(zip(bars, pvals, original_coefs, capped_coefs)):
        y_pos = bar.get_y() + bar.get_height()/2
        
        # Add significance markers
        significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        if significance:
            sig_x = capped_coef + (0.05 if capped_coef >= 0 else -0.05)
            ax1.text(sig_x, y_pos, significance, ha='left' if capped_coef >= 0 else 'right', 
                    va='center', fontweight='bold', color='red', fontsize=10)
        
        # Add extreme value indicators and actual values
        if abs(orig_coef) > 1:
            # Show that value was capped
            extreme_marker = ">" if orig_coef > 1 else "<"
            marker_x = 1.05 if orig_coef > 0 else -1.05
            ax1.text(marker_x, y_pos, f"{extreme_marker}1", ha='center', va='center', 
                    fontweight='bold', color='darkred', fontsize=9)
            
            # Show actual value
            actual_x = 1.15 if orig_coef > 0 else -1.15
            ax1.text(actual_x, y_pos, f"({orig_coef:.2f})", ha='left' if orig_coef > 0 else 'right', 
                    va='center', fontsize=8, color='darkred', style='italic')
        
        # Show actual values for all coefficients at the bar end
        value_x = capped_coef + (0.02 if capped_coef >= 0 else -0.02)
        if abs(orig_coef) <= 1:  # Only show if not already shown as extreme
            ax1.text(value_x, y_pos, f"{orig_coef:.3f}", ha='left' if capped_coef >= 0 else 'right', 
                    va='center', fontsize=8, color='black')
    
    # 2. Residuals vs Fitted
    fitted_values = ols_top.fittedvalues
    residuals = ols_top.resid
    ax2.scatter(fitted_values, residuals, alpha=0.6, color='blue', s=30)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel("Fitted Values", fontsize=12)
    ax2.set_ylabel("Residuals", fontsize=12)
    ax2.set_title("Residuals vs Fitted Values", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # 3. Q-Q plot for normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    
    # 4. Model statistics
    ax4.axis('off')
    stats_text = f"""
REGRESSION STATISTICS

R-squared: {ols_top.rsquared:.4f}
Adjusted R-squared: {ols_top.rsquared_adj:.4f}
F-statistic: {ols_top.fvalue:.2f}
Prob (F-statistic): {ols_top.f_pvalue:.2e}

MODEL QUALITY
AIC: {ols_top.aic:.2f}
BIC: {ols_top.bic:.2f}
Log-Likelihood: {ols_top.llf:.2f}

SIGNIFICANT FACTORS
p < 0.001: {sum(pvals < 0.001)} factors
p < 0.01:  {sum(pvals < 0.01)} factors  
p < 0.05:  {sum(pvals < 0.05)} factors

FACTOR NAMES (Full):
{chr(10).join([f"• {truncate_factor_name(f, 20)}" for f in coefs.index[:5]])}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.suptitle(f'OLS Regression Analysis - POP:{POP_SIZE} | GEN:{GEN}', 
                fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()
    
    # Print detailed OLS summary
    print("\n" + "="*80)
    print(f"📊 DETAILED OLS SUMMARY - POP:{POP_SIZE} | GEN:{GEN}")
    print("="*80)
    print(ols_top.summary())

def create_cross_config_comparison_dashboard(all_summaries, all_factor_data):
    """
    Create comprehensive comparison dashboard across all configurations
    """
    print("\n🌟 Creating Cross-Configuration Comparison Dashboard")
    
    # Prepare comparison data
    configs = []
    avg_r2_scores = []
    std_r2_scores = []
    pop_sizes = []
    generations = []
    best_r2_scores = []
    best_factors_list = [] 
    
    for summary in all_summaries:
        configs.append(f"P{summary['Population Size']}_G{summary['Generations']}")
        avg_r2_scores.append(summary['Average R²'])
        std_r2_scores.append(summary['Std R²'])
        pop_sizes.append(summary['Population Size'])
        generations.append(summary['Generations'])
        best_factors_list.append(', '.join([truncate_factor_name(f, 10) for f in summary['Best Factors'][:3]]))  # Use truncate function
        
        # Find best R² for this config
        key = f"POP{summary['Population Size']}_GEN{summary['Generations']}"
        if key in all_factor_data:
            # Filter out invalid scores when finding best
            best_scores = []
            for scores in all_factor_data[key]["per_gen_best"]:
                valid_scores = [s for s in scores if s > -100]
                if valid_scores:
                    best_scores.append(max(valid_scores))
            best_r2_scores.append(max(best_scores) if best_scores else summary['Average R²'])
        else:
            best_r2_scores.append(summary['Average R²'])
    
    # ============ DASHBOARD 1: Performance Comparison (4 plots) ============
    fig1 = plt.figure(figsize=(20, 12))
    gs1 = GridSpec(2, 2, figure=fig1, hspace=0.4, wspace=0.3)
    
    # 1. Configuration Performance Comparison
    ax1 = fig1.add_subplot(gs1[0, 0])
    x_pos = np.arange(len(configs))
    bars1 = ax1.bar(x_pos, avg_r2_scores, yerr=std_r2_scores, capsize=5, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(configs))), alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs, rotation=45, ha='right', fontsize=12)
    ax1.set_ylabel('Average R²', fontsize=14)
    ax1.set_title('Configuration Performance Comparison', fontsize=16, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Add value labels
    for i, (bar, avg, std) in enumerate(zip(bars1, avg_r2_scores, std_r2_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                f'{avg:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Best vs Average Performance
    ax2 = fig1.add_subplot(gs1[0, 1])
    width = 0.35
    x = np.arange(len(configs))
    bars_avg = ax2.bar(x - width/2, avg_r2_scores, width, label='Average R²', 
                      color='skyblue', alpha=0.8)
    bars_best = ax2.bar(x + width/2, best_r2_scores, width, label='Best R²', 
                       color='orange', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=12)
    ax2.set_ylabel('R²', fontsize=14)
    ax2.set_title('Average vs Best Performance', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # 3. Population Size vs Performance
    ax3 = fig1.add_subplot(gs1[1, 0])
    unique_pop_sizes = sorted(list(set(pop_sizes)))
    pop_performance = {pop: [] for pop in unique_pop_sizes}
    
    for i, pop in enumerate(pop_sizes):
        pop_performance[pop].append(avg_r2_scores[i])
    
    box_data = [pop_performance[pop] for pop in unique_pop_sizes]
    box_plot = ax3.boxplot(box_data, labels=unique_pop_sizes, patch_artist=True, notch=True)
    
    colors_pop = plt.cm.Set2(np.linspace(0, 1, len(box_plot['boxes'])))
    for patch, color in zip(box_plot['boxes'], colors_pop):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Population Size', fontsize=14)
    ax3.set_ylabel('Average R²', fontsize=14)
    ax3.set_title('Population Size Impact', fontsize=16, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # 4. Generation Count vs Performance
    ax4 = fig1.add_subplot(gs1[1, 1])
    unique_generations = sorted(list(set(generations)))
    gen_performance = {gen: [] for gen in unique_generations}
    
    for i, gen in enumerate(generations):
        gen_performance[gen].append(avg_r2_scores[i])
    
    box_data_gen = [gen_performance[gen] for gen in unique_generations]
    box_plot_gen = ax4.boxplot(box_data_gen, labels=unique_generations, patch_artist=True, notch=True)
    
    colors_gen = plt.cm.Set3(np.linspace(0, 1, len(box_plot_gen['boxes'])))
    for patch, color in zip(box_plot_gen['boxes'], colors_gen):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Generations', fontsize=14)
    ax4.set_ylabel('Average R²', fontsize=14)
    ax4.set_title('Generation Count Impact', fontsize=16, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    plt.suptitle('GA Configuration Comparison Dashboard 1/2', 
                fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()
    
    # ============ DASHBOARD 2: Advanced Analysis (4 plots) ============
    fig2 = plt.figure(figsize=(20, 12))
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.4, wspace=0.3)
    
    # 1. Heatmap of Configuration Performance
    ax5 = fig2.add_subplot(gs2[0, 0])
    
    # Create performance matrix
    pop_gen_matrix = {}
    for i, summary in enumerate(all_summaries):
        pop = summary['Population Size']
        gen = summary['Generations']
        if pop not in pop_gen_matrix:
            pop_gen_matrix[pop] = {}
        pop_gen_matrix[pop][gen] = avg_r2_scores[i]
    
    # Convert to matrix format
    pop_sorted = sorted(pop_gen_matrix.keys())
    gen_sorted = sorted(set(generations))
    matrix = np.zeros((len(pop_sorted), len(gen_sorted)))
    
    for i, pop in enumerate(pop_sorted):
        for j, gen in enumerate(gen_sorted):
            if gen in pop_gen_matrix[pop]:
                matrix[i, j] = pop_gen_matrix[pop][gen]
            else:
                matrix[i, j] = np.nan
    
    im = ax5.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax5.set_xticks(range(len(gen_sorted)))
    ax5.set_xticklabels(gen_sorted, fontsize=12)
    ax5.set_yticks(range(len(pop_sorted)))
    ax5.set_yticklabels(pop_sorted, fontsize=12)
    ax5.set_xlabel('Generations', fontsize=14)
    ax5.set_ylabel('Population Size', fontsize=14)
    ax5.set_title('Performance Heatmap', fontsize=16, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pop_sorted)):
        for j in range(len(gen_sorted)):
            if not np.isnan(matrix[i, j]):
                ax5.text(j, i, f'{matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    
    plt.colorbar(im, ax=ax5, label='Average R²')
    
    # 2. Factor Consistency Analysis
    ax6 = fig2.add_subplot(gs2[0, 1])
    
    # Collect all factors across configurations
    all_factors_frequency = defaultdict(int)
    for key, data in all_factor_data.items():
        for factor in data["factor_scores"].keys():
            # Count only valid scores
            valid_scores = [s for s in data["factor_scores"][factor] if s > -100]
            all_factors_frequency[factor] += len(valid_scores)
    
    # Get consistent factors
    top_consistent = sorted(all_factors_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    factor_names = [truncate_factor_name(f, 10) for f, _ in top_consistent]  # Use truncate function
    factor_counts = [count for _, count in top_consistent]
    
    bars_factors = ax6.bar(range(len(factor_names)), factor_counts, 
                          color=plt.cm.plasma(np.linspace(0, 1, len(factor_names))), alpha=0.8)
    ax6.set_xticks(range(len(factor_names)))
    ax6.set_xticklabels(factor_names, rotation=45, ha='right', fontsize=10)
    ax6.set_ylabel('Total Selections', fontsize=14)
    ax6.set_title('Top Most Consistent Factors', fontsize=16, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    ax6.tick_params(axis='both', which='major', labelsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars_factors, factor_counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Convergence Speed Analysis
    ax7 = fig2.add_subplot(gs2[1, 0])
    
    convergence_speeds = []
    config_names = []
    
    for key, data in all_factor_data.items():
        pop_size = int(key.split("_")[0][3:])
        gen_count = int(key.split("_")[1][3:])
        config_name = f"P{pop_size}_G{gen_count}"
        
        # Calculate average generations to reach 90% of final performance
        speeds = []
        for run_scores in data["per_gen_best"]:
            if len(run_scores) == 0:
                continue
            
            # Filter valid scores first
            valid_scores = [s for s in run_scores if s > -100]
            if len(valid_scores) == 0:
                continue
                
            final_score = valid_scores[-1]
            target_score = 0.9 * final_score
            
            # Find generation indices that correspond to valid scores
            valid_indices = [i for i, s in enumerate(run_scores) if s > -100]
            
            for idx, score in zip(valid_indices, valid_scores):
                if score >= target_score:
                    speeds.append(idx + 1)  # Generation number (1-based)
                    break
            else:
                speeds.append(len(run_scores))  # Didn't reach target
        
        if speeds:  # Only add if we have valid data
            convergence_speeds.append(np.mean(speeds))
            config_names.append(config_name)
        else:
            convergence_speeds.append(0)
            config_names.append(config_name)
    
    bars_conv = ax7.bar(range(len(config_names)), convergence_speeds, 
                       color='lightgreen', alpha=0.8)
    ax7.set_xticks(range(len(config_names)))
    ax7.set_xticklabels(config_names, rotation=45, ha='right', fontsize=12)
    ax7.set_ylabel('Avg Generations to 90% Performance', fontsize=14)
    ax7.set_title('Convergence Speed Analysis', fontsize=16, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    ax7.tick_params(axis='both', which='major', labelsize=12)
    
    # Add value labels
    for bar, speed in zip(bars_conv, convergence_speeds):
        height = bar.get_height()
        if height > 0:  # Only add label if we have valid data
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{speed:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Overall Summary Statistics
    ax8 = fig2.add_subplot(gs2[1, 1])
    ax8.axis('off')
    
    # Calculate summary statistics
    best_config_idx = np.argmax(avg_r2_scores)
    most_stable_idx = np.argmin(std_r2_scores)
    
    # Handle case where convergence_speeds might be empty or contain zeros
    valid_convergence = [s for s in convergence_speeds if s > 0]
    if valid_convergence:
        fastest_convergence_idx = convergence_speeds.index(min(valid_convergence))
        fastest_config = config_names[fastest_convergence_idx]
        fastest_speed = convergence_speeds[fastest_convergence_idx]
    else:
        fastest_config = "N/A"
        fastest_speed = 0
    
    summary_text = f"""
CROSS-CONFIGURATION SUMMARY

OVERALL STATISTICS
Total Configurations: {len(all_summaries)}
Best Average R²: {max(avg_r2_scores):.4f}
Worst Average R²: {min(avg_r2_scores):.4f}
Average Std Dev: {np.mean(std_r2_scores):.4f}

BEST PERFORMERS
Highest R²: {configs[best_config_idx]}
  Score: {avg_r2_scores[best_config_idx]:.4f}
Most Stable: {configs[most_stable_idx]}
  Std Dev: {std_r2_scores[most_stable_idx]:.4f}
Fastest Convergence: {fastest_config}
  Speed: {fastest_speed:.1f} generations

TOP 3 CONSISTENT FACTORS
{', '.join([truncate_factor_name(f[0], 15) for f in top_consistent[:3]])}

PERFORMANCE INSIGHTS
• Pop sizes tested: {sorted(list(set(pop_sizes)))}
• Generation counts: {sorted(list(set(generations)))}
• Runs per config: {len(all_factor_data[list(all_factor_data.keys())[0]]["per_gen_best"]) if all_factor_data else 0}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.suptitle('GA Configuration Comparison Dashboard 2/2', 
                fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()


def create_interactive_plotly_dashboard(all_summaries, all_factor_data):
    """
    Create interactive Plotly dashboard for web-based exploration
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available. Skipping interactive dashboard.")
        return
        
    print("\n🌐 Creating Interactive Plotly Dashboard")
    
    # Prepare data for interactive plots
    configs = []
    avg_r2 = []
    std_r2 = []
    pop_sizes = []
    generations = []
    best_factors_list = []
    
    for summary in all_summaries:
        configs.append(f"Pop{summary['Population Size']}_Gen{summary['Generations']}")
        avg_r2.append(summary['Average R²'])
        std_r2.append(summary['Std R²'])
        pop_sizes.append(summary['Population Size'])
        generations.append(summary['Generations'])
        best_factors_list.append(', '.join([truncate_factor_name(f, 10) for f in summary['Best Factors'][:3]]))  # Top 3 factors with truncation
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Configuration Performance', 'Population vs Generations Impact', 
                       'Convergence Comparison'),
        specs=[[{"secondary_y": True}, {"type": "scatter3d"}],
               [{"colspan": 2}, None]]
    )
    
    # 1. Performance bar chart with error bars
    fig.add_trace(
        go.Bar(
            x=configs, y=avg_r2, 
            error_y=dict(type='data', array=std_r2),
            name='Average R²',
            marker_color='rgba(55, 128, 191, 0.8)',
            text=[f'R²: {r:.3f}<br>Factors: {f}' for r, f in zip(avg_r2, best_factors_list)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Average R²: %{y:.4f}<br>Std Dev: %{error_y.array:.4f}<br>%{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=pop_sizes, y=generations, z=avg_r2,
            mode='markers',
            marker=dict(
                size=10,
                color=avg_r2,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Average R²", x=0.9)
            ),
            text=configs,
            name='Configurations',
            hovertemplate='<b>%{text}</b><br>Population: %{x}<br>Generations: %{y}<br>R²: %{z:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Convergence curves (combined plot)
    colors = px.colors.qualitative.Set3 if PLOTLY_AVAILABLE else ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
    for i, (key, data) in enumerate(all_factor_data.items()):
        pop_size = int(key.split("_")[0][3:])
        gen_count = int(key.split("_")[1][3:])
        config_name = f"Pop{pop_size}_Gen{gen_count}"
        
        # Calculate mean convergence (handle different lengths and filter invalid scores)
        if data["per_gen_best"]:
            valid_runs = []
            for scores in data["per_gen_best"]:
                # Filter out invalid scores
                valid_scores = [s for s in scores if s > -100]
                if valid_scores:
                    valid_runs.append(valid_scores)
            
            if valid_runs:
                max_len = max(len(scores) for scores in valid_runs)
                padded_runs = []
                for scores in valid_runs:
                    if len(scores) < max_len:
                        # Pad with final value if early stopping
                        padded = list(scores) + [scores[-1]] * (max_len - len(scores))
                        padded_runs.append(padded)
                    else:
                        padded_runs.append(scores)
                
                mean_convergence = np.mean(padded_runs, axis=0)
                generations_range = list(range(len(mean_convergence)))
            else:
                continue  # Skip if no valid data
        else:
            continue  # Skip if no data
        
        fig.add_trace(
            go.Scatter(
                x=generations_range, y=mean_convergence,
                mode='lines+markers',
                name=config_name,
                line=dict(color=colors[i % len(colors)]),
                hovertemplate=f'<b>{config_name}</b><br>Generation: %{{x}}<br>R²: %{{y:.4f}}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text="🚀 Interactive GA Performance Dashboard",
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        height=800
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Configurations", row=1, col=1)
    fig.update_yaxes(title_text="Average R²", row=1, col=1)
    fig.update_scenes(
        xaxis_title="Population Size",
        yaxis_title="Generations", 
        zaxis_title="Average R²",
        row=1, col=2
    )
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="R²", row=2, col=1)
    
    # Show interactive plot
    fig.show()
    
    # Save as HTML file
    fig.write_html("ga_dashboard_interactive.html")
    print("💾 Interactive dashboard saved as 'ga_dashboard_interactive.html'")


def visualize_and_ols(all_factor_data, X_all, y_all, target_col):
    """
    Create Individual Configuration Dashboards only (redundant simple convergence plot removed)
    """
    print("\n🎨 Creating Individual Configuration Dashboards...")
    
    for key, data in all_factor_data.items():
        # Create comprehensive dashboard for each configuration
        create_individual_config_dashboard(key, data, X_all, y_all, target_col)
    
    print("✅ Individual configuration dashboards completed")

# ==================================================
# -------------------- MAIN --------------------
# ==================================================
if __name__ == "__main__":
    print("🚀 Starting GA with Dashboard Visualizations")
    print("=" * 80)
    
    # STEP 1: Load dataset
    print("📊 Loading dataset...")
    df_data, X_all, y_all, usable, target_col = load_data()
    print(f"✅ Dataset loaded: {len(df_data)} rows, {len(usable)} factors")

    # STEP 2: Define GA experiment settings
    pop_sizes = [20, 50, 100, 500]         # population sizes to test
    generations = [20, 50, 100]       # number of generations
    runs_per_config = 2          # number of runs per config
    
    print(f"🔧 Experiment Settings:")
    print(f"   Population sizes: {pop_sizes}")
    print(f"   Generations: {generations}")
    print(f"   Runs per config: {runs_per_config}")
    print(f"   Chromosome size: {CHROM_SIZE}")

    # STEP 3: Run GA
    print("\n🧬 Running Genetic Algorithm...")
    all_summaries, all_factor_data = run_ga(X_all, y_all, usable, target_col, pop_sizes, generations, runs_per_config)

    # STEP 4: Create comprehensive dashboards
    print("\n🎨 Creating Comprehensive Visualization Dashboards...")
    
    # Individual configuration dashboards
    visualize_and_ols(all_factor_data, X_all, y_all, target_col)
    
    # Cross-configuration comparison dashboard
    create_cross_config_comparison_dashboard(all_summaries, all_factor_data)
    
    # Interactive Plotly dashboard
    create_interactive_plotly_dashboard(all_summaries, all_factor_data)

    # STEP 5: Print final summary
    print("\n\n📊📊📊 FINAL SUMMARY ACROSS ALL CONFIGURATIONS 📊📊📊")
    print("=" * 80)
    
    best_config = max(all_summaries, key=lambda x: x['Average R²'])
    most_stable = min(all_summaries, key=lambda x: x['Std R²'])
    
    print(f"🏆 BEST PERFORMANCE:")
    print(f"   Configuration: Pop{best_config['Population Size']}_Gen{best_config['Generations']}")
    print(f"   Average R²: {best_config['Average R²']:.4f}")
    print(f"   Best Factors: {best_config['Best Factors']}")
    
    print(f"\n📊 MOST STABLE:")
    print(f"   Configuration: Pop{most_stable['Population Size']}_Gen{most_stable['Generations']}")
    print(f"   Std Dev R²: {most_stable['Std R²']:.4f}")
    print(f"   Average R²: {most_stable['Average R²']:.4f}")
    
    print(f"\n📈 ALL CONFIGURATIONS:")
    for i, summary in enumerate(all_summaries, 1):
        print(f"\n   Config #{i}: Pop{summary['Population Size']}_Gen{summary['Generations']}")
        print(f"   Average R²: {summary['Average R²']:.4f} (±{summary['Std R²']:.4f})")
        print(f"   Best Factors: {', '.join([truncate_factor_name(f, 15) for f in summary['Best Factors'][:3]])}")
    
    print("\n" + "=" * 80)
    print("🎉 Analysis Complete! Check the generated dashboards for detailed insights.")
    print("💡 Key files generated:")
    print("   - Individual configuration dashboards (matplotlib)")
    print("   - Cross-configuration comparison dashboard")
    if PLOTLY_AVAILABLE:
        print("   - ga_dashboard_interactive.html (interactive plotly dashboard)")
    else:
        print("   - Interactive dashboard skipped (install plotly to enable)")
    print("=" * 80)