## Evaluation of Factor Models using genetic algorithms

## Table of Contents

1. [Project Structure ](#project-structure )  
2. [Project Description](#project-description)  
3. [Requirements](#requirements)  
4. [Setup Instructions](#setup-instructions)  
5. [Usage](#usage)  
7. [Output & Visualization](#output--visualization)  
8. [Notes](#notes)  

---

## Project Structure 

```plaintext
kushal
├── data/
│ ├── 1000Lines_ Clusters_Mappings.xlsm # Dataset with clusters and mappings
│ ├── query(new).xlsx                   # Query dataset for analysis
│ ├── 10000lines.csv                    # 10000lines dataset
│ └── F-F_Research_Data_Factors.csv     # Kenneth French 3-Factor Data
├── algorithms/
│ ├── geek_for_geek.py                  # Implementation of genetic algorithm (GeeksforGeeks reference)
│ ├── fama_french.py                    # Implementation of the Fama-French factor model
│ ├── genetic_numbers.py                # Genetic algorithm targeting arrays of numbers
│ └── query_und_1000lines.py            # Genetic algorithm on query(new).xlsx and 1000Lines dataset
├── .gitignore                          # Excludes unnecessary files (venv, etc)
├── README.md                           # Project overview and usage instructions
├── requirements.txt                    # List of required Python dependencies
└── description.pdf                     # Technical explanation document
```

---


## Project Description

The program query_und_1000lines.py uses Genetic Algorithms (GAs) to identify the best subset of factors from a list of large data (1000Lines_ Clusters_Mappings.xlsm or query(new).xlsx) in order to have the best explanatory power (R²) in a regression model. The algorithm draws its inspiration from natural selection and mimics the evolutionary processes of "survival of the fittest", but applied to sets of factors rather than to living things.

The script query_und_1000lines.py performs the following steps:

1. Loads financial factor data from a query(new).xksx or 1000Lines_ Clusters_Mappings.xlsm.
2. Processes data (uppercasing columns, filtering numeric factors, eliminating missing values).
3. Runs a Genetic Algorithm (GA) with various population sizes and generations, each for 3 runs to find best combinations of factors.
4. Optimizes each combination by **R²**.
5. Plots:
    - GA run convergence
    - Highest factor importance (average R²)
    - Significant factor coefficients (p-values)
6. Gives an overall summary of top-performing factor combinations for all setups in the terminal.

--

## Requirements

1. The project is developed in **Python 3.13.5+** but it should work from Python 3.7, as recent pandas/numpy versions have dropped 3.6 support:
2. For smooth installation of the libraries, **pip ≥ 20** is recommended.

--

## Setup Instructions

# Clone the Project
Use the command below to clone this repository in the local system.
```bash
    #Using SSH
    git clone -b kushal --single-branch git@git-ce.rwth-aachen.de:factorzoo/Group_1.git
    cd Group_1

    #Using HTTPS
    git clone -b kushal --single-branch https://git-ce.rwth-aachen.de/factorzoo/Group_1.git
    cd Group_1
```

# Virtual Environment (recommended)
If you want to work by creating a virtual environment to isolate our project’s Python packages, so they don’t conflict with other projects and can be reproduced easily, you can follow the commands below:

1. Create a virtual environment
```bash
python -m venv venv
```

2. Activate the virtual environment
# Windows:
```bash
venv\Scripts\activate
```
# Linux/macOS:
```bash
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. (Optional) Freeze current packages to requirements.txt
```bash
pip freeze > requirements.txt
```

# Without Virtual Environment
You can also work without virtual environment but it installs all required packages globally. You can do so, by following the commands bellow:
```bash
pip install pandas numpy statsmodels matplotlib seaborn openpyxl

#Run this if you are running fama_french.py
pip install yfinance


#you can run this to get aditional interactive dashboard (optional)
pip install plotly 
```

--

## Usage 

1. Activate your virtual environment:
```bash
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
```
2. Run the script
```bash
    python query_und_1000lines.py
```
3. Change the value of USE_QUERY_DATA to use query(new).xlsx or 1000Lines_ Clusters_Mappings.xlsm
USE_QUERY_DATA = False  # Set True to use query(new).xlsx, False for 1000Lines_ Clusters_Mappings.xlsm

4. (Optional) Modify parameters like CHROM_SIZE, POP_SIZE, GENERATIONS or runs_per_config in the script to test different configurations.
```bash
    CHROM_SIZE = 3
    pop_sizes = [50, 200]
    generations = [50, 200]
    runs_per_config = 3
```
5. Change the mutation rate (rate=0.2) to experiment
```bash
    def mutate(chrom, usable, rate=0.2):
```

--

## Output & Visualization

# Console output:

1. Shows generation by generation progress, R sq. value of the chromosome and the factors associated to it, for every (POP_SIZE, GENERATIONS) combination and across different runs_per_config.
2. Summary of the config with Average R², Std R² and Best factors across runs.
3. Before the SINGLE plot of CONVERGENCE(in vizualiation phase) for every (POP_SIZE, GENERATIONS) combination, OLS Summary printed.
4. At the end of the vizualization, summary of the latest run per (POP_SIZE, GENERATIONS) is printed with the following values:
    Config #number
    Population Size: 
    Generations: 
    Average R²:
    Std R²:
    Best Factors:


# Vizualization

View in full screen for better experience
For each Population size, Generation combination

Dashboard 1 (Performance Analysis)

- Convergence Plot - Line chart showing R² progression across generations for each run 
- Performance Statistics - Histogram of final R² distribution with mean/median lines 
- Learning Curve Analysis - Line chart with mean R² ± std deviation across generations 
- Configuration Summary - Text summary box with key metrics

Dashboard 2 (Factor Analysis)

- Factor Importance Analysis - Bar chart showing factor selection frequency 
- Factor Performance Boxplot - Box plots of R² scores for top factors 
- Factor P-value Analysis - Bar chart of average p-values with significance thresholds 
- Optimization Efficiency - Bar chart of improvements per run

Dashboard 3 (OLS Regression Analysis Dashboard)

- Regression Coefficients - Horizontal bar chart with significance markers
- Residuals vs Fitted - Scatter plot for residual analysis
- Q-Q Plot - Quantile-quantile plot for normality check
- Model Statistics - Text box with regression statistics

Cross-Configuration Comparison Dashboards

Dashboard 1 (Performance Comparison)

- Configuration Performance Comparison - Bar chart with error bars comparing all configs
- Average vs Best Performance - Grouped bar chart showing average vs best R²
- Population Size Impact - Box plots showing performance by population size
- Generation Count Impact - Box plots showing performance by generation count

Dashboard 2 (Other Analysis)

- Performance Heatmap - 2D heatmap of population size vs generations
- Top Most Consistent Factors - Bar chart of factors selected across configs
- Convergence Speed Analysis - Bar chart of generations to reach 90% performance
- Overall Summary Statistics - Text summary box

Interactive Plotly Dashboard:

- Configuration Performance - Interactive bar chart with error bars
- Population vs Generations Impact - 3D scatter plot
- Convergence Comparison - Interactive line chart with multiple series

--

## Notes

- **Data files required**:
  - If `USE_QUERY_DATA=True`, the script expects `../data/query(new).xlsx`.  
  - If `USE_QUERY_DATA=False`, the script expects `../data/1000Lines_ Clusters_Mappings.xlsm` with sheets:
    - `1000Lines`
    - `Groupings`

- **Excel dependency**: Reading `.xlsx` and `.xlsm` files requires `openpyxl`. Install them if not already available (if steps are followed above, it is installed):
```bash
  pip install openpyxl xlrd
```
- Ensure the dataset paths in the script are correct before running.
- Large population sizes and generations may increase runtime significantly.
- If not using a virtual environment, install the packages globally as described in the installation section.
- For Fama-French, internet access is required to download data via `yfinance`.
- Python 3.7 or higher is recommended for compatibility.
