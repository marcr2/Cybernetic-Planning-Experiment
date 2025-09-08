# Cybernetic Central Planning System

An AI-enhanced central planning software system that generates comprehensive 5-year economic plans using Input-Output analysis and labor-time accounting. This system implements the principles of cybernetic economic planning, integrating the Input-Output models of Wassily Leontief with the labor-time accounting methods proposed by W. Paul Cockshott and Allin Cottrell.

## Features

- **Mathematical Foundation**: Implements Leontief Input-Output models, labor value calculations, and constrained optimization
- **Multi-Agent AI System**: Specialized agents for economics, resources, policy, and report generation
- **Dynamic Planning**: 5-year economic planning with capital accumulation and technological change
- **Data Processing**: Comprehensive I-O table parsing and validation
- **Report Generation**: Detailed markdown reports with mathematical transparency
- **Visualization**: Charts and graphs for economic analysis

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd "Central Planning Experiment"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On Windows
   py -m venv .venv
   .venv\Scripts\activate

   # On Unix/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Using the GUI

1. **Run the GUI**:
   ```bash
   .\run_gui.bat
   ```

2. **Load Data**:
   - Click "Process USA Zip File" to process USA I-O data
   - Click "Load from File" to load existing processed data
   - Click "Generate Synthetic Data" for testing

3. **Create Plans**:
   - Enter policy goals in the Planning Configuration tab
   - Click "Create Plan" to generate economic plans
   - View results in the Results & Analysis tab

### Basic Usage

```python
from src.cybernetic_planning.planning_system import CyberneticPlanningSystem

# Initialize the planning system
system = CyberneticPlanningSystem()

# Create synthetic data for testing
data = system.create_synthetic_data(n_sectors=8, technology_density=0.4)

# Create an economic plan
policy_goals = [
    "Increase healthcare capacity by 15%",
    "Reduce carbon emissions by 20%",
    "Improve education infrastructure"
]

plan = system.create_plan(policy_goals=policy_goals)

# Generate a comprehensive report
report = system.generate_report()
print(report)
```

### Processing Real Data

```python
# Process USA Input-Output data from zip file
from usa_zip_processor import USAZipProcessor

processor = USAZipProcessor()
result_file = processor.process_zip_file("usa_data.zip")
system.load_data_from_file(result_file)
```

This will:
1. Automatically detect USA I-O files in the zip
2. Select the most recent year (2024)
3. Process USE, Final Demand, and Make tables
4. Create a ready-to-use JSON file in the data/ folder
5. Load the data into the planning system

## Project Structure

```
Central Planning Experiment/
├── src/
│   └── cybernetic_planning/
│       ├── core/                    # Core mathematical algorithms
│       │   ├── leontief.py         # Leontief Input-Output model
│       │   ├── labor_values.py     # Labor value calculations
│       │   ├── optimization.py     # Constrained optimization
│       │   └── dynamic_planning.py # Multi-year planning
│       ├── agents/                  # Multi-agent AI system
│       │   ├── base.py             # Base agent class
│       │   ├── manager.py          # Central coordinator
│       │   ├── economics.py        # Economic analysis agent
│       │   ├── resource.py         # Resource management agent
│       │   ├── policy.py           # Policy analysis agent
│       │   └── writer.py           # Report generation agent
│       ├── data/                    # Data processing utilities
│       │   ├── io_parser.py        # I-O table parsing
│       │   ├── matrix_builder.py   # Matrix construction
│       │   └── data_validator.py   # Data validation
│       ├── utils/                   # Utility functions
│       │   ├── visualization.py    # Chart generation
│       │   └── helpers.py          # Helper functions
│       └── planning_system.py      # Main system orchestrator
├── tests/                          # Unit tests
├── data/                           # Data storage
├── outputs/                        # Generated reports and visualizations
├── design_documents/               # Project specifications
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Core Components

### 1. Mathematical Foundation

- **Leontief Model**: Solves the equation `x = (I - A)^(-1) * d` for total output
- **Labor Values**: Calculates total direct and indirect labor using `v = l(I - A)^(-1)`
- **Constrained Optimization**: Minimizes labor cost subject to demand and resource constraints
- **Dynamic Planning**: Handles multi-year planning with capital accumulation

### 2. Multi-Agent System

- **Manager Agent**: Central coordinator for the planning process
- **Economics Agent**: Performs sensitivity analysis and economic forecasting
- **Resource Agent**: Manages resource constraints and environmental impact
- **Policy Agent**: Translates natural language goals into quantitative adjustments
- **Writer Agent**: Generates comprehensive markdown reports

### 3. Data Processing

- **I-O Parser**: Supports CSV, Excel, and JSON formats
- **Matrix Builder**: Constructs and validates economic matrices
- **Data Validator**: Ensures data consistency and mathematical properties

## Mathematical Models

### Leontief Input-Output Model

The system solves the fundamental equation:
```
x = Ax + d
```

Where:
- `x` = total output vector
- `A` = technology matrix (input coefficients)
- `d` = final demand vector

### Labor Value Calculation

Following Cockshott's model:
```
v = vA + l
```

Where:
- `v` = labor value vector
- `l` = direct labor input vector

### Constrained Optimization

Minimizes total labor cost:
```
min l · x
```

Subject to:
- `(I - A)x ≥ d` (demand fulfillment)
- `Rx ≤ R_max` (resource constraints)
- `x ≥ 0` (non-negativity)

## Usage Examples

### Creating a 5-Year Plan

```python
# Create a comprehensive 5-year plan
five_year_plan = system.create_five_year_plan(
    policy_goals=[
        "Increase healthcare capacity by 15%",
        "Reduce carbon emissions by 20%",
        "Improve education infrastructure"
    ],
    consumption_growth_rate=0.02,
    investment_ratio=0.2
)

# Access individual year plans
for year, plan in five_year_plan.items():
    print(f"Year {year}: Total Output = {plan['total_output'].sum():.2f}")
```

### Loading Real Data

```python
# Load data from file
data = system.load_data_from_file("input_output_table.csv")

# Or load from dictionary
data = {
    'technology_matrix': np.array([[0.1, 0.2], [0.3, 0.1]]),
    'final_demand': np.array([100, 200]),
    'labor_input': np.array([0.5, 0.8]),
    'resource_matrix': np.array([[1.0, 0.5], [0.3, 1.2]]),
    'max_resources': np.array([1000, 800])
}

system.load_data_from_dict(data)
```

### Generating Visualizations

```python
from src.cybernetic_planning.utils.visualization import create_plan_visualizations

# Create comprehensive visualizations
visualizations = create_plan_visualizations(plan_data, "outputs/visualizations")

print("Generated visualizations:")
for name, path in visualizations.items():
    print(f"- {name}: {path}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/cybernetic_planning

# Run specific test file
python -m pytest tests/test_core.py -v
```

## Configuration

The system can be configured through the `pyproject.toml` file or by passing a configuration dictionary:

```python
config = {
    'max_iterations': 15,
    'convergence_threshold': 0.005,
    'optimization_solver': 'ECOS'
}

system = CyberneticPlanningSystem(config)
```

## Output Formats

### Markdown Reports

The system generates comprehensive markdown reports including:
- Executive Summary with key metrics
- Sector-by-Sector Analysis
- Resource Allocation details
- Labor Budget breakdown
- Risk Assessment and sensitivity analysis

### Data Export

Plans can be exported in multiple formats:
- JSON: `system.save_plan("plan.json", "json")`
- CSV: `system.save_plan("plan.csv", "csv")`
- Excel: `system.save_plan("plan.xlsx", "excel")`

## Advanced Features

### Custom Policy Goals

The system supports natural language policy goals that are automatically translated into quantitative adjustments:

```python
policy_goals = [
    "Increase healthcare capacity by 15%",
    "Reduce carbon emissions by 20%",
    "Ensure minimum food production of 2200 calories per capita",
    "Improve education infrastructure",
    "Increase employment by 8%"
]
```

### Sensitivity Analysis

The economics agent performs comprehensive sensitivity analysis:
- Technology matrix sensitivity: `∂x/∂A_ij`
- Demand sensitivity: `∂x/∂d`
- Critical sector identification
- Supply chain analysis

### Resource Management

The resource agent handles:
- Resource constraint optimization
- Environmental impact assessment
- Sustainability analysis
- Resource substitution opportunities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Leontief, W. (1986). Input-Output Economics. Oxford University Press.
- Cockshott, W. P., & Cottrell, A. (1993). Towards a New Socialism. Spokesman Books.
- Cockshott, W. P., & Cottrell, A. (2018). Economic planning in a market economy. Science & Society, 82(2), 159-186.

## Support

For questions, issues, or contributions, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Contact the development team

---

*This system implements cutting-edge cybernetic planning principles for modern economic planning and analysis.*
