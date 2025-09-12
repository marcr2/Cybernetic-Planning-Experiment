# Synthetic Sector Generation System

## Overview

The Synthetic Sector Generation System is a dynamic economic sector management system designed for economic simulations. It creates and manages economic sectors that evolve over time based on technological progress, ensuring all sector names come from the predefined `sectors.md` file.

## Key Features

### ✅ Core Requirements Met

1. **Sector Naming Convention**: All sectors MUST have names sourced from `sectors.md`
2. **Sector Count Constraints**: Minimum 6 sectors, maximum 1000 sectors
3. **Initial Sector Setup**: First 6 sectors are mandatory (Healthcare, Food and Agriculture, Energy, Housing and Construction, Education, Transportation)
4. **Subdivision Logic**: Core sectors can be subdivided if user limit allows
5. **Dynamic Sector Evolution**: New sectors generated based on technological progress
6. **Employment Integration**: New sectors integrate with employment and economic planning

### 🚀 Advanced Features

- **Technological Breakthroughs**: AI Revolution, Quantum Computing, Space Technology, Biotechnology, Renewable Energy
- **Employment Management**: Automatic employment distribution and unemployment tracking
- **Technology Tree**: Prerequisites and dependencies between sectors
- **Research Investment**: Technology advancement through research spending
- **Population Dynamics**: Employment scaling with population changes

## Quick Start

### Basic Usage

```python
from src.cybernetic_planning.data.synthetic_sector_generator import SyntheticSectorGenerator

# Create generator with default settings (6-1000 sectors)
generator = SyntheticSectorGenerator()

# Initialize sectors
sectors = generator.initialize_sectors()

# Get basic information
print(f"Total sectors: {generator.get_sector_count()}")
print(f"Core sectors: {len(generator.get_core_sectors())}")
```

### Custom Configuration

```python
# Create generator with custom constraints
generator = SyntheticSectorGenerator(
    max_sectors=100,    # Maximum 100 sectors
    min_sectors=10,    # Minimum 10 sectors
    sectors_file_path="path/to/sectors.md"  # Custom sectors file
)
```

### Simulation Over Time

```python
# Simulate 5 years with research investment
for year in range(5):
    research_investment = 1000000 * (year + 1)  # Increasing investment
    generator.advance_simulation_year(research_investment)
    
    print(f"Year {generator.current_year}: "
          f"{generator.get_sector_count()} sectors, "
          f"Tech Level: {generator.technological_level:.2f}")
```

## Integration with Existing System

### Using SectorIntegrationManager

The `SectorIntegrationManager` provides a unified interface for synthetic sector generation:

```python
from src.cybernetic_planning.data.sector_integration import (
    SectorIntegrationManager,
    create_synthetic_sector_manager
)

# Synthetic mode (the only supported mode)
synthetic_manager = create_synthetic_sector_manager(max_sectors=100)
synthetic_manager.initialize_sectors()
```

## API Reference

### SyntheticSectorGenerator

#### Constructor
```python
SyntheticSectorGenerator(
    sectors_file_path: Optional[str] = None,
    max_sectors: int = 1000,
    min_sectors: int = 6
)
```

#### Key Methods

##### Sector Information
- `get_sector_count() -> int`: Get total number of sectors
- `get_sector_names() -> List[str]`: Get list of all sector names
- `get_core_sectors() -> List[SectorDefinition]`: Get the 6 core sectors
- `get_sectors_by_technology_level(level: TechnologyLevel) -> List[SectorDefinition]`: Get sectors by tech level

##### Simulation Control
- `advance_simulation_year(research_investment: float = 0.0)`: Advance simulation by one year
- `update_population(new_population: float)`: Update total population

##### Employment Data
- `get_employment_by_sector() -> Dict[int, float]`: Get employment by sector
- `get_total_employment() -> float`: Get total employment
- `get_unemployment_rate() -> float`: Get unemployment rate

##### Data Export
- `export_sector_data() -> Dict[str, Any]`: Export comprehensive sector data
- `print_sector_summary()`: Print formatted summary

### SectorDefinition

Each sector has the following properties:

```python
@dataclass
class SectorDefinition:
    id: int                                    # Unique sector ID
    name: str                                  # Sector name (from sectors.md)
    category: SectorCategory                   # Main category
    subcategory: str                           # Subcategory
    technology_level: TechnologyLevel          # Tech level (BASIC to FUTURE)
    prerequisites: List[int]                   # Required sectors
    unlocks: List[int]                        # Sectors this unlocks
    description: str                          # Description
    importance_weight: float                  # Importance (0.0-1.0)
    economic_impact: str                      # 'critical', 'high', 'medium', 'low'
    labor_intensity: str                       # 'high', 'medium', 'low'
    capital_intensity: str                    # 'very_high', 'high', 'medium', 'low'
    environmental_impact: str                  # 'very_high', 'high', 'medium', 'low'
    development_cost: float                   # Cost to develop
    research_requirements: List[str]          # Required research areas
    employment_capacity: float                 # Workers this sector can employ
    is_core_sector: bool                      # Is this a core sector?
    is_subdivision: bool                      # Is this a subdivision?
    parent_sector_id: Optional[int]           # Parent sector ID if subdivision
    creation_year: Optional[int]               # Year created
    technological_breakthrough_required: bool  # Requires breakthrough?
```

## Technology Levels

The system uses 5 technology levels:

1. **BASIC**: Available from start (core sectors)
2. **INTERMEDIATE**: Requires basic sectors (subdivisions)
3. **ADVANCED**: Requires intermediate sectors
4. **CUTTING_EDGE**: Requires advanced sectors
5. **FUTURE**: Requires cutting-edge sectors

## Technological Breakthroughs

The system includes 5 major technological breakthroughs:

1. **Artificial Intelligence Revolution**: Unlocks AI-related sectors
2. **Quantum Computing Breakthrough**: Unlocks quantum technology sectors
3. **Space Technology Revolution**: Unlocks space-related sectors
4. **Biotechnology Revolution**: Unlocks biotech sectors
5. **Renewable Energy Revolution**: Unlocks clean energy sectors

Breakthroughs are achieved when:
- Technological level reaches required threshold
- Sufficient research investment has been made

## Employment System

### Automatic Employment Distribution

- **Core Sectors**: Each employs ~15% of population
- **Subdivisions**: Each employs ~1% of population
- **New Sectors**: Each employs ~1% of population
- **Labor Intensity**: High labor sectors get 1.5x capacity, low labor sectors get 0.5x capacity

### Employment Scaling

Employment automatically scales with population changes:

```python
# Update population (employment scales proportionally)
generator.update_population(2000000)  # Double population
```

## Examples

### Example 1: Basic Sector Generation

```python
from src.cybernetic_planning.data.synthetic_sector_generator import SyntheticSectorGenerator

# Create generator
generator = SyntheticSectorGenerator(max_sectors=50, min_sectors=6)

# Print initial summary
generator.print_sector_summary()

# Output:
# Synthetic Sector Generation Summary
# Total Sectors: 48
# Core Sectors: 6
# Current Year: 2024
# Technological Level: 0.00
# Population: 1,000,000
# Total Employment: 950,000
# Unemployment Rate: 5.0%
```

### Example 2: Multi-Year Simulation

```python
# Simulate 10 years with increasing research investment
for year in range(10):
    research_investment = 500000 * (year + 1)
    generator.advance_simulation_year(research_investment)
    
    if generator.achieved_breakthroughs:
        print(f"Breakthroughs achieved in {generator.current_year}:")
        for breakthrough in generator.achieved_breakthroughs:
            print(f"  - {breakthrough}")

# Print final summary
generator.print_sector_summary()
```

### Example 3: Employment Analysis

```python
# Get employment data
employment_data = generator.get_employment_by_sector()

# Find sectors with highest employment
sorted_employment = sorted(
    employment_data.items(), 
    key=lambda x: x[1], 
    reverse=True
)

print("Top 10 Sectors by Employment:")
for sector_id, employment in sorted_employment[:10]:
    sector_name = generator.sectors[sector_id].name
    print(f"  {sector_name}: {employment:,.0f} workers")
```

### Example 4: Technology Tree Analysis

```python
# Get sectors that can be unlocked
developed_sectors = {sector.id for sector in generator.get_core_sectors()}
unlocked_sectors = generator.get_unlocked_sectors(developed_sectors)

print(f"Sectors that can be unlocked: {len(unlocked_sectors)}")
for sector_id in unlocked_sectors:
    sector = generator.sectors[sector_id]
    print(f"  {sector.name} (Tech Level: {sector.technology_level.value})")
```

## Testing

The system includes comprehensive tests:

```bash
# Run all tests
python tests/test_synthetic_sector_generator.py

# Run with verbose output
python -m unittest tests.test_synthetic_sector_generator -v
```

### Test Coverage

- ✅ Sector naming convention validation
- ✅ Sector count constraints (6-1000)
- ✅ Core sectors mandatory requirement
- ✅ Subdivision logic validation
- ✅ Dynamic sector evolution testing
- ✅ Employment integration testing
- ✅ Technology level distribution
- ✅ Sector properties validation
- ✅ Technological breakthrough system
- ✅ Export functionality
- ✅ Population update functionality
- ✅ Constraint validation
- ✅ Sector unlocking mechanism

## Configuration Options

### Sector Count Constraints

```python
# Minimum sectors (must be >= 6)
min_sectors = 6

# Maximum sectors (must be <= 1000)
max_sectors = 1000

# Examples:
generator = SyntheticSectorGenerator(max_sectors=50, min_sectors=6)   # 6-50 sectors
generator = SyntheticSectorGenerator(max_sectors=100, min_sectors=10)  # 10-100 sectors
generator = SyntheticSectorGenerator(max_sectors=1000, min_sectors=6)  # 6-1000 sectors
```

### Custom Sectors File

```python
# Use custom sectors.md file
generator = SyntheticSectorGenerator(
    sectors_file_path="path/to/custom_sectors.md"
)
```

## Integration with Economic Planning

### With Dynamic Planning System

```python
from src.cybernetic_planning.core.dynamic_planning import DynamicPlanner
from src.cybernetic_planning.data.synthetic_sector_generator import SyntheticSectorGenerator

# Create sector generator
generator = SyntheticSectorGenerator(max_sectors=100)

# Create dynamic planner with generated sectors
planner = DynamicPlanner(
    n_sectors=generator.get_sector_count(),
    # ... other parameters
)

# Advance both systems together
for year in range(5):
    # Advance sector generation
    generator.advance_simulation_year(1000000)
    
    # Update planner with new sectors if any were added
    if generator.get_sector_count() > planner.n_sectors:
        # Expand planner to accommodate new sectors
        planner.expand_sectors(generator.get_sector_count())
```

### With Marxist Reproduction System

```python
from src.cybernetic_planning.core.marxist_reproduction import MarxistReproductionSystem

# Create sector generator
generator = SyntheticSectorGenerator(max_sectors=175)  # 50+50+75 for departments

# Create Marxist reproduction system
marxist_system = MarxistReproductionSystem(
    technology_matrix=technology_matrix,
    final_demand=final_demand,
    labor_vector=labor_vector,
    n_dept_I=50,
    n_dept_II=50,
    n_dept_III=75
)

# Integrate employment data
employment_data = generator.get_employment_by_sector()
# Use employment data in Marxist calculations...
```

## Performance Considerations

### Memory Usage

- **Small Scale** (6-50 sectors): ~1-5 MB memory
- **Medium Scale** (50-200 sectors): ~5-20 MB memory  
- **Large Scale** (200-1000 sectors): ~20-100 MB memory

### Computation Time

- **Initial Generation**: < 1 second for any scale
- **Year Advancement**: < 0.1 seconds per year
- **Breakthrough Processing**: < 0.1 seconds per breakthrough

### Optimization Tips

1. **Use appropriate sector limits**: Don't generate 1000 sectors if you only need 50
2. **Batch research investments**: Advance multiple years at once rather than one by one
3. **Cache sector data**: Export and reuse sector data rather than regenerating

## Troubleshooting

### Common Issues

#### Issue: "Sectors file not found"
**Solution**: Ensure `sectors.md` exists in the data directory or provide full path:
```python
generator = SyntheticSectorGenerator(
    sectors_file_path="full/path/to/sectors.md"
)
```

#### Issue: "Invalid sector count constraints"
**Solution**: Ensure min_sectors >= 6 and max_sectors <= 1000:
```python
generator = SyntheticSectorGenerator(
    min_sectors=6,    # Must be >= 6
    max_sectors=1000   # Must be <= 1000
)
```

#### Issue: "No sectors generated"
**Solution**: Check that `sectors.md` contains properly formatted sector definitions:
```
1. Healthcare
2. Food and Agriculture
3. Energy
...
```

#### Issue: "Employment not updating"
**Solution**: Ensure you're using synthetic mode and calling `advance_simulation_year()`:
```python
# Only synthetic mode supports employment updates
generator = SyntheticSectorGenerator()  # Uses synthetic mode by default
generator.advance_simulation_year(1000000)  # This updates employment
```

### Debug Mode

Enable debug output by setting environment variable:
```bash
export DEBUG_SECTOR_GENERATION=1
python your_script.py
```

## Future Enhancements

### Planned Features

1. **Sector Merging**: Ability to merge sectors based on economic conditions
2. **Regional Sectors**: Geographic distribution of sectors
3. **Sector Lifecycle**: Sectors can become obsolete and disappear
4. **Advanced Dependencies**: More complex prerequisite relationships
5. **Sector Specialization**: Sectors can specialize over time
6. **Economic Indicators**: Integration with economic performance metrics

### Extension Points

The system is designed to be extensible:

1. **Custom Breakthroughs**: Add new technological breakthroughs
2. **Custom Categories**: Define new sector categories
3. **Custom Employment Models**: Implement different employment distribution algorithms
4. **Custom Technology Trees**: Define custom prerequisite relationships

## Contributing

When contributing to the synthetic sector generation system:

1. **Follow Requirements**: Ensure all requirements are met
2. **Add Tests**: Include comprehensive tests for new features
3. **Update Documentation**: Keep this documentation current
4. **Validate Constraints**: Ensure sector count and naming constraints are respected
5. **Performance**: Consider performance implications of changes

## License

This system is part of the Cybernetic Planning Experiment project and follows the same license terms.
