# Resource Data Collection System

This module provides comprehensive web scraping and data processing capabilities for collecting real resource constraint data from various government agencies and data sources to replace synthetic data in the cybernetic planning system.

## Overview

The Resource Data Collection System is designed to collect real-world data on:
- **Energy Resources**: Coal, natural gas, petroleum, nuclear, and renewable energy consumption by sector
- **Material Resources**: Critical materials including lithium, cobalt, rare earth elements, copper, aluminum, and steel
- **Labor Resources**: Employment, wages, and labor intensity by skill category and sector
- **Environmental Resources**: Carbon emissions, water usage, and waste generation by sector

## Architecture

```
src/cybernetic_planning/data/
├── web_scrapers/           # Web scraping modules
│   ├── base_scraper.py     # Base scraper with common functionality
│   ├── eia_scraper.py      # Energy Information Administration scraper
│   ├── usgs_scraper.py     # US Geological Survey scraper
│   ├── bls_scraper.py      # Bureau of Labor Statistics scraper
│   ├── epa_scraper.py      # Environmental Protection Agency scraper
│   └── data_collector.py   # Main data collection orchestrator
├── sector_mapper.py        # Sector mapping and data integration
├── resource_matrix_builder.py  # Resource constraint matrix builder
├── enhanced_data_loader.py # Enhanced data loader with real data integration
└── README.md              # This documentation
```

## Key Features

### 1. Real Data Collection Only
- **No Synthetic Data**: The system never generates synthetic data to replace missing real data
- **Empty Data Handling**: Missing data is properly marked as unavailable rather than estimated
- **Data Quality Tracking**: Comprehensive quality assessment and reporting

### 2. Multi-Source Data Integration
- **EIA (Energy)**: Energy consumption and intensity data by sector
- **USGS (Materials)**: Mineral production, consumption, and critical materials assessment
- **BLS (Labor)**: Employment, wages, and labor intensity by occupation and sector
- **EPA (Environmental)**: Carbon emissions, water usage, and waste generation data

### 3. BEA Sector Mapping
- **175-Sector Classification**: Maps all collected data to BEA 175-sector classification
- **Cross-Walking**: Handles different sector classification systems
- **Missing Data Handling**: Properly marks sectors without data rather than estimating

### 4. Resource Constraint Matrices
- **Energy Matrix**: (5 energy types × 175 sectors)
- **Material Matrix**: (6 critical materials × 175 sectors)
- **Labor Matrix**: (5 skill categories × 175 sectors)
- **Environmental Matrix**: (3 environmental factors × 175 sectors)
- **Combined Matrix**: All resource constraints in a single matrix

## Usage

### Basic Usage

```python
from src.cybernetic_planning.data.enhanced_data_loader import EnhancedDataLoader

# Initialize the enhanced data loader
loader = EnhancedDataLoader(
    eia_api_key="your_eia_api_key",  # Optional
    data_dir="data",
    cache_dir="cache"
)

# Load comprehensive data with real resource constraints
data = loader.load_comprehensive_data(
    year=2024,
    use_real_data=True
)

# Access resource matrices
energy_matrix = data['resource_matrices']['energy_matrix']
material_matrix = data['resource_matrices']['material_matrix']
labor_matrix = data['resource_matrices']['labor_matrix']
environmental_matrix = data['resource_matrices']['environmental_matrix']
```

### Integration with Planning System

```python
from src.cybernetic_planning.planning_system import CyberneticPlanningSystem

# Initialize the planning system
system = CyberneticPlanningSystem()

# Load comprehensive data with real resource constraints
data = system.load_comprehensive_data(
    year=2024,
    use_real_data=True,
    eia_api_key="your_eia_api_key"
)

# Create a plan with real resource constraints
plan = system.create_plan(
    policy_goals=[
        "Increase renewable energy production by 20%",
        "Reduce carbon emissions by 15%",
        "Improve labor productivity in manufacturing"
    ]
)
```

### Individual Scraper Usage

```python
from src.cybernetic_planning.data.web_scrapers import EIAScraper, USGSScraper, BLSScraper, EPAScraper

# Initialize scrapers
eia_scraper = EIAScraper(api_key="your_eia_api_key")
usgs_scraper = USGSScraper()
bls_scraper = BLSScraper()
epa_scraper = EPAScraper()

# Collect data from specific sources
energy_data = eia_scraper.scrape_all_energy_data(year=2024)
material_data = usgs_scraper.scrape_all_material_data(year=2024)
labor_data = bls_scraper.scrape_all_labor_data(year=2024)
environmental_data = epa_scraper.scrape_all_environmental_data(year=2024)
```

## Data Sources

### Energy Information Administration (EIA)
- **Base URL**: https://api.eia.gov/v2
- **Data Types**: Energy consumption, electricity consumption, energy intensity, renewable energy
- **API Key**: Required for enhanced access (optional for basic data)
- **Rate Limit**: 5000 requests per hour

### US Geological Survey (USGS)
- **Base URL**: https://minerals.usgs.gov/minerals/pubs/commodity
- **Data Types**: Mineral production, material consumption, critical materials assessment
- **Rate Limit**: Conservative (1 request per second)

### Bureau of Labor Statistics (BLS)
- **Base URL**: https://api.bls.gov/publicAPI/v2
- **Data Types**: Employment, wages, labor intensity, occupational skills
- **Rate Limit**: 1 request per second

### Environmental Protection Agency (EPA)
- **Base URL**: https://www.epa.gov/enviro
- **Data Types**: Carbon emissions, water usage, waste generation, environmental intensity
- **Rate Limit**: 1 request per second

## Data Quality and Validation

### Quality Metrics
- **Data Completeness**: Percentage of sectors with available data
- **Source Reliability**: Government agencies > Academic institutions > Industry associations
- **Temporal Coverage**: Data year and collection timestamps
- **Unit Consistency**: Normalized units across all data sources

### Validation Process
1. **Source Validation**: Verify data comes from authoritative sources
2. **Format Validation**: Ensure data matches expected structure
3. **Range Validation**: Check for reasonable value ranges
4. **Consistency Validation**: Cross-reference with known economic relationships

### Missing Data Handling
- **No Estimation**: Missing data is never estimated or synthesized
- **Explicit Marking**: Missing sectors are clearly marked as unavailable
- **Quality Reporting**: Comprehensive reporting of data availability and quality

## Configuration

### Environment Variables
```bash
# Optional EIA API key for enhanced data access
export EIA_API_KEY="your_eia_api_key"

# Data directories
export DATA_DIR="data"
export CACHE_DIR="cache"
```

### Configuration Files
The system uses configuration files for sector mapping and data source settings. These are located in the `config/` directory and can be customized for different data sources or sector classifications.

## Error Handling

### Common Issues
1. **API Rate Limits**: Automatic retry with exponential backoff
2. **Network Timeouts**: Configurable timeout settings
3. **Data Format Changes**: Robust parsing with fallback options
4. **Missing Data Sources**: Graceful handling of unavailable data

### Logging
The system provides comprehensive logging at multiple levels:
- **DEBUG**: Detailed scraping and processing information
- **INFO**: General progress and status updates
- **WARNING**: Non-critical issues and data quality concerns
- **ERROR**: Critical errors that prevent data collection

## Performance Considerations

### Caching
- **Response Caching**: API responses are cached for 24 hours
- **Data Persistence**: Collected data is saved to disk for reuse
- **Incremental Updates**: Only collect new data when needed

### Rate Limiting
- **Respectful Scraping**: All scrapers respect rate limits
- **Exponential Backoff**: Automatic retry with increasing delays
- **Request Queuing**: Manage multiple data source requests efficiently

## Dependencies

### Required Packages
```
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
jsonschema>=4.17.0
numpy>=1.24.0
pandas>=2.0.0
```

### Optional Packages
```
openpyxl>=3.1.0  # For Excel file handling
xlrd>=2.0.0      # For legacy Excel files
```

## Contributing

### Adding New Data Sources
1. Create a new scraper class inheriting from `BaseScraper`
2. Implement required methods: `get_available_datasets()` and `scrape_dataset()`
3. Add sector mapping configuration
4. Update the data collector to include the new scraper

### Adding New Resource Types
1. Update resource specifications in `ResourceMatrixBuilder`
2. Add sector mapping for the new resource type
3. Implement matrix building logic
4. Update validation and quality assessment

## Troubleshooting

### Common Problems

#### No Data Available
- **Cause**: Data sources may be temporarily unavailable or rate-limited
- **Solution**: Check network connectivity and API key validity
- **Fallback**: System will return empty data rather than synthetic data

#### Sector Mapping Issues
- **Cause**: Data source uses different sector classification
- **Solution**: Update sector mapping configuration
- **Verification**: Check mapping confidence scores

#### API Key Issues
- **Cause**: Invalid or expired API key
- **Solution**: Verify API key and check rate limits
- **Fallback**: Some data may still be available without API key

### Debug Mode
Enable debug logging to see detailed information about data collection by setting the logging level to DEBUG in your application.

## License

This module is part of the Central Planning Experiment project and is licensed under the same terms as the main project.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logging output for error details
3. Verify data source availability and API key validity
4. Check the project's issue tracker for known problems
