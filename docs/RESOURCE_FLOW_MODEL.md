# Resource Flow and Transportation Cost Model

## Overview

The Resource Flow and Transportation Cost Model is a comprehensive framework designed to integrate with the existing cybernetic planning system. It provides the data structures and algorithms necessary to model resource flows throughout the economy, calculate transportation costs, and optimize resource allocation and distribution.

## Core Components

### 1. Resource Data Structure

The `ResourceDefinition` class defines the primary data structure for all resources in the economic system:

```python
@dataclass
class ResourceDefinition:
    resource_id: str                        # Unique identifier (e.g., "R001", "R002")
    resource_name: str                      # Human-readable name (e.g., "Iron Ore", "Crude Oil")
    resource_type: ResourceType             # Category of resource
    producing_sector_id: str                # ID of primary producing sector
    base_unit: str                          # Standard unit of measurement
    density: float = 1.0                    # kg/m³ for transportation calculations
    value_per_unit: float = 0.0             # Base economic value per unit
    perishability: float = 0.0              # 0.0 = non-perishable, 1.0 = highly perishable
    hazard_class: int = 0                   # 0 = safe, 1-9 = hazardous materials
    storage_requirements: List[str] = []    # Special storage needs
    transport_restrictions: List[str] = []  # Transport limitations
    substitutability: float = 0.5           # 0.0 = unique, 1.0 = highly substitutable
    criticality: float = 0.5                # 0.0 = optional, 1.0 = critical for economy
    metadata: Dict[str, Any] = {}           # Additional properties
```

#### Resource Types

The model supports various resource categories:

- **RAW_MATERIAL**: Iron ore, crude oil, timber
- **PROCESSED_MATERIAL**: Steel, gasoline, lumber
- **MANUFACTURED_GOOD**: Electronics, machinery, vehicles
- **ENERGY**: Electricity, fuel, natural gas
- **LABOR**: Skilled, unskilled, technical
- **SERVICES**: Healthcare, education, transportation
- **AGRICULTURAL**: Crops, livestock, food products
- **CONSTRUCTION**: Cement, concrete, building materials
- **TECHNOLOGY**: Software, hardware, data
- **FINANCIAL**: Capital, credit, insurance

### 2. Resource Input Matrix (R)

The `ResourceInputMatrix` class implements the Resource Input Matrix (R) that maps resource consumption to sector outputs:

**Dimensions**: (Number of Resources × Number of Sectors)
**R[i][j]**: Units of resource i required to produce one unit of output for sector j

```python
class ResourceInputMatrix:
    def __init__(self, resources: List[ResourceDefinition], sectors: List[str]):
        self.resources = {r.resource_id: r for r in resources}
        self.sectors = sectors
        self.matrix = np.zeros((len(resources), len(sectors)))
    
    def set_consumption(self, resource_id: str, sector_id: str, 
                       consumption_rate: float, consumption_type: str = "direct"):
        # Set consumption rate for resource-sector pair
    
    def get_total_resource_demand(self, resource_id: str, sector_outputs: np.ndarray) -> float:
        # Calculate total demand for a resource given sector outputs
```

#### Example Usage

```python
# Create matrix with 3 resources and 3 sectors
resources = [iron_ore, steel, electricity]
sectors = ["S001", "S002", "S003"]
matrix = ResourceInputMatrix(resources, sectors)

# Set consumption patterns
matrix.set_consumption("R001", "S001", 0.5)  # 0.5 tons iron ore per unit S001 output
matrix.set_consumption("R002", "S002", 1.0)  # 1.0 ton steel per unit S002 output

# Calculate total demand
sector_outputs = np.array([100, 200, 150])
iron_demand = matrix.get_total_resource_demand("R001", sector_outputs)  # 50 tons
```

### 3. Transportation Cost Calculation Framework

The `TransportationCostCalculator` class provides comprehensive cost calculation for resource transportation:

```python
class TransportationCostCalculator:
    def calculate_transportation_cost(
        self,
        resource: ResourceDefinition,
        quantity: float,
        origin: Location,
        destination: Location,
        transport_mode: TransportMode,
        vehicle_type: Optional[VehicleType] = None
    ) -> Dict[str, Any]:
        # Calculate comprehensive transportation cost
```

#### Cost Components

The transportation cost calculation includes:

1. **Base Cost**: Distance × base cost per km × resource-specific multiplier
2. **Fuel Cost**: Fuel consumption × fuel price × weight factor
3. **Labor Cost**: Travel time × labor rate
4. **Maintenance Cost**: Distance × maintenance rate
5. **Insurance Cost**: Base cost × insurance rate (adjusted for risk)
6. **Environmental Cost**: Emissions calculation for CO2 tracking

#### Transport Modes

The system supports multiple transportation modes:

- **TRUCK**: Road transportation
- **RAIL**: Railway transportation
- **AIRCRAFT**: Air transportation
- **PIPELINE**: Pipeline transportation
- **SHIP**: Maritime transportation

#### Cost Factors

Different resource types have different cost multipliers:

```python
cost_multipliers = {
    "raw_material": {
        "truck": 1.0, "rail": 0.6, "ship": 0.3, "pipeline": 0.2, "aircraft": 3.0
    },
    "processed_material": {
        "truck": 1.0, "rail": 0.7, "ship": 0.4, "pipeline": 0.3, "aircraft": 2.5
    },
    "manufactured_good": {
        "truck": 1.0, "rail": 0.8, "ship": 0.5, "pipeline": 0.0, "aircraft": 2.0
    }
}
```

### 4. Integration Layer

The `ResourcePlanningIntegration` class provides seamless integration with the existing economic planning system:

```python
class ResourcePlanningIntegration:
    def integrate_with_leontief_model(self, leontief_model: LeontiefModel) -> Dict[str, Any]:
        # Integrate resource constraints with Leontief input-output model
    
    def optimize_resource_allocation(self, sector_outputs: np.ndarray) -> Dict[str, Any]:
        # Optimize resource allocation given sector outputs and constraints
    
    def calculate_transportation_network_costs(self, resource_flows: Dict, locations: Dict) -> Dict[str, Any]:
        # Calculate comprehensive transportation costs for resource flows
    
    def generate_resource_flow_report(self, sector_outputs: np.ndarray) -> Dict[str, Any]:
        # Generate comprehensive resource flow report
```

## Usage Examples

### Basic Resource Definition

```python
# Define a resource
iron_ore = ResourceDefinition(
    resource_id="R001",
    resource_name="Iron Ore",
    resource_type=ResourceType.RAW_MATERIAL,
    producing_sector_id="S051",  # Mining sector
    base_unit="ton",
    density=5000.0,  # kg/m³
    value_per_unit=50.0,  # $/ton
    criticality=0.9
)
```

### Resource Consumption Matrix

```python
# Create resource flow model
model = ResourceFlowModel()
model.add_resource(iron_ore)
model.initialize_resource_matrix(["S001", "S002", "S003"])

# Set consumption patterns
model.set_resource_consumption("R001", "S001", 0.5)  # Healthcare needs iron
model.set_resource_consumption("R001", "S002", 1.2)  # Agriculture needs more iron

# Calculate total demand
sector_outputs = np.array([1000, 2000, 1500])
total_demand = model.calculate_total_resource_demand(sector_outputs)
print(f"Iron ore demand: {total_demand['R001']:.1f} tons")
```

### Transportation Cost Calculation

```python
# Create locations
origin = Location("MINE001", "Iron Mine", 40.0, -100.0)
destination = Location("FACTORY001", "Steel Factory", 41.0, -99.0)

# Calculate transportation cost
calculator = TransportationCostCalculator()
cost_details = calculator.calculate_transportation_cost(
    resource=iron_ore,
    quantity=1000.0,  # 1000 tons
    origin=origin,
    destination=destination,
    transport_mode=TransportMode.TRUCK
)

print(f"Total cost: ${cost_details['total_cost']:.2f}")
print(f"Distance: {cost_details['distance_km']:.1f} km")
print(f"Emissions: {cost_details['emissions_kg_co2']:.1f} kg CO2")
```

### Integrated Resource Planning

```python
# Create integrated system
integration = create_integrated_resource_system()

# Define economic plan
sector_outputs = np.array([1000, 2000, 1500, 500, 100, 5000, 3000, 2000])

# Generate comprehensive report
report = integration.generate_resource_flow_report(sector_outputs)

print(f"Total resource value: ${report['resource_summary']['total_value']:,.2f}")
print(f"Critical resources: {len(report['resource_summary']['critical_resources'])}")
print(f"Recommendations: {len(report['recommendations'])}")
```

## Example Output Format

Following the specified format with example sectors and resources:

### Sectors
- **S1**: Healthcare
- **S2**: Food and Agriculture  
- **S3**: Energy

### Resources
- **R1.1**: Medicines (produced by S1, consumed by S1)
- **R1.2**: PPE (produced by S1, consumed by S1)
- **R2**: Farm Equipment (produced by S2, consumed by S2)
- **R3**: Electric Components (produced by S3, consumed by S3)

### Resource Input Matrix (R)
```
Resource        S1 (Healthcare)  S2 (Agriculture)  S3 (Energy)
R1.1 (Medicines)     0.100           0.000           0.000
R1.2 (PPE)           0.050           0.000           0.000
R2 (Farm Equipment)  0.000           0.001           0.000
R3 (Electric Comp.)  0.010           0.020           0.200
```

### Transportation Cost Example
```
Transporting 1000 tons of Iron Ore from Mine to Factory:
- Distance: 150.2 km
- Truck: $1,250.50 (2.5 hours, 45.2 kg CO2)
- Rail: $750.30 (1.8 hours, 13.6 kg CO2)
- Aircraft: $3,750.00 (0.2 hours, 225.3 kg CO2)
```

## Integration with Existing System

The resource flow model integrates seamlessly with the existing cybernetic planning system:

1. **Data Integration**: Uses existing sector mappings and data loaders
2. **Matrix Operations**: Compatible with Leontief input-output matrices
3. **Transportation System**: Leverages existing transportation infrastructure
4. **Economic Planning**: Provides resource constraints for optimization
5. **Reporting**: Generates comprehensive resource flow reports

## File Structure

```
src/cybernetic_planning/data/
├── resource_flow_model.py          # Core resource flow model
├── resource_flow_integration.py    # Integration layer
└── ...

examples/
└── resource_flow_demo.py           # Comprehensive demonstration

tests/
└── test_resource_flow_model.py     # Test suite

docs/
└── RESOURCE_FLOW_MODEL.md          # This documentation
```

## Testing

Run the test suite to validate the implementation:

```bash
python -m pytest tests/test_resource_flow_model.py -v
```

Run the demonstration to see the model in action:

```bash
python examples/resource_flow_demo.py
```

## Future Extensions

The resource flow model is designed to be extensible:

1. **Additional Resource Types**: Easy to add new resource categories
2. **Advanced Transportation**: Support for more transport modes and optimization
3. **Regional Planning**: Multi-regional resource allocation
4. **Dynamic Pricing**: Real-time cost updates based on market conditions
5. **Sustainability Metrics**: Enhanced environmental impact tracking
6. **Machine Learning**: Predictive resource demand modeling

## Conclusion

The Resource Flow and Transportation Cost Model provides a comprehensive framework for modeling resource flows in the cybernetic planning system. It successfully addresses all the core requirements:

1. ✅ **Resource Data Structure**: Complete with all required attributes
2. ✅ **Resource Input Matrix (R)**: Maps resource consumption to sector outputs
3. ✅ **Transportation Cost Framework**: Calculates costs for all transport modes
4. ✅ **Integration Layer**: Seamlessly connects with existing economic planning system
5. ✅ **Example Implementation**: Comprehensive examples and test cases

The model is production-ready and can be immediately integrated into the cybernetic planning system to provide sophisticated resource flow modeling and cost optimization capabilities.
