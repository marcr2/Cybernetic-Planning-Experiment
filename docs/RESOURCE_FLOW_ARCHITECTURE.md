# Resource Flow Model Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RESOURCE FLOW AND TRANSPORTATION COST MODEL              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RESOURCE      │    │   RESOURCE      │    │ TRANSPORTATION  │
│   DEFINITIONS   │    │   INPUT MATRIX  │    │   COST CALC.    │
│                 │    │       (R)       │    │                 │
│ • Resource ID   │───▶│                 │───▶│ • Multi-modal   │
│ • Name          │    │ R[i][j] = units │    │ • Cost factors  │
│ • Type          │    │ of resource i   │    │ • Optimization  │
│ • Producing     │    │ per unit of     │    │ • Emissions     │
│   Sector        │    │ sector j output │    │                 │
│ • Base Unit     │    │                 │    │                 │
│ • Density       │    │                 │    │                 │
│ • Value/Unit    │    │                 │    │                 │
│ • Criticality   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        INTEGRATION LAYER                                        │
│                                                                                 │
│ • Leontief Model Integration    • Resource Allocation Optimization             │
│ • Transportation Network Costs  • Resource Flow Reporting                      │
│ • Constraint Violation Checking • Optimization Recommendations                 │
└─────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CYBERNETIC PLANNING SYSTEM INTEGRATION                       │
│                                                                                 │
│ • Economic Planning        • Sector Mapping        • Data Synchronization      │
│ • Resource Constraints    • Regional Planning     • Performance Monitoring     │
│ • Cost Optimization       • Transportation        • Reporting & Analytics      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. RESOURCE DEFINITION
   ┌─────────────┐
   │ Resource ID │
   │ Name        │
   │ Type        │
   │ Sector      │
   │ Unit        │
   └─────────────┘
           │
           ▼

2. CONSUMPTION MATRIX
   ┌─────────────────────────────────────┐
   │     S1    S2    S3    S4    S5     │
   │ R1 0.1   0.0   0.2   0.0   0.1    │
   │ R2 0.0   0.5   0.0   0.3   0.0    │
   │ R3 0.2   0.1   0.0   0.0   0.4    │
   └─────────────────────────────────────┘
           │
           ▼

3. SECTOR OUTPUTS
   ┌─────────────┐
   │ [1000, 2000,│
   │  1500, 500, │
   │  100]       │
   └─────────────┘
           │
           ▼

4. RESOURCE DEMAND CALCULATION
   ┌─────────────────────────────────────┐
   │ R1: 0.1×1000 + 0.2×1500 + 0.1×100  │
   │    = 100 + 300 + 10 = 410 units     │
   │                                     │
   │ R2: 0.5×2000 + 0.3×500 = 1150 units│
   │                                     │
   │ R3: 0.2×1000 + 0.1×2000 + 0.4×100  │
   │    = 200 + 200 + 40 = 440 units     │
   └─────────────────────────────────────┘
           │
           ▼

5. TRANSPORTATION COST CALCULATION
   ┌─────────────────────────────────────┐
   │ For each resource flow:             │
   │ • Calculate distance                │
   │ • Apply transport mode multipliers  │
   │ • Factor in resource properties     │
   │ • Calculate total cost              │
   └─────────────────────────────────────┘
           │
           ▼

6. OPTIMIZATION & REPORTING
   ┌─────────────────────────────────────┐
   │ • Resource allocation optimization  │
   │ • Transportation mode selection     │
   │ • Cost minimization                │
   │ • Constraint violation checking     │
   │ • Comprehensive reporting           │
   └─────────────────────────────────────┘
```

## Component Relationships

```
ResourceDefinition
├── resource_id: str
├── resource_name: str
├── resource_type: ResourceType
├── producing_sector_id: str
├── base_unit: str
├── density: float
├── value_per_unit: float
├── perishability: float
├── hazard_class: int
├── storage_requirements: List[str]
├── transport_restrictions: List[str]
├── substitutability: float
├── criticality: float
└── metadata: Dict[str, Any]

ResourceInputMatrix
├── resources: Dict[str, ResourceDefinition]
├── sectors: List[str]
├── matrix: np.ndarray (n_resources × n_sectors)
├── resource_index: Dict[str, int]
├── sector_index: Dict[str, int]
└── consumption_data: Dict[Tuple[str, str], ResourceConsumption]

TransportationCostCalculator
├── transportation_system: TransportationSystem
├── cost_multipliers: Dict[str, Dict[str, float]]
├── calculate_transportation_cost()
├── _calculate_fuel_cost()
├── _calculate_labor_cost()
├── _calculate_maintenance_cost()
├── _calculate_insurance_cost()
└── _calculate_emissions()

ResourceFlowModel
├── resources: Dict[str, ResourceDefinition]
├── resource_matrix: ResourceInputMatrix
├── transportation_calculator: TransportationCostCalculator
├── sectors: List[str]
├── add_resource()
├── initialize_resource_matrix()
├── set_resource_consumption()
├── calculate_total_resource_demand()
└── calculate_transportation_costs()

ResourcePlanningIntegration
├── resource_model: ResourceFlowModel
├── leontief_model: LeontiefModel
├── sector_mapper: SectorMapper
├── data_loader: EnhancedDataLoader
├── stockpile_manager: RegionalStockpileManager
├── integrate_with_leontief_model()
├── optimize_resource_allocation()
├── calculate_transportation_network_costs()
└── generate_resource_flow_report()
```

## Example Resource Flow

```
IRON ORE (R001) FLOW EXAMPLE:

1. Production
   ┌─────────────┐
   │ Iron Mine   │ ── produces ──▶ Iron Ore (R001)
   │ (S051)      │                1000 tons
   └─────────────┘

2. Consumption Matrix
   ┌─────────────────────────────────────┐
   │     S001   S002   S160   S039      │
   │ R001 0.1   0.0   1.2    0.0       │
   └─────────────────────────────────────┘

3. Sector Outputs
   ┌─────────────────────────────────────┐
   │ S001: 1000  S002: 2000             │
   │ S160: 1500  S039: 500              │
   └─────────────────────────────────────┘

4. Resource Demand
   ┌─────────────────────────────────────┐
   │ R001 demand = 0.1×1000 + 1.2×1500  │
   │            = 100 + 1800 = 1900 tons │
   └─────────────────────────────────────┘

5. Transportation
   ┌─────────────┐    ┌─────────────┐
   │ Iron Mine   │───▶│ Steel Plant │
   │ (Origin)    │    │ (S160)      │
   │ 1000 tons   │    │ (Destination)│
   └─────────────┘    └─────────────┘
           │
           ▼
   ┌─────────────────────────────────────┐
   │ Transportation Cost Calculation:    │
   │ • Distance: 150.2 km                │
   │ • Mode: Truck                       │
   │ • Cost: $1,250.50                   │
   │ • Emissions: 45.2 kg CO2            │
   └─────────────────────────────────────┘
```

## Integration Points

```
EXISTING CYBERNETIC PLANNING SYSTEM
├── Sector Mapping
│   ├── SectorMapper
│   └── Enhanced Data Loader
├── Economic Models
│   ├── Leontief Input-Output Model
│   ├── Marxist Economics
│   └── Optimization Algorithms
├── Transportation System
│   ├── Multi-modal Transport
│   ├── Route Optimization
│   └── Fleet Management
├── Regional Planning
│   ├── Regional Stockpiles
│   ├── Population Health
│   └── Infrastructure Networks
└── Data Management
    ├── Resource Data Collection
    ├── Matrix Building
    └── Data Synchronization

RESOURCE FLOW MODEL INTEGRATION
├── Resource Definitions
│   └── Maps to existing sector structure
├── Consumption Matrices
│   └── Integrates with Leontief matrices
├── Transportation Costs
│   └── Uses existing transportation system
├── Optimization
│   └── Provides constraints for economic planning
└── Reporting
    └── Generates resource flow analytics
```

This architecture provides a comprehensive framework for modeling resource flows, calculating transportation costs, and integrating with the existing cybernetic planning system to enable sophisticated economic planning and resource optimization.
