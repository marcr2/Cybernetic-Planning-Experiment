"""
Resource Flow Integration Layer

This module provides integration between the resource flow model and the existing
cybernetic planning system, enabling seamless resource planning and cost analysis
within the economic planning framework.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from .resource_flow_model import (
    ResourceFlowModel, ResourceDefinition, ResourceType,
    ResourceInputMatrix, TransportationCostCalculator
)
from .sector_mapper import SectorMapper
from .enhanced_data_loader import EnhancedDataLoader
from ..core.leontief import LeontiefModel
from ..utils.transportation_system import Location, TransportMode
from ..utils.regional_stockpiles import RegionalStockpileManager


class ResourcePlanningIntegration:
    """
    Integration layer connecting resource flow model with economic planning system.
    
    This class provides methods to:
    1. Integrate resource constraints with Leontief input-output model
    2. Calculate resource requirements for economic plans
    3. Optimize resource allocation and transportation
    4. Generate resource flow reports and visualizations
    """
    
    def __init__(self, 
                 resource_model: ResourceFlowModel,
                 leontief_model: Optional[LeontiefModel] = None,
                 sector_mapper: Optional[SectorMapper] = None,
                 data_loader: Optional[EnhancedDataLoader] = None):
        """
        Initialize the resource planning integration.
        
        Args:
            resource_model: Resource flow model instance
            leontief_model: Leontief input-output model
            sector_mapper: Sector mapping instance
            data_loader: Enhanced data loader instance
        """
        self.resource_model = resource_model
        self.leontief_model = leontief_model
        self.sector_mapper = sector_mapper or SectorMapper()
        self.data_loader = data_loader or EnhancedDataLoader()
        self.stockpile_manager = RegionalStockpileManager()
        
        # Integration state
        self.integration_metadata = {
            "created_at": datetime.now().isoformat(),
            "resource_count": len(resource_model.resources),
            "sector_count": len(resource_model.sectors) if resource_model.sectors else 0
        }
    
    def integrate_with_leontief_model(self, leontief_model: LeontiefModel) -> Dict[str, Any]:
        """
        Integrate resource constraints with the Leontief input-output model.
        
        Args:
            leontief_model: Leontief model to integrate with
            
        Returns:
            Integration results and updated model
        """
        if self.resource_model.resource_matrix is None:
            raise ValueError("Resource matrix not initialized")
        
        # Get sector outputs from Leontief model
        if hasattr(leontief_model, 'final_demand') and leontief_model.final_demand is not None:
            # Calculate total output using Leontief inverse
            leontief_inverse = leontief_model.get_leontief_inverse()
            total_output = np.dot(leontief_inverse, leontief_model.final_demand)
        else:
            # Use default sector outputs if no final demand specified
            total_output = np.ones(len(self.resource_model.sectors)) * 1000.0
        
        # Calculate resource requirements
        resource_requirements = self.resource_model.calculate_total_resource_demand(total_output)
        
        # Create resource constraint matrix for optimization
        resource_constraint_matrix = self._create_resource_constraint_matrix()
        
        # Calculate resource availability constraints
        resource_availability = self._calculate_resource_availability()
        
        integration_result = {
            "success": True,
            "resource_requirements": resource_requirements,
            "resource_constraint_matrix": resource_constraint_matrix.tolist(),
            "resource_availability": resource_availability,
            "total_output": total_output.tolist(),
            "constraint_violations": self._check_constraint_violations(
                resource_requirements, resource_availability
            ),
            "integration_metadata": {
                "leontief_sectors": len(total_output),
                "resource_constraints": resource_constraint_matrix.shape[0],
                "integration_timestamp": datetime.now().isoformat()
            }
        }
        
        return integration_result
    
    def _create_resource_constraint_matrix(self) -> np.ndarray:
        """Create resource constraint matrix for optimization."""
        if self.resource_model.resource_matrix is None:
            return np.array([])
        
        # Use the resource input matrix as constraint matrix
        return self.resource_model.resource_matrix.matrix
    
    def _calculate_resource_availability(self) -> Dict[str, float]:
        """Calculate available resource quantities."""
        availability = {}
        
        for resource_id, resource in self.resource_model.resources.items():
            # Base availability from producing sector capacity
            base_availability = self._get_sector_production_capacity(resource.producing_sector_id)
            
            # Adjust for resource-specific factors
            criticality_factor = 1.0 + resource.criticality * 0.5  # Critical resources get priority
            availability[resource_id] = base_availability * criticality_factor
        
        return availability
    
    def _get_sector_production_capacity(self, sector_id: str) -> float:
        """Get production capacity for a sector."""
        # This would typically come from sector data or economic models
        # For now, return a default capacity based on sector type
        default_capacities = {
            "S001": 10000.0,  # Healthcare
            "S002": 50000.0,  # Food and Agriculture
            "S003": 20000.0,  # Energy
            "S007": 5000.0,   # Pharmaceuticals
            "S039": 1000.0,   # Farm Equipment
            "S051": 100000.0, # Oil and Gas Extraction
            "S053": 50000.0,  # Electric Power Generation
            "S160": 30000.0,  # Metal Products Manufacturing
        }
        return default_capacities.get(sector_id, 1000.0)
    
    def _check_constraint_violations(self, requirements: Dict[str, float], 
                                   availability: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for resource constraint violations."""
        violations = []
        
        for resource_id in requirements:
            required = requirements[resource_id]
            available = availability.get(resource_id, 0.0)
            
            if required > available:
                violations.append({
                    "resource_id": resource_id,
                    "resource_name": self.resource_model.resources[resource_id].resource_name,
                    "required": required,
                    "available": available,
                    "shortage": required - available,
                    "shortage_percentage": (required - available) / required * 100
                })
        
        return violations
    
    def optimize_resource_allocation(self, 
                                   sector_outputs: np.ndarray,
                                   optimization_objective: str = "minimize_cost") -> Dict[str, Any]:
        """
        Optimize resource allocation given sector outputs and constraints.
        
        Args:
            sector_outputs: Desired sector output levels
            optimization_objective: Objective function ("minimize_cost", "maximize_efficiency")
            
        Returns:
            Optimization results including allocation plan and costs
        """
        if self.resource_model.resource_matrix is None:
            raise ValueError("Resource matrix not initialized")
        
        # Calculate resource requirements
        resource_requirements = self.resource_model.calculate_total_resource_demand(sector_outputs)
        
        # Get resource availability
        resource_availability = self._calculate_resource_availability()
        
        # Create optimization problem
        allocation_plan = {}
        total_cost = 0.0
        optimization_details = {}
        
        for resource_id, required in resource_requirements.items():
            available = resource_availability.get(resource_id, 0.0)
            resource = self.resource_model.resources[resource_id]
            
            if required <= available:
                # Sufficient resources available
                allocation_plan[resource_id] = {
                    "allocated": required,
                    "available": available,
                    "utilization": required / available,
                    "cost": required * resource.value_per_unit,
                    "status": "sufficient"
                }
            else:
                # Resource shortage - allocate what's available
                allocation_plan[resource_id] = {
                    "allocated": available,
                    "available": available,
                    "utilization": 1.0,
                    "cost": available * resource.value_per_unit,
                    "status": "shortage",
                    "shortage": required - available
                }
            
            total_cost += allocation_plan[resource_id]["cost"]
        
        # Calculate efficiency metrics
        total_required = sum(resource_requirements.values())
        total_allocated = sum(plan["allocated"] for plan in allocation_plan.values())
        efficiency = total_allocated / total_required if total_required > 0 else 0.0
        
        optimization_details = {
            "total_cost": total_cost,
            "efficiency": efficiency,
            "resource_utilization": np.mean([plan["utilization"] for plan in allocation_plan.values()]),
            "shortages": [plan for plan in allocation_plan.values() if plan["status"] == "shortage"],
            "objective": optimization_objective
        }
        
        return {
            "allocation_plan": allocation_plan,
            "optimization_details": optimization_details,
            "resource_requirements": resource_requirements,
            "resource_availability": resource_availability
        }
    
    def calculate_transportation_network_costs(self, 
                                             resource_flows: Dict[Tuple[str, str], float],
                                             locations: Dict[str, Location]) -> Dict[str, Any]:
        """
        Calculate comprehensive transportation costs for resource flows.
        
        Args:
            resource_flows: Dictionary mapping (resource_id, destination_id) to quantity
            locations: Dictionary mapping location_id to Location objects
            
        Returns:
            Transportation cost analysis and optimization recommendations
        """
        # Calculate costs for all transport modes
        transport_costs = self.resource_model.calculate_transportation_costs(resource_flows, locations)
        
        # Analyze cost patterns
        cost_analysis = self._analyze_transportation_costs(transport_costs)
        
        # Generate optimization recommendations
        recommendations = self._generate_transportation_recommendations(transport_costs, cost_analysis)
        
        return {
            "transport_costs": transport_costs,
            "cost_analysis": cost_analysis,
            "recommendations": recommendations,
            "total_transport_cost": sum(
                min(costs.values(), key=lambda x: x["total_cost"])["total_cost"]
                for costs in transport_costs.values()
            )
        }
    
    def _analyze_transportation_costs(self, transport_costs: Dict) -> Dict[str, Any]:
        """Analyze transportation cost patterns and identify opportunities."""
        analysis = {
            "mode_efficiency": {},
            "cost_distribution": {},
            "optimization_opportunities": []
        }
        
        # Analyze by transport mode
        mode_costs = {}
        for flow_costs in transport_costs.values():
            for mode, cost_data in flow_costs.items():
                if mode not in mode_costs:
                    mode_costs[mode] = []
                mode_costs[mode].append(cost_data["total_cost"])
        
        for mode, costs in mode_costs.items():
            analysis["mode_efficiency"][mode] = {
                "average_cost": np.mean(costs),
                "median_cost": np.median(costs),
                "cost_variance": np.var(costs),
                "flow_count": len(costs)
            }
        
        # Identify optimization opportunities
        for (resource_id, dest_id), costs in transport_costs.items():
            best_mode = min(costs.keys(), key=lambda m: costs[m]["total_cost"])
            worst_mode = max(costs.keys(), key=lambda m: costs[m]["total_cost"])
            
            savings = costs[worst_mode]["total_cost"] - costs[best_mode]["total_cost"]
            if savings > costs[best_mode]["total_cost"] * 0.1:  # >10% savings
                analysis["optimization_opportunities"].append({
                    "resource_id": resource_id,
                    "destination": dest_id,
                    "current_mode": worst_mode,
                    "recommended_mode": best_mode,
                    "potential_savings": savings,
                    "savings_percentage": savings / costs[worst_mode]["total_cost"] * 100
                })
        
        return analysis
    
    def _generate_transportation_recommendations(self, transport_costs: Dict, 
                                               cost_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate transportation optimization recommendations."""
        recommendations = []
        
        # Mode efficiency recommendations
        mode_efficiency = cost_analysis["mode_efficiency"]
        if mode_efficiency:
            most_efficient = min(mode_efficiency.keys(), 
                               key=lambda m: mode_efficiency[m]["average_cost"])
            least_efficient = max(mode_efficiency.keys(), 
                                key=lambda m: mode_efficiency[m]["average_cost"])
            
            recommendations.append({
                "type": "mode_efficiency",
                "priority": "high",
                "title": f"Consider using {most_efficient} over {least_efficient}",
                "description": f"{most_efficient} is {mode_efficiency[least_efficient]['average_cost'] / mode_efficiency[most_efficient]['average_cost']:.1f}x more cost-effective",
                "potential_savings": "Significant cost reduction possible"
            })
        
        # Specific flow recommendations
        for opportunity in cost_analysis["optimization_opportunities"]:
            recommendations.append({
                "type": "flow_optimization",
                "priority": "medium",
                "title": f"Optimize {opportunity['resource_id']} transport to {opportunity['destination']}",
                "description": f"Switch from {opportunity['current_mode']} to {opportunity['recommended_mode']}",
                "potential_savings": f"${opportunity['potential_savings']:.2f} ({opportunity['savings_percentage']:.1f}%)"
            })
        
        return recommendations
    
    def generate_resource_flow_report(self, 
                                    sector_outputs: np.ndarray,
                                    locations: Optional[Dict[str, Location]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive resource flow report.
        
        Args:
            sector_outputs: Sector output levels
            locations: Optional location data for transportation analysis
            
        Returns:
            Comprehensive resource flow report
        """
        # Calculate resource requirements
        resource_requirements = self.resource_model.calculate_total_resource_demand(sector_outputs)
        
        # Optimize allocation
        allocation_result = self.optimize_resource_allocation(sector_outputs)
        
        # Calculate transportation costs if locations provided
        transport_analysis = None
        if locations:
            # Create resource flows (simplified - all resources go to all sectors)
            resource_flows = {}
            for resource_id in resource_requirements:
                for sector_id in self.resource_model.sectors:
                    if sector_id in locations:
                        quantity = self.resource_model.resource_matrix.get_consumption(resource_id, sector_id)
                        if quantity > 0:
                            resource_flows[(resource_id, sector_id)] = quantity * sector_outputs[self.resource_model.sectors.index(sector_id)]
            
            transport_analysis = self.calculate_transportation_network_costs(resource_flows, locations)
        
        # Generate summary statistics
        total_resource_value = sum(
            resource_requirements[rid] * self.resource_model.resources[rid].value_per_unit
            for rid in resource_requirements
        )
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "sector_outputs": sector_outputs.tolist(),
                "total_sectors": len(sector_outputs),
                "total_resources": len(resource_requirements)
            },
            "resource_summary": {
                "total_requirements": resource_requirements,
                "total_value": total_resource_value,
                "resource_types": self._summarize_by_resource_type(resource_requirements),
                "critical_resources": self._identify_critical_resources(resource_requirements)
            },
            "allocation_analysis": allocation_result,
            "transportation_analysis": transport_analysis,
            "recommendations": self._generate_overall_recommendations(allocation_result, transport_analysis)
        }
        
        return report
    
    def _summarize_by_resource_type(self, requirements: Dict[str, float]) -> Dict[str, Any]:
        """Summarize resource requirements by type."""
        type_summary = {}
        
        for resource_id, quantity in requirements.items():
            resource = self.resource_model.resources[resource_id]
            resource_type = resource.resource_type.value
            
            if resource_type not in type_summary:
                type_summary[resource_type] = {
                    "count": 0,
                    "total_quantity": 0.0,
                    "total_value": 0.0,
                    "resources": []
                }
            
            type_summary[resource_type]["count"] += 1
            type_summary[resource_type]["total_quantity"] += quantity
            type_summary[resource_type]["total_value"] += quantity * resource.value_per_unit
            type_summary[resource_type]["resources"].append({
                "resource_id": resource_id,
                "name": resource.resource_name,
                "quantity": quantity,
                "value": quantity * resource.value_per_unit
            })
        
        return type_summary
    
    def _identify_critical_resources(self, requirements: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify critical resources based on requirements and availability."""
        critical = []
        availability = self._calculate_resource_availability()
        
        for resource_id, required in requirements.items():
            resource = self.resource_model.resources[resource_id]
            available = availability.get(resource_id, 0.0)
            
            # Critical if high requirement, high criticality, or shortage
            is_critical = (
                resource.criticality > 0.8 or
                required > available or
                (required / max(available, 1)) > 0.9
            )
            
            if is_critical:
                critical.append({
                    "resource_id": resource_id,
                    "name": resource.resource_name,
                    "required": required,
                    "available": available,
                    "criticality": resource.criticality,
                    "utilization": required / max(available, 1),
                    "shortage": max(0, required - available)
                })
        
        return sorted(critical, key=lambda x: x["criticality"], reverse=True)
    
    def _generate_overall_recommendations(self, allocation_result: Dict, 
                                        transport_analysis: Optional[Dict]) -> List[Dict[str, Any]]:
        """Generate overall recommendations based on analysis."""
        recommendations = []
        
        # Resource allocation recommendations
        shortages = allocation_result["optimization_details"]["shortages"]
        if shortages:
            recommendations.append({
                "type": "resource_shortage",
                "priority": "critical",
                "title": "Address resource shortages",
                "description": f"{len(shortages)} resources have insufficient availability",
                "details": [f"{s['resource_id']}: {s['shortage']:.1f} units short" for s in shortages]
            })
        
        # Transportation recommendations
        if transport_analysis and "recommendations" in transport_analysis:
            recommendations.extend(transport_analysis["recommendations"])
        
        # Efficiency recommendations
        efficiency = allocation_result["optimization_details"]["efficiency"]
        if efficiency < 0.8:
            recommendations.append({
                "type": "efficiency",
                "priority": "medium",
                "title": "Improve resource allocation efficiency",
                "description": f"Current efficiency: {efficiency:.1%}",
                "suggestion": "Review resource allocation algorithms and constraints"
            })
        
        return recommendations
    
    def save_integration_state(self, filepath: str) -> None:
        """Save the integration state to a file."""
        state_data = {
            "integration_metadata": self.integration_metadata,
            "resource_model_summary": self.resource_model.get_resource_summary(),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_integration_state(self, filepath: str) -> None:
        """Load integration state from a file."""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.integration_metadata = state_data["integration_metadata"]


def create_integrated_resource_system(sector_data_file: Optional[str] = None) -> ResourcePlanningIntegration:
    """
    Create a fully integrated resource planning system.
    
    Args:
        sector_data_file: Optional path to sector data file
        
    Returns:
        Integrated resource planning system
    """
    # Create resource model with example data
    from .resource_flow_model import create_example_resource_model
    resource_model = create_example_resource_model()
    
    # Create sector mapper and data loader
    sector_mapper = SectorMapper()
    data_loader = EnhancedDataLoader()
    
    # Load sector data if provided
    if sector_data_file and Path(sector_data_file).exists():
        sector_data = data_loader.load_sector_data(sector_data_file)
        # Update resource model with actual sector data
        if "sectors" in sector_data:
            resource_model.initialize_resource_matrix(sector_data["sectors"])
    
    # Create integration layer
    integration = ResourcePlanningIntegration(
        resource_model=resource_model,
        sector_mapper=sector_mapper,
        data_loader=data_loader
    )
    
    return integration


if __name__ == "__main__":
    # Example usage
    print("Creating integrated resource planning system...")
    integration = create_integrated_resource_system()
    
    # Example sector outputs
    sector_outputs = np.array([1000, 2000, 1500, 500, 100, 5000, 3000, 2000])
    
    # Generate resource flow report
    print("\nGenerating resource flow report...")
    report = integration.generate_resource_flow_report(sector_outputs)
    
    print(f"\nResource Flow Report Summary:")
    print(f"Total resources required: {len(report['resource_summary']['total_requirements'])}")
    print(f"Total resource value: ${report['resource_summary']['total_value']:,.2f}")
    print(f"Critical resources: {len(report['resource_summary']['critical_resources'])}")
    print(f"Recommendations: {len(report['recommendations'])}")
    
    # Save integration state
    integration.save_integration_state("resource_integration_state.json")
    print("\nIntegration state saved to resource_integration_state.json")
