"""
Enhanced Sector-Settlement Mapping System

Provides sophisticated mapping between economic sectors and spatial settlements
with realistic constraints, optimization algorithms, and economic geography principles.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import heapq

logger = logging.getLogger(__name__)

class SectorType(Enum):
    """Types of economic sectors."""
    PRIMARY = "primary"        # Agriculture, mining, forestry
    SECONDARY = "secondary"    # Manufacturing, construction
    TERTIARY = "tertiary"      # Services, trade, finance
    QUATERNARY = "quaternary"  # Information, R&D, education

class SettlementHierarchy(Enum):
    """Settlement hierarchy levels."""
    MEGACITY = "megacity"      # > 10M population
    METROPOLIS = "metropolis"  # 1M - 10M population
    CITY = "city"              # 100K - 1M population
    TOWN = "town"              # 10K - 100K population
    VILLAGE = "village"         # 1K - 10K population
    RURAL = "rural"            # < 1K population

@dataclass
class SectorCharacteristics:
    """Characteristics of an economic sector."""
    sector_id: str
    sector_name: str
    sector_type: SectorType
    labor_intensity: float          # Labor per unit output
    capital_intensity: float        # Capital per unit output
    resource_intensity: float      # Resource consumption per unit output
    transport_sensitivity: float   # How much transport costs affect viability
    agglomeration_preference: float # Preference for clustering with similar sectors
    infrastructure_requirements: List[str] = field(default_factory=list)
    environmental_impact: float = 0.0
    technology_level: float = 1.0
    market_access_importance: float = 0.5

@dataclass
class SettlementCharacteristics:
    """Characteristics of a settlement."""
    settlement_id: str
    settlement_name: str
    hierarchy_level: SettlementHierarchy
    population: int
    economic_importance: float
    infrastructure_quality: float
    resource_endowment: float
    market_access: float
    labor_availability: float
    capital_availability: float
    environmental_capacity: float
    existing_sectors: List[str] = field(default_factory=list)
    specialization_index: float = 0.0  # How specialized the settlement is

@dataclass
class MappingConstraint:
    """Constraint for sector-settlement mapping."""
    constraint_type: str
    sector_id: Optional[str] = None
    settlement_id: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    weight: float = 1.0
    description: str = ""

@dataclass
class MappingResult:
    """Result of sector-settlement mapping."""
    sector_id: str
    settlement_id: str
    suitability_score: float
    production_capacity: float
    efficiency_factors: Dict[str, float]
    constraints_satisfied: bool
    optimization_notes: str = ""

class EnhancedSectorSettlementMapper:
    """
    Enhanced sector-settlement mapping with realistic economic geography.
    
    Features:
    - Economic geography principles (Central Place Theory, Agglomeration Economics)
    - Multi-objective optimization
    - Constraint satisfaction
    - Hierarchical settlement analysis
    - Sector clustering and specialization
    - Infrastructure and resource constraints
    """
    
    def __init__(self):
        """Initialize the enhanced mapper."""
        self.sector_characteristics: Dict[str, SectorCharacteristics] = {}
        self.settlement_characteristics: Dict[str, SettlementCharacteristics] = {}
        self.mapping_constraints: List[MappingConstraint] = []
        self.mapping_results: List[MappingResult] = []
        
        # Optimization parameters
        self.optimization_weights = {
            'efficiency': 0.3,
            'specialization': 0.2,
            'infrastructure': 0.2,
            'resource_access': 0.15,
            'market_access': 0.15
        }
        
        # Economic geography parameters
        self.agglomeration_threshold = 0.7
        self.specialization_threshold = 0.6
        self.minimum_settlement_size = 1000
        
        logger.info("Enhanced sector-settlement mapper initialized")
    
    def add_sector(self, sector_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a sector with its characteristics."""
        try:
            sector_id = sector_data.get('id', f'sector_{len(self.sector_characteristics)}')
            
            # Determine sector type
            sector_name = sector_data.get('name', '').lower()
            if any(word in sector_name for word in ['agriculture', 'farming', 'mining', 'forestry']):
                sector_type = SectorType.PRIMARY
            elif any(word in sector_name for word in ['manufacturing', 'construction', 'industry']):
                sector_type = SectorType.SECONDARY
            elif any(word in sector_name for word in ['service', 'trade', 'finance', 'healthcare']):
                sector_type = SectorType.TERTIARY
            else:
                sector_type = SectorType.QUATERNARY
            
            characteristics = SectorCharacteristics(
                sector_id=sector_id,
                sector_name=sector_data.get('name', 'Unknown'),
                sector_type=sector_type,
                labor_intensity=sector_data.get('labor_intensity', 0.5),
                capital_intensity=sector_data.get('capital_intensity', 0.5),
                resource_intensity=sector_data.get('resource_intensity', 0.5),
                transport_sensitivity=sector_data.get('transport_sensitivity', 0.5),
                agglomeration_preference=sector_data.get('agglomeration_preference', 0.5),
                infrastructure_requirements=sector_data.get('infrastructure_requirements', []),
                environmental_impact=sector_data.get('environmental_impact', 0.0),
                technology_level=sector_data.get('technology_level', 1.0),
                market_access_importance=sector_data.get('market_access_importance', 0.5)
            )
            
            self.sector_characteristics[sector_id] = characteristics
            
            logger.info(f"Added sector '{sector_name}' ({sector_type.value})")
            
            return {
                "success": True,
                "sector_id": sector_id,
                "sector_type": sector_type.value,
                "characteristics": characteristics
            }
            
        except Exception as e:
            logger.error(f"Failed to add sector: {e}")
            return {"success": False, "error": str(e)}
    
    def add_settlement(self, settlement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a settlement with its characteristics."""
        try:
            settlement_id = settlement_data.get('id', f'settlement_{len(self.settlement_characteristics)}')
            
            # Determine hierarchy level based on population
            population = settlement_data.get('population', 0)
            if population > 10000000:
                hierarchy_level = SettlementHierarchy.MEGACITY
            elif population > 1000000:
                hierarchy_level = SettlementHierarchy.METROPOLIS
            elif population > 100000:
                hierarchy_level = SettlementHierarchy.CITY
            elif population > 10000:
                hierarchy_level = SettlementHierarchy.TOWN
            elif population > 1000:
                hierarchy_level = SettlementHierarchy.VILLAGE
            else:
                hierarchy_level = SettlementHierarchy.RURAL
            
            characteristics = SettlementCharacteristics(
                settlement_id=settlement_id,
                settlement_name=settlement_data.get('name', 'Unknown'),
                hierarchy_level=hierarchy_level,
                population=population,
                economic_importance=settlement_data.get('economic_importance', 0.5),
                infrastructure_quality=settlement_data.get('infrastructure_quality', 0.5),
                resource_endowment=settlement_data.get('resource_endowment', 0.5),
                market_access=settlement_data.get('market_access', 0.5),
                labor_availability=settlement_data.get('labor_availability', 0.5),
                capital_availability=settlement_data.get('capital_availability', 0.5),
                environmental_capacity=settlement_data.get('environmental_capacity', 1.0),
                existing_sectors=settlement_data.get('existing_sectors', []),
                specialization_index=settlement_data.get('specialization_index', 0.0)
            )
            
            self.settlement_characteristics[settlement_id] = characteristics
            
            logger.info(f"Added settlement '{settlement_data.get('name', 'Unknown')}' ({hierarchy_level.value})")
            
            return {
                "success": True,
                "settlement_id": settlement_id,
                "hierarchy_level": hierarchy_level.value,
                "characteristics": characteristics
            }
            
        except Exception as e:
            logger.error(f"Failed to add settlement: {e}")
            return {"success": False, "error": str(e)}
    
    def add_constraint(self, constraint: MappingConstraint) -> Dict[str, Any]:
        """Add a mapping constraint."""
        try:
            self.mapping_constraints.append(constraint)
            
            logger.info(f"Added constraint: {constraint.description}")
            
            return {
                "success": True,
                "constraint_type": constraint.constraint_type,
                "description": constraint.description
            }
            
        except Exception as e:
            logger.error(f"Failed to add constraint: {e}")
            return {"success": False, "error": str(e)}
    
    def optimize_mapping(self, method: str = "hungarian") -> Dict[str, Any]:
        """
        Optimize sector-settlement mapping using specified method.
        
        Args:
            method: Optimization method ("hungarian", "genetic", "greedy", "clustering")
        
        Returns:
            Dictionary with optimization results
        """
        try:
            if not self.sector_characteristics or not self.settlement_characteristics:
                return {"success": False, "error": "No sectors or settlements available for mapping"}
            
            logger.info(f"Starting mapping optimization using {method} method")
            
            if method == "hungarian":
                results = self._hungarian_optimization()
            elif method == "genetic":
                results = self._genetic_optimization()
            elif method == "greedy":
                results = self._greedy_optimization()
            elif method == "clustering":
                results = self._clustering_optimization()
            else:
                return {"success": False, "error": f"Unknown optimization method: {method}"}
            
            self.mapping_results = results
            
            # Validate constraints
            constraint_violations = self._validate_constraints(results)
            
            logger.info(f"Mapping optimization completed. {len(results)} mappings created.")
            
            return {
                "success": True,
                "method": method,
                "mappings_created": len(results),
                "average_suitability": np.mean([r.suitability_score for r in results]),
                "constraint_violations": len(constraint_violations),
                "optimization_summary": self._generate_optimization_summary(results)
            }
            
        except Exception as e:
            logger.error(f"Mapping optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _hungarian_optimization(self) -> List[MappingResult]:
        """Optimize mapping using Hungarian algorithm."""
        sectors = list(self.sector_characteristics.keys())
        settlements = list(self.settlement_characteristics.keys())
        
        # Create cost matrix
        cost_matrix = np.zeros((len(sectors), len(settlements)))
        
        for i, sector_id in enumerate(sectors):
            for j, settlement_id in enumerate(settlements):
                # Calculate negative suitability (cost)
                suitability = self._calculate_suitability(sector_id, settlement_id)
                cost_matrix[i, j] = 1.0 - suitability  # Convert to cost
        
        # Solve assignment problem
        sector_indices, settlement_indices = linear_sum_assignment(cost_matrix)
        
        # Create mapping results
        results = []
        for i, j in zip(sector_indices, settlement_indices):
            sector_id = sectors[i]
            settlement_id = settlements[j]
            
            suitability = self._calculate_suitability(sector_id, settlement_id)
            production_capacity = self._calculate_production_capacity(sector_id, settlement_id)
            efficiency_factors = self._calculate_efficiency_factors(sector_id, settlement_id)
            
            result = MappingResult(
                sector_id=sector_id,
                settlement_id=settlement_id,
                suitability_score=suitability,
                production_capacity=production_capacity,
                efficiency_factors=efficiency_factors,
                constraints_satisfied=self._check_constraints(sector_id, settlement_id),
                optimization_notes="Hungarian algorithm optimization"
            )
            
            results.append(result)
        
        return results
    
    def _genetic_optimization(self) -> List[MappingResult]:
        """Optimize mapping using genetic algorithm (simplified version)."""
        # Simplified genetic algorithm implementation
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        
        # Initialize population
        population = self._initialize_population(population_size)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual) for individual in population]
            
            # Select parents
            parents = self._select_parents(population, fitness_scores)
            
            # Create offspring
            offspring = self._create_offspring(parents)
            
            # Mutate offspring
            offspring = self._mutate_offspring(offspring, mutation_rate)
            
            # Replace population
            population = self._replace_population(population, offspring, fitness_scores)
        
        # Return best individual
        best_individual = max(population, key=self._evaluate_fitness)
        return self._individual_to_mappings(best_individual)
    
    def _greedy_optimization(self) -> List[MappingResult]:
        """Optimize mapping using greedy algorithm."""
        sectors = list(self.sector_characteristics.keys())
        settlements = list(self.settlement_characteristics.keys())
        
        # Create priority queue of sector-settlement pairs
        priority_queue = []
        for sector_id in sectors:
            for settlement_id in settlements:
                suitability = self._calculate_suitability(sector_id, settlement_id)
                heapq.heappush(priority_queue, (-suitability, sector_id, settlement_id))
        
        # Greedy assignment
        assigned_sectors = set()
        assigned_settlements = set()
        results = []
        
        while priority_queue and len(assigned_sectors) < len(sectors):
            neg_suitability, sector_id, settlement_id = heapq.heappop(priority_queue)
            
            if sector_id not in assigned_sectors and settlement_id not in assigned_settlements:
                suitability = -neg_suitability
                production_capacity = self._calculate_production_capacity(sector_id, settlement_id)
                efficiency_factors = self._calculate_efficiency_factors(sector_id, settlement_id)
                
                result = MappingResult(
                    sector_id=sector_id,
                    settlement_id=settlement_id,
                    suitability_score=suitability,
                    production_capacity=production_capacity,
                    efficiency_factors=efficiency_factors,
                    constraints_satisfied=self._check_constraints(sector_id, settlement_id),
                    optimization_notes="Greedy algorithm optimization"
                )
                
                results.append(result)
                assigned_sectors.add(sector_id)
                assigned_settlements.add(settlement_id)
        
        return results
    
    def _clustering_optimization(self) -> List[MappingResult]:
        """Optimize mapping using clustering approach."""
        # Group sectors by type
        sector_groups = {}
        for sector_id, sector in self.sector_characteristics.items():
            sector_type = sector.sector_type
            if sector_type not in sector_groups:
                sector_groups[sector_type] = []
            sector_groups[sector_type].append(sector_id)
        
        # Group settlements by hierarchy
        settlement_groups = {}
        for settlement_id, settlement in self.settlement_characteristics.items():
            hierarchy = settlement.hierarchy_level
            if hierarchy not in settlement_groups:
                settlement_groups[hierarchy] = []
            settlement_groups[hierarchy].append(settlement_id)
        
        results = []
        
        # Map sector groups to settlement groups
        for sector_type, sectors in sector_groups.items():
            # Find best settlement group for this sector type
            best_hierarchy = self._find_best_hierarchy_for_sector_type(sector_type)
            
            if best_hierarchy in settlement_groups:
                settlements = settlement_groups[best_hierarchy]
                
                # Map sectors to settlements within the group
                for i, sector_id in enumerate(sectors):
                    if i < len(settlements):
                        settlement_id = settlements[i]
                        
                        suitability = self._calculate_suitability(sector_id, settlement_id)
                        production_capacity = self._calculate_production_capacity(sector_id, settlement_id)
                        efficiency_factors = self._calculate_efficiency_factors(sector_id, settlement_id)
                        
                        result = MappingResult(
                            sector_id=sector_id,
                            settlement_id=settlement_id,
                            suitability_score=suitability,
                            production_capacity=production_capacity,
                            efficiency_factors=efficiency_factors,
                            constraints_satisfied=self._check_constraints(sector_id, settlement_id),
                            optimization_notes=f"Clustering optimization ({sector_type.value} -> {best_hierarchy.value})"
                        )
                        
                        results.append(result)
        
        return results
    
    def _calculate_suitability(self, sector_id: str, settlement_id: str) -> float:
        """Calculate suitability score for sector-settlement pair."""
        sector = self.sector_characteristics[sector_id]
        settlement = self.settlement_characteristics[settlement_id]
        
        # Base suitability factors
        factors = {
            'hierarchy_match': self._calculate_hierarchy_match(sector, settlement),
            'infrastructure_match': self._calculate_infrastructure_match(sector, settlement),
            'resource_match': self._calculate_resource_match(sector, settlement),
            'market_access': settlement.market_access * sector.market_access_importance,
            'labor_availability': settlement.labor_availability * sector.labor_intensity,
            'capital_availability': settlement.capital_availability * sector.capital_intensity,
            'environmental_capacity': settlement.environmental_capacity - sector.environmental_impact,
            'specialization': self._calculate_specialization_fit(sector, settlement)
        }
        
        # Weighted suitability score
        suitability = sum(factors[key] * self.optimization_weights.get(key, 0.1) 
                         for key in factors.keys() 
                         if key in self.optimization_weights)
        
        # Apply agglomeration effects
        agglomeration_bonus = self._calculate_agglomeration_bonus(sector, settlement)
        suitability += agglomeration_bonus
        
        return min(1.0, max(0.0, suitability))
    
    def _calculate_hierarchy_match(self, sector: SectorCharacteristics, settlement: SettlementCharacteristics) -> float:
        """Calculate hierarchy match score."""
        # Different sector types prefer different settlement hierarchies
        hierarchy_preferences = {
            SectorType.PRIMARY: [SettlementHierarchy.RURAL, SettlementHierarchy.VILLAGE],
            SectorType.SECONDARY: [SettlementHierarchy.TOWN, SettlementHierarchy.CITY],
            SectorType.TERTIARY: [SettlementHierarchy.CITY, SettlementHierarchy.METROPOLIS],
            SectorType.QUATERNARY: [SettlementHierarchy.METROPOLIS, SettlementHierarchy.MEGACITY]
        }
        
        preferred_hierarchies = hierarchy_preferences.get(sector.sector_type, [])
        
        if settlement.hierarchy_level in preferred_hierarchies:
            return 1.0
        else:
            # Partial match based on hierarchy proximity
            hierarchy_levels = [
                SettlementHierarchy.RURAL,
                SettlementHierarchy.VILLAGE,
                SettlementHierarchy.TOWN,
                SettlementHierarchy.CITY,
                SettlementHierarchy.METROPOLIS,
                SettlementHierarchy.MEGACITY
            ]
            
            sector_pref_level = preferred_hierarchies[0] if preferred_hierarchies else SettlementHierarchy.CITY
            sector_pref_index = hierarchy_levels.index(sector_pref_level)
            settlement_index = hierarchy_levels.index(settlement.hierarchy_level)
            
            distance = abs(sector_pref_index - settlement_index)
            return max(0.0, 1.0 - distance * 0.2)
    
    def _calculate_infrastructure_match(self, sector: SectorCharacteristics, settlement: SettlementCharacteristics) -> float:
        """Calculate infrastructure match score."""
        if not sector.infrastructure_requirements:
            return 1.0
        
        # Simple infrastructure matching
        infrastructure_score = settlement.infrastructure_quality
        
        # Adjust based on specific requirements
        for requirement in sector.infrastructure_requirements:
            if requirement.lower() in ['port', 'harbor'] and settlement.hierarchy_level in [SettlementHierarchy.CITY, SettlementHierarchy.METROPOLIS]:
                infrastructure_score += 0.2
            elif requirement.lower() in ['airport'] and settlement.hierarchy_level in [SettlementHierarchy.METROPOLIS, SettlementHierarchy.MEGACITY]:
                infrastructure_score += 0.2
            elif requirement.lower() in ['railway'] and settlement.hierarchy_level in [SettlementHierarchy.TOWN, SettlementHierarchy.CITY]:
                infrastructure_score += 0.1
        
        return min(1.0, infrastructure_score)
    
    def _calculate_resource_match(self, sector: SectorCharacteristics, settlement: SettlementCharacteristics) -> float:
        """Calculate resource match score."""
        # Primary sectors are more resource-dependent
        if sector.sector_type == SectorType.PRIMARY:
            return settlement.resource_endowment
        else:
            # Other sectors have moderate resource requirements
            return 0.5 + (settlement.resource_endowment - 0.5) * 0.5
    
    def _calculate_specialization_fit(self, sector: SectorCharacteristics, settlement: SettlementCharacteristics) -> float:
        """Calculate specialization fit score."""
        if settlement.specialization_index < self.specialization_threshold:
            # Underspecialized settlement - good for diversification
            return 1.0 - settlement.specialization_index
        else:
            # Specialized settlement - check if sector fits specialization
            existing_sectors = settlement.existing_sectors
            if existing_sectors:
                # Check if similar sectors exist
                sector_type_match = any(
                    self.sector_characteristics.get(existing_id, SectorCharacteristics("", "", SectorType.PRIMARY, 0, 0, 0, 0, 0)).sector_type == sector.sector_type
                    for existing_id in existing_sectors
                )
                return 0.8 if sector_type_match else 0.3
            else:
                return 0.5
    
    def _calculate_agglomeration_bonus(self, sector: SectorCharacteristics, settlement: SettlementCharacteristics) -> float:
        """Calculate agglomeration bonus."""
        if sector.agglomeration_preference < 0.3:
            return 0.0  # Low agglomeration preference
        
        # Check for similar sectors in settlement
        similar_sectors = 0
        for existing_sector_id in settlement.existing_sectors:
            existing_sector = self.sector_characteristics.get(existing_sector_id)
            if existing_sector and existing_sector.sector_type == sector.sector_type:
                similar_sectors += 1
        
        # Agglomeration bonus increases with similar sectors
        bonus = min(0.2, similar_sectors * 0.05 * sector.agglomeration_preference)
        return bonus
    
    def _calculate_production_capacity(self, sector_id: str, settlement_id: str) -> float:
        """Calculate production capacity for sector-settlement pair."""
        sector = self.sector_characteristics[sector_id]
        settlement = self.settlement_characteristics[settlement_id]
        
        # Base capacity from population
        base_capacity = settlement.population / 1000.0
        
        # Adjust for sector characteristics
        labor_factor = sector.labor_intensity * settlement.labor_availability
        capital_factor = sector.capital_intensity * settlement.capital_availability
        
        # Adjust for hierarchy level
        hierarchy_multiplier = {
            SettlementHierarchy.RURAL: 0.5,
            SettlementHierarchy.VILLAGE: 0.7,
            SettlementHierarchy.TOWN: 1.0,
            SettlementHierarchy.CITY: 1.5,
            SettlementHierarchy.METROPOLIS: 2.0,
            SettlementHierarchy.MEGACITY: 3.0
        }.get(settlement.hierarchy_level, 1.0)
        
        capacity = base_capacity * labor_factor * capital_factor * hierarchy_multiplier
        return max(0.0, capacity)
    
    def _calculate_efficiency_factors(self, sector_id: str, settlement_id: str) -> Dict[str, float]:
        """Calculate efficiency factors for sector-settlement pair."""
        sector = self.sector_characteristics[sector_id]
        settlement = self.settlement_characteristics[settlement_id]
        
        return {
            'labor_efficiency': settlement.labor_availability * sector.labor_intensity,
            'capital_efficiency': settlement.capital_availability * sector.capital_intensity,
            'resource_efficiency': settlement.resource_endowment * sector.resource_intensity,
            'infrastructure_efficiency': settlement.infrastructure_quality,
            'market_efficiency': settlement.market_access * sector.market_access_importance,
            'environmental_efficiency': settlement.environmental_capacity - sector.environmental_impact,
            'technology_efficiency': settlement.infrastructure_quality * sector.technology_level
        }
    
    def _check_constraints(self, sector_id: str, settlement_id: str) -> bool:
        """Check if sector-settlement pair satisfies all constraints."""
        for constraint in self.mapping_constraints:
            if not self._satisfies_constraint(sector_id, settlement_id, constraint):
                return False
        return True
    
    def _satisfies_constraint(self, sector_id: str, settlement_id: str, constraint: MappingConstraint) -> bool:
        """Check if a specific constraint is satisfied."""
        if constraint.constraint_type == "minimum_population":
            settlement = self.settlement_characteristics[settlement_id]
            return settlement.population >= (constraint.min_value or 0)
        
        elif constraint.constraint_type == "maximum_sectors_per_settlement":
            # Count existing sectors in settlement
            existing_count = len(settlement.existing_sectors)
            return existing_count < (constraint.max_value or float('inf'))
        
        elif constraint.constraint_type == "sector_type_restriction":
            sector = self.sector_characteristics[sector_id]
            # This would need more specific constraint data
            return True
        
        # Add more constraint types as needed
        return True
    
    def _validate_constraints(self, results: List[MappingResult]) -> List[str]:
        """Validate all constraints for mapping results."""
        violations = []
        
        for result in results:
            if not result.constraints_satisfied:
                violations.append(f"Sector {result.sector_id} -> Settlement {result.settlement_id} violates constraints")
        
        return violations
    
    def _generate_optimization_summary(self, results: List[MappingResult]) -> Dict[str, Any]:
        """Generate optimization summary."""
        if not results:
            return {}
        
        suitability_scores = [r.suitability_score for r in results]
        production_capacities = [r.production_capacity for r in results]
        
        return {
            'total_mappings': len(results),
            'average_suitability': np.mean(suitability_scores),
            'min_suitability': np.min(suitability_scores),
            'max_suitability': np.max(suitability_scores),
            'total_production_capacity': sum(production_capacities),
            'average_production_capacity': np.mean(production_capacities),
            'constraint_satisfaction_rate': sum(1 for r in results if r.constraints_satisfied) / len(results)
        }
    
    def _find_best_hierarchy_for_sector_type(self, sector_type: SectorType) -> SettlementHierarchy:
        """Find best settlement hierarchy for a sector type."""
        hierarchy_preferences = {
            SectorType.PRIMARY: SettlementHierarchy.RURAL,
            SectorType.SECONDARY: SettlementHierarchy.TOWN,
            SectorType.TERTIARY: SettlementHierarchy.CITY,
            SectorType.QUATERNARY: SettlementHierarchy.METROPOLIS
        }
        
        return hierarchy_preferences.get(sector_type, SettlementHierarchy.CITY)
    
    # Genetic algorithm helper methods
    def _initialize_population(self, size: int) -> List[List[Tuple[str, str]]]:
        """Initialize population for genetic algorithm."""
        population = []
        sectors = list(self.sector_characteristics.keys())
        settlements = list(self.settlement_characteristics.keys())
        
        for _ in range(size):
            # Random assignment
            individual = []
            used_settlements = set()
            
            for sector_id in sectors:
                available_settlements = [s for s in settlements if s not in used_settlements]
                if available_settlements:
                    settlement_id = np.random.choice(available_settlements)
                    individual.append((sector_id, settlement_id))
                    used_settlements.add(settlement_id)
            
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual: List[Tuple[str, str]]) -> float:
        """Evaluate fitness of an individual."""
        total_fitness = 0.0
        
        for sector_id, settlement_id in individual:
            suitability = self._calculate_suitability(sector_id, settlement_id)
            total_fitness += suitability
        
        return total_fitness / len(individual) if individual else 0.0
    
    def _select_parents(self, population: List[List[Tuple[str, str]]], fitness_scores: List[float]) -> List[List[Tuple[str, str]]]:
        """Select parents for reproduction."""
        # Tournament selection
        parents = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_index])
        
        return parents
    
    def _create_offspring(self, parents: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        """Create offspring from parents."""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Simple crossover
                child1 = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
                child2 = parent2[:len(parent2)//2] + parent1[len(parent1)//2:]
                
                offspring.extend([child1, child2])
        
        return offspring
    
    def _mutate_offspring(self, offspring: List[List[Tuple[str, str]]], mutation_rate: float) -> List[List[Tuple[str, str]]]:
        """Mutate offspring."""
        mutated = []
        settlements = list(self.settlement_characteristics.keys())
        
        for individual in offspring:
            if np.random.random() < mutation_rate:
                # Random mutation
                mutated_individual = individual.copy()
                mutation_index = np.random.randint(len(mutated_individual))
                sector_id, _ = mutated_individual[mutation_index]
                
                # Choose random settlement
                settlement_id = np.random.choice(settlements)
                mutated_individual[mutation_index] = (sector_id, settlement_id)
                
                mutated.append(mutated_individual)
            else:
                mutated.append(individual)
        
        return mutated
    
    def _replace_population(self, population: List[List[Tuple[str, str]]], 
                          offspring: List[List[Tuple[str, str]]], 
                          fitness_scores: List[float]) -> List[List[Tuple[str, str]]]:
        """Replace population with offspring."""
        # Keep best individuals
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_size = len(population) // 4
        
        new_population = []
        
        # Keep elite
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]])
        
        # Add offspring
        new_population.extend(offspring[:len(population) - elite_size])
        
        return new_population
    
    def _individual_to_mappings(self, individual: List[Tuple[str, str]]) -> List[MappingResult]:
        """Convert individual to mapping results."""
        results = []
        
        for sector_id, settlement_id in individual:
            suitability = self._calculate_suitability(sector_id, settlement_id)
            production_capacity = self._calculate_production_capacity(sector_id, settlement_id)
            efficiency_factors = self._calculate_efficiency_factors(sector_id, settlement_id)
            
            result = MappingResult(
                sector_id=sector_id,
                settlement_id=settlement_id,
                suitability_score=suitability,
                production_capacity=production_capacity,
                efficiency_factors=efficiency_factors,
                constraints_satisfied=self._check_constraints(sector_id, settlement_id),
                optimization_notes="Genetic algorithm optimization"
            )
            
            results.append(result)
        
        return results
    
    def get_mapping_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of current mappings."""
        if not self.mapping_results:
            return {"success": False, "error": "No mapping results available"}
        
        try:
            # Sector type analysis
            sector_type_distribution = {}
            for result in self.mapping_results:
                sector = self.sector_characteristics[result.sector_id]
                sector_type = sector.sector_type.value
                if sector_type not in sector_type_distribution:
                    sector_type_distribution[sector_type] = []
                sector_type_distribution[sector_type].append(result)
            
            # Settlement hierarchy analysis
            hierarchy_distribution = {}
            for result in self.mapping_results:
                settlement = self.settlement_characteristics[result.settlement_id]
                hierarchy = settlement.hierarchy_level.value
                if hierarchy not in hierarchy_distribution:
                    hierarchy_distribution[hierarchy] = []
                hierarchy_distribution[hierarchy].append(result)
            
            # Efficiency analysis
            efficiency_analysis = self._analyze_efficiency_distribution()
            
            return {
                "success": True,
                "total_mappings": len(self.mapping_results),
                "sector_type_distribution": {
                    sector_type: len(results) for sector_type, results in sector_type_distribution.items()
                },
                "hierarchy_distribution": {
                    hierarchy: len(results) for hierarchy, results in hierarchy_distribution.items()
                },
                "efficiency_analysis": efficiency_analysis,
                "constraint_analysis": self._analyze_constraint_satisfaction(),
                "specialization_analysis": self._analyze_specialization_patterns()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate mapping analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_efficiency_distribution(self) -> Dict[str, Any]:
        """Analyze efficiency distribution of mappings."""
        if not self.mapping_results:
            return {}
        
        suitability_scores = [r.suitability_score for r in self.mapping_results]
        
        return {
            "average_suitability": np.mean(suitability_scores),
            "suitability_std": np.std(suitability_scores),
            "min_suitability": np.min(suitability_scores),
            "max_suitability": np.max(suitability_scores),
            "high_efficiency_count": len([s for s in suitability_scores if s > 0.8]),
            "medium_efficiency_count": len([s for s in suitability_scores if 0.5 <= s <= 0.8]),
            "low_efficiency_count": len([s for s in suitability_scores if s < 0.5])
        }
    
    def _analyze_constraint_satisfaction(self) -> Dict[str, Any]:
        """Analyze constraint satisfaction."""
        if not self.mapping_results:
            return {}
        
        satisfied_count = sum(1 for r in self.mapping_results if r.constraints_satisfied)
        total_count = len(self.mapping_results)
        
        return {
            "satisfied_constraints": satisfied_count,
            "total_constraints": total_count,
            "satisfaction_rate": satisfied_count / total_count if total_count > 0 else 0.0,
            "violation_count": total_count - satisfied_count
        }
    
    def _analyze_specialization_patterns(self) -> Dict[str, Any]:
        """Analyze specialization patterns in mappings."""
        if not self.mapping_results:
            return {}
        
        # Count sectors per settlement
        settlement_sector_counts = {}
        for result in self.mapping_results:
            settlement_id = result.settlement_id
            settlement_sector_counts[settlement_id] = settlement_sector_counts.get(settlement_id, 0) + 1
        
        # Analyze specialization
        specialization_levels = list(settlement_sector_counts.values())
        
        return {
            "average_sectors_per_settlement": np.mean(specialization_levels),
            "max_sectors_per_settlement": np.max(specialization_levels),
            "min_sectors_per_settlement": np.min(specialization_levels),
            "specialized_settlements": len([s for s in specialization_levels if s > 3]),
            "diversified_settlements": len([s for s in specialization_levels if s <= 2])
        }
