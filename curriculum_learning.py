"""
Curriculum Learning System for Durotaxis Training
Implements progressive learning stages with reward shaping
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CurriculumStage(Enum):
    """Enumeration of curriculum learning stages"""
    NAVIGATION = "navigation"
    MANAGEMENT = "management" 
    OPTIMIZATION = "optimization"


@dataclass
class StageMetrics:
    """Metrics for evaluating stage completion"""
    episode_lengths: List[int]
    boundary_violations: List[int]
    spawn_actions: List[int]
    delete_actions: List[int]
    rightward_progress: List[float]
    success_completions: List[bool]
    efficiency_scores: List[float]


class CurriculumManager:
    """Manages curriculum learning progression and reward shaping"""
    
    def __init__(self, config: Dict):
        self.config = config.get('curriculum_learning', {})
        self.curriculum_enabled = self.config.get('enable_curriculum', False)
        
        if not self.curriculum_enabled:
            return
            
        # Stage definitions
        self.stages = {
            CurriculumStage.NAVIGATION: self.config.get('stage_1_navigation', {}),
            CurriculumStage.MANAGEMENT: self.config.get('stage_2_management', {}),
            CurriculumStage.OPTIMIZATION: self.config.get('stage_3_optimization', {})
        }
        
        # Progression settings
        self.progression = self.config.get('curriculum_progression', {})
        self.reward_shaping = self.config.get('reward_shaping', {})
        
        # Current state
        self.current_stage = CurriculumStage.NAVIGATION
        self.current_episode = 0
        self.stage_metrics = StageMetrics([], [], [], [], [], [], [])
        self.milestones_achieved = set()
        
        # Stage transition tracking
        self.stage_start_episode = 0
        self.auto_advance = self.progression.get('auto_advance', True)
        self.min_success_rate = self.progression.get('min_success_rate', 0.6)
        self.evaluation_window = self.progression.get('evaluation_window', 50)
        
        print(f"🎓 Curriculum Learning Initialized")
        print(f"   📚 Stage 1: Navigation ({self.stages[CurriculumStage.NAVIGATION]['episode_start']}-{self.stages[CurriculumStage.NAVIGATION]['episode_end']})")
        print(f"   📚 Stage 2: Management ({self.stages[CurriculumStage.MANAGEMENT]['episode_start']}-{self.stages[CurriculumStage.MANAGEMENT]['episode_end']})")
        print(f"   📚 Stage 3: Optimization ({self.stages[CurriculumStage.OPTIMIZATION]['episode_start']}-{self.stages[CurriculumStage.OPTIMIZATION]['episode_end']})")
    
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage"""
        if not self.curriculum_enabled:
            return CurriculumStage.OPTIMIZATION
        return self.current_stage
    
    def should_advance_stage(self) -> bool:
        """Check if we should advance to the next stage"""
        if not self.curriculum_enabled or not self.auto_advance:
            return False
            
        # Check episode-based advancement
        stage_config = self.stages[self.current_stage]
        if self.current_episode >= stage_config.get('episode_end', float('inf')):
            return True
            
        # Check success-rate based advancement
        if len(self.stage_metrics.success_completions) >= self.evaluation_window:
            recent_success_rate = np.mean(self.stage_metrics.success_completions[-self.evaluation_window:])
            return recent_success_rate >= self.min_success_rate
            
        return False
    
    def advance_stage(self):
        """Advance to the next curriculum stage"""
        if self.current_stage == CurriculumStage.NAVIGATION:
            self.current_stage = CurriculumStage.MANAGEMENT
            print(f"🎓 Advanced to Stage 2: Node Management (Episode {self.current_episode})")
        elif self.current_stage == CurriculumStage.MANAGEMENT:
            self.current_stage = CurriculumStage.OPTIMIZATION
            print(f"🎓 Advanced to Stage 3: Full Optimization (Episode {self.current_episode})")
        
        # Reset stage metrics
        self.stage_metrics = StageMetrics([], [], [], [], [], [], [])
        self.stage_start_episode = self.current_episode
    
    def update_episode(self, episode_num: int, episode_data: Dict):
        """Update curriculum state with episode results"""
        self.current_episode = episode_num
        
        if not self.curriculum_enabled:
            return
            
        # Extract episode metrics
        episode_length = episode_data.get('steps', 0)
        boundary_violations = episode_data.get('boundary_violations', 0)
        spawn_actions = episode_data.get('spawn_actions', 0)
        delete_actions = episode_data.get('delete_actions', 0)
        rightward_progress = episode_data.get('rightward_progress', 0.0)
        success = episode_data.get('success', False)
        efficiency = episode_data.get('efficiency', 0.0)
        
        # Update stage metrics
        self.stage_metrics.episode_lengths.append(episode_length)
        self.stage_metrics.boundary_violations.append(boundary_violations)
        self.stage_metrics.spawn_actions.append(spawn_actions)
        self.stage_metrics.delete_actions.append(delete_actions)
        self.stage_metrics.rightward_progress.append(rightward_progress)
        self.stage_metrics.success_completions.append(success)
        self.stage_metrics.efficiency_scores.append(efficiency)
        
        # Check for stage advancement
        if self.should_advance_stage():
            self.advance_stage()
    
    def get_reward_multipliers(self) -> Dict[str, float]:
        """Get reward multipliers for current stage"""
        if not self.curriculum_enabled:
            return {}
            
        stage_config = self.stages[self.current_stage]
        return stage_config.get('reward_multipliers', {})
    
    def get_environment_settings(self) -> Dict[str, any]:
        """Get environment settings for current stage"""
        if not self.curriculum_enabled:
            return {}
            
        stage_config = self.stages[self.current_stage]
        settings = {}
        
        # Max nodes allowed
        if 'max_nodes_allowed' in stage_config:
            settings['max_nodes_override'] = stage_config['max_nodes_allowed']
            
        # Action space modifications
        if stage_config.get('simplified_actions', False):
            settings['simplified_actions'] = True
            
        # Other stage-specific settings
        if 'unlock_advanced_actions' in stage_config:
            settings['advanced_actions'] = stage_config['unlock_advanced_actions']
            
        return settings
    
    def get_success_criteria(self) -> Dict[str, float]:
        """Get success criteria for current stage"""
        if not self.curriculum_enabled:
            return {}
            
        stage_config = self.stages[self.current_stage]
        return stage_config.get('success_criteria', {})
    
    def check_milestones(self, episode_data: Dict) -> List[str]:
        """Check for milestone achievements and return bonuses"""
        if not self.curriculum_enabled:
            return []
            
        milestones = self.reward_shaping.get('milestones', {})
        achieved_milestones = []
        
        # Check each milestone
        for milestone_name, bonus_value in milestones.items():
            if milestone_name in self.milestones_achieved:
                continue
                
            achieved = False
            
            if milestone_name == 'first_successful_navigation':
                if episode_data.get('rightward_progress', 0) >= 10.0 and episode_data.get('boundary_violations', 1) == 0:
                    achieved = True
                    
            elif milestone_name == 'first_successful_spawn':
                if episode_data.get('spawn_actions', 0) >= 1:
                    achieved = True
                    
            elif milestone_name == 'first_successful_deletion':
                if episode_data.get('delete_actions', 0) >= 1:
                    achieved = True
                    
            elif milestone_name == 'first_long_episode':
                if episode_data.get('steps', 0) >= 30:
                    achieved = True
                    
            elif milestone_name == 'boundary_violation_free':
                if episode_data.get('boundary_violations', 1) == 0 and episode_data.get('steps', 0) >= 10:
                    achieved = True
                    
            elif milestone_name == 'task_completion':
                if episode_data.get('success', False):
                    achieved = True
            
            if achieved:
                self.milestones_achieved.add(milestone_name)
                achieved_milestones.append(milestone_name)
                print(f"🏆 Milestone Achieved: {milestone_name} (+{bonus_value} reward)")
        
        return achieved_milestones
    
    def get_milestone_bonus(self, milestone_name: str) -> float:
        """Get bonus reward for achieving a milestone"""
        milestones = self.reward_shaping.get('milestones', {})
        return milestones.get(milestone_name, 0.0)
    
    def get_stage_info(self) -> Dict[str, any]:
        """Get information about current stage"""
        if not self.curriculum_enabled:
            return {"stage": "disabled", "description": "Curriculum learning disabled"}
            
        stage_config = self.stages[self.current_stage]
        
        return {
            "stage": self.current_stage.value,
            "description": stage_config.get('description', ''),
            "focus": stage_config.get('focus', ''),
            "episode_range": f"{stage_config.get('episode_start', 0)}-{stage_config.get('episode_end', 'end')}",
            "current_episode": self.current_episode,
            "episodes_in_stage": self.current_episode - self.stage_start_episode,
            "milestones_achieved": len(self.milestones_achieved),
            "recent_success_rate": np.mean(self.stage_metrics.success_completions[-10:]) if len(self.stage_metrics.success_completions) >= 10 else 0.0
        }
    
    def get_stage_progress_summary(self) -> str:
        """Get a summary string of current stage progress"""
        if not self.curriculum_enabled:
            return "Curriculum: Disabled"
            
        info = self.get_stage_info()
        return f"Stage: {info['stage'].title()} | Ep: {info['current_episode']} | Focus: {info['focus']} | Milestones: {info['milestones_achieved']}"


def create_curriculum_manager(config: Dict) -> CurriculumManager:
    """Factory function to create curriculum manager"""
    return CurriculumManager(config)