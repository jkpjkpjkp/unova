"""
Utility functions for logging intermediate results during graph execution.
This helps the optimizer understand what's happening inside the graph execution.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class IntermediateLogger:
    """
    A utility class to help log intermediate results during graph execution.
    This is designed to be used within graph.py files to log steps and 
    help the optimizer understand what's happening with the execution.
    """
    
    def __init__(self, log_path: Optional[str] = None, benchmark=None):
        """
        Initialize the logger with a log path or a benchmark instance.
        
        Args:
            log_path: The path where log.json should be saved
            benchmark: A BaseBenchmark instance used for evaluation
        """
        self.log_path = log_path
        self.benchmark = benchmark
        self._current_problem = None
    
    def set_current_problem(self, problem: str):
        """
        Set the current problem being processed, to be associated with intermediate logs.
        
        Args:
            problem: The MultimodalMessage representing the current problem
        """
        self._current_problem = problem
    
    def log_step(self, 
                 step_name: str, 
                 result: Any, 
                 additional_info: Optional[Dict[str, Any]] = None):
        """
        Log an intermediate step in the execution process.
        
        Args:
            step_name: Name of the step (e.g., "parse_input", "generate_solution", etc.)
            result: The result of this step
            additional_info: Additional information to include in the log
        """
        if self._current_problem is None:
            logger.warning("No current problem set for logging intermediate results")
            return
        
        if self.benchmark is not None:
            self.benchmark.log_intermediate_result(
                self._current_problem, 
                step_name, 
                result, 
                additional_info
            )
        elif self.log_path:
            # Implementation for direct file logging if benchmark is not available
            self._log_to_file(step_name, result, additional_info)
        else:
            logger.warning("No log path or benchmark provided for logging intermediate results")
    
    def _log_to_file(self, 
                    step_name: str, 
                    result: Any, 
                    additional_info: Optional[Dict[str, Any]] = None):
        """
        Directly log to a file when no benchmark instance is available.
        
        Args:
            step_name: Name of the step
            result: The result of this step
            additional_info: Additional information to include in the log
        """
        import datetime
        import json
        
        # Convert MM to serializable format
        serializable_problem = {
            "text": str(self._current_problem),
            "images_count": len(self._current_problem.images) if hasattr(self._current_problem, "images") else 0,
        }
        
        log_data = {
            "question": serializable_problem,
            "step_name": step_name,
            "intermediate_result": str(result),
            "timestamp": datetime.datetime.now().isoformat(),
            "is_intermediate": True,
        }
        
        if additional_info:
            log_data.update(additional_info)
            
        log_file = Path(self.log_path) / "log.json"
        
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
        else:
            data = []
            
        data.append(log_data)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)