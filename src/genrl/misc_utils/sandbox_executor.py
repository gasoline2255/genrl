"""
Sandbox executor for running foreign code in a secure environment.

This module provides a class for executing arbitrary code in a sandboxed
environment with timeout handling and error checking.
"""

import asyncio
from typing import Optional, Any
from langchain_sandbox import PyodideSandbox

from genrl.logging_utils.global_defs import get_logger


class CodeSandboxExecutor:
    """Executor for running code in a sandboxed environment.
    
    This class manages a PyodideSandbox instance and provides methods for
    safely executing code with timeout handling and error checking.
    """
    
    def __init__(self, sandbox_timeout: int = 5, sandbox_memory_limit_mb: int = 100):
        """Initialize the sandbox executor.

        Args:
            sandbox_timeout: The timeout for the sandbox execution in seconds.
        """
        self.sandbox = PyodideSandbox(allow_net=True)
        self.logger = get_logger()
        self.sandbox_timeout = sandbox_timeout
        self.memory_limit_mb = sandbox_memory_limit_mb
    
    async def execute_async(self, code: str) -> Optional[Any]:
        """Execute code asynchronously in the sandbox with timeout.
        
        Args:
            code: The code to execute.
            
        Returns:
            The execution result object, or None if execution times out.
        """
        try:
            result = await asyncio.wait_for(
                self.sandbox.execute(code, timeout_seconds=self.sandbox_timeout, memory_limit_mb=self.memory_limit_mb), 
                timeout=self.sandbox_timeout
            )
            return result
        except asyncio.TimeoutError:
            self.logger.info(f"Code execution timed out after {self.sandbox_timeout} seconds.")
            return None
    
    def execute(self, code: str) -> Optional[Any]:
        """Execute code synchronously in the sandbox with timeout.
        
        Args:
            code: The code to execute.
            
        Returns:
            The execution result object, or None if execution times out.
        """
        return asyncio.run(self.execute_async(code=code))
    
    def check_execution_success(self, result: Optional[Any]) -> bool:
        """Check if the execution result indicates success.
        
        Args:
            result: The execution result from the sandbox.
            
        Returns:
            True if execution was successful, False otherwise.
        """
        if result is None:
            return False
        
        errors = result.stderr
        status = result.status
        
        if errors or status == 'error':
            return False
        
        return True
    
    def execute_with_validation(
        self, 
        code: str
    ) -> tuple[Optional[Any], bool]:
        """Execute code and return both result and success status.
        
        Args:
            code: The code to execute.
            
        Returns:
            A tuple of (result, success) where result is the execution result
            and success is a boolean indicating if execution was successful.
        """
        result = self.execute(code)
        success = self.check_execution_success(result)
        return result, success

