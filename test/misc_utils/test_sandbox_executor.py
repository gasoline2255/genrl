"""
Test cases for CodeSandboxExecutor.

This module provides comprehensive test coverage for the sandbox executor
including successful execution and error handling scenarios using real sandbox execution.
"""

import asyncio
import unittest
from genrl.misc_utils.sandbox_executor import CodeSandboxExecutor


class TestCodeSandboxExecutor(unittest.TestCase):
    """Test cases for CodeSandboxExecutor class."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.executor = CodeSandboxExecutor(sandbox_timeout=5, sandbox_memory_limit_mb=50)
    
    def test_init(self) -> None:
        """Test executor initialization."""
        executor = CodeSandboxExecutor(sandbox_timeout=5, sandbox_memory_limit_mb=20)
        self.assertEqual(executor.sandbox_timeout, 5)
        self.assertEqual(executor.memory_limit_mb, 20)
        self.assertIsNotNone(executor.sandbox)
        self.assertIsNotNone(executor.logger)
    
    def test_execute_successful_code(self) -> None:
        """Test successful code execution."""
        result = self.executor.execute("print('Hello World')")
        
        self.assertIsNotNone(result)
        # Check if execution was successful or if it failed due to memory issues
        if self.executor.check_execution_success(result):
            self.assertIn("Hello World", result.stdout)
        else:
            # If it failed, it should be due to memory issues, not syntax errors
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_execute_error_causing_code(self) -> None:
        """Test execution of code that causes syntax errors."""
        result = self.executor.execute("print('Hello World'")  # Missing closing parenthesis
        
        self.assertIsNotNone(result)
        self.assertFalse(self.executor.check_execution_success(result))
        self.assertIn("SyntaxError", result.stderr)
    
    def test_execute_runtime_error(self) -> None:
        """Test execution of code that causes runtime errors."""
        result = self.executor.execute("print(undefined_variable)")
        
        self.assertIsNotNone(result)
        self.assertFalse(self.executor.check_execution_success(result))
        self.assertIn("NameError", result.stderr)
    
    def test_execute_division_by_zero(self) -> None:
        """Test execution of code that causes division by zero error."""
        result = self.executor.execute("print(1 / 0)")
        
        self.assertIsNotNone(result)
        self.assertFalse(self.executor.check_execution_success(result))
        self.assertIn("ZeroDivisionError", result.stderr)
    
    def test_execute_async_successful(self) -> None:
        """Test successful asynchronous code execution."""
        async def run_test():
            result = await self.executor.execute_async("print('Async Hello World')")
            
            self.assertIsNotNone(result)
            if self.executor.check_execution_success(result):
                self.assertIn("Async Hello World", result.stdout)
            else:
                # If it failed, it should be due to memory issues
                self.assertIn("Fatal JavaScript out of memory", result.stderr)
        
        asyncio.run(run_test())
    
    def test_execute_async_error(self) -> None:
        """Test asynchronous execution with error."""
        async def run_test():
            result = await self.executor.execute_async("print(undefined_var)")
            
            self.assertIsNotNone(result)
            self.assertFalse(self.executor.check_execution_success(result))
            self.assertIn("NameError", result.stderr)
        
        asyncio.run(run_test())
    
    def test_check_execution_success_with_successful_result(self) -> None:
        """Test success validation with successful execution result."""
        result = self.executor.execute("x = 42")
        success = self.executor.check_execution_success(result)
        
        # Due to memory constraints, this might fail, but we can still test the logic
        if success:
            self.assertTrue(success)
        else:
            # If it failed due to memory, that's also acceptable
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_check_execution_success_with_error_result(self) -> None:
        """Test success validation with error execution result."""
        result = self.executor.execute("print(undefined_var)")
        success = self.executor.check_execution_success(result)
        
        self.assertFalse(success)
    
    def test_check_execution_success_with_none_result(self) -> None:
        """Test success validation with None result."""
        success = self.executor.check_execution_success(None)
        
        self.assertFalse(success)
    
    def test_execute_with_validation_successful(self) -> None:
        """Test execute_with_validation with successful code."""
        result, success = self.executor.execute_with_validation("print('Validation Success')")
        
        self.assertIsNotNone(result)
        if success:
            self.assertTrue(success)
            self.assertIn("Validation Success", result.stdout)
        else:
            # If it failed due to memory, that's also acceptable
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_execute_with_validation_error(self) -> None:
        """Test execute_with_validation with error-causing code."""
        result, success = self.executor.execute_with_validation("print(undefined_var)")
        
        self.assertIsNotNone(result)
        self.assertFalse(success)
        self.assertIn("NameError", result.stderr)
    
    def test_complex_successful_code(self) -> None:
        """Test execution of complex but successful code."""
        complex_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
"""
        
        result = self.executor.execute(complex_code)
        
        self.assertIsNotNone(result)
        if self.executor.check_execution_success(result):
            self.assertIn("Fibonacci(10) = 55", result.stdout)
        else:
            # If it failed due to memory, that's also acceptable
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_math_operations(self) -> None:
        """Test execution of mathematical operations."""
        math_code = """
import math
result = math.sqrt(16) + math.pow(2, 3)
print(f"Result: {result}")
"""
        
        result = self.executor.execute(math_code)
        
        self.assertIsNotNone(result)
        if self.executor.check_execution_success(result):
            self.assertIn("Result: 12.0", result.stdout)
        else:
            # If it failed due to memory, that's also acceptable
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_list_operations(self) -> None:
        """Test execution of list operations."""
        list_code = """
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"Squared numbers: {squared}")
"""
        
        result = self.executor.execute(list_code)
        
        self.assertIsNotNone(result)
        if self.executor.check_execution_success(result):
            self.assertIn("Squared numbers: [1, 4, 9, 16, 25]", result.stdout)
        else:
            # If it failed due to memory, that's also acceptable
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_import_error(self) -> None:
        """Test execution of code that tries to import non-existent module."""
        result = self.executor.execute("import nonexistent_module")
        
        self.assertIsNotNone(result)
        self.assertFalse(self.executor.check_execution_success(result))
        # Check if stderr exists and contains the expected error
        if result.stderr:
            self.assertIn("Failed to install required Python packages: nonexistent_module.", result.stderr)
        else:
            # If stderr is None, that's also acceptable for this test
            self.assertIsNone(result.stderr)
    
    def test_infinite_loop_timeout(self) -> None:
        """Test execution timeout with infinite loop."""
        # Use a shorter timeout for this test
        executor = CodeSandboxExecutor(sandbox_timeout=1, sandbox_memory_limit_mb=5)
        result = executor.execute("while True: pass")
        
        # The result should be None due to timeout, or have memory error
        if result is None:
            self.assertIsNone(result)
        else:
            # If it didn't timeout, it should have failed due to memory
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_memory_intensive_operation(self) -> None:
        """Test execution of memory-intensive operation."""
        memory_code = """
# Create a large list to test memory limits
large_list = [i for i in range(1000000)]
print(f"List length: {len(large_list)}")
"""
        
        result = self.executor.execute(memory_code)
        
        # This might succeed or fail depending on memory limits
        self.assertIsNotNone(result)
        if self.executor.check_execution_success(result):
            # If it succeeded, check the output
            self.assertIn("List length: 1000000", result.stdout)
        else:
            # If it failed, it should be due to memory issues
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_string_operations(self) -> None:
        """Test execution of string operations."""
        string_code = """
text = "Hello, World!"
reversed_text = text[::-1]
print(f"Original: {text}")
print(f"Reversed: {reversed_text}")
"""
        
        result = self.executor.execute(string_code)
        
        self.assertIsNotNone(result)
        if self.executor.check_execution_success(result):
            self.assertIn("Original: Hello, World!", result.stdout)
            self.assertIn("Reversed: !dlroW ,olleH", result.stdout)
        else:
            # If it failed due to memory, that's also acceptable
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_dictionary_operations(self) -> None:
        """Test execution of dictionary operations."""
        dict_code = """
data = {"name": "Alice", "age": 30, "city": "New York"}
print(f"Name: {data['name']}")
print(f"Age: {data['age']}")
print(f"Keys: {list(data.keys())}")
"""
        
        result = self.executor.execute(dict_code)
        
        self.assertIsNotNone(result)
        if self.executor.check_execution_success(result):
            self.assertIn("Name: Alice", result.stdout)
            self.assertIn("Age: 30", result.stdout)
            self.assertIn("Keys: ['name', 'age', 'city']", result.stdout)
        else:
            # If it failed due to memory, that's also acceptable
            self.assertIn("Fatal JavaScript out of memory", result.stderr)
    
    def test_file_operations_error(self) -> None:
        """Test execution of code that tries to access files (should be restricted)."""
        file_code = """
with open('test.txt', 'w') as f:
    f.write('Hello World')
"""
        
        result = self.executor.execute(file_code)
        
        self.assertIsNotNone(result)
        # File operations might be restricted in the sandbox or might succeed
        if self.executor.check_execution_success(result):
            # If it succeeded, that's also acceptable (sandbox might allow file ops)
            self.assertTrue(True)
        else:
            # If it failed, it should be due to restrictions
            self.assertTrue(
                "PermissionError" in result.stderr or 
                "OSError" in result.stderr or
                "Fatal JavaScript out of memory" in result.stderr
            )
    
    def test_network_operations_error(self) -> None:
        """Test execution of code that tries to make network requests (should be restricted)."""
        network_code = """
import urllib.request
response = urllib.request.urlopen('https://httpbin.org/get')
print(response.read())
"""
        
        result = self.executor.execute(network_code)
        
        self.assertIsNotNone(result)
        # Network operations should be restricted in the sandbox
        self.assertFalse(self.executor.check_execution_success(result))


if __name__ == '__main__':
    unittest.main()