"""
Demo of debug mode vs production mode.

This script demonstrates how exceptions are handled differently
in debug mode vs production mode.
"""

import asyncio
from agent_runtime_framework import configure
from agent_runtime_framework.executor.loop import CallableToolExecutor


async def demo_production_mode():
    """Demo production mode - exceptions are caught and returned as errors."""
    print("\n" + "="*60)
    print("PRODUCTION MODE - Exceptions are caught gracefully")
    print("="*60)
    
    configure(debug=False)
    
    def divide(a: int, b: int) -> str:
        """Divide two numbers."""
        result = a / b  # This will raise ZeroDivisionError if b=0
        return f"Result: {result}"
    
    executor = CallableToolExecutor({"divide": divide})
    
    # This will catch the exception and return an error message
    result = await executor.execute("divide", {"a": 10, "b": 0})
    print(f"\nResult: {result}")
    print("✅ Agent continues running despite the error")


async def demo_debug_mode():
    """Demo debug mode - exceptions propagate immediately."""
    print("\n" + "="*60)
    print("DEBUG MODE - Exceptions propagate for debugging")
    print("="*60)
    
    configure(debug=True)
    
    def divide(a: int, b: int) -> str:
        """Divide two numbers."""
        result = a / b  # This will raise ZeroDivisionError if b=0
        return f"Result: {result}"
    
    executor = CallableToolExecutor({"divide": divide})
    
    # This will raise the exception immediately
    try:
        result = await executor.execute("divide", {"a": 10, "b": 0})
        print(f"\nResult: {result}")
    except ZeroDivisionError as e:
        print(f"\n❌ Exception raised: {type(e).__name__}: {e}")
        print("✅ Full stack trace available for debugging!")
        import traceback
        traceback.print_exc()


async def demo_successful_call():
    """Demo a successful tool call."""
    print("\n" + "="*60)
    print("SUCCESSFUL CALL - Works in both modes")
    print("="*60)
    
    def divide(a: int, b: int) -> str:
        """Divide two numbers."""
        result = a / b
        return f"Result: {result}"
    
    executor = CallableToolExecutor({"divide": divide})
    
    # This works fine
    result = await executor.execute("divide", {"a": 10, "b": 2})
    print(f"\nResult: {result}")


async def main():
    """Run all demos."""
    print("\n🎯 Agent Runtime Framework - Debug Mode Demo")
    print("=" * 60)
    
    # Demo successful call
    await demo_successful_call()
    
    # Demo production mode
    await demo_production_mode()
    
    # Demo debug mode
    await demo_debug_mode()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("• Production mode: Exceptions caught → error messages")
    print("• Debug mode: Exceptions propagate → full stack traces")
    print("• Use debug mode during development")
    print("• Use production mode in deployment")
    print("\nSet via code: configure(debug=True)")
    print("Set via env: AGENT_RUNTIME_DEBUG=1")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

