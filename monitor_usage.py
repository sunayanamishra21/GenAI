#!/usr/bin/env python3
"""
Monitoring script for RAG system usage and costs
"""

import json
import time
from datetime import datetime
import config

class UsageMonitor:
    def __init__(self, log_file="usage_log.json"):
        self.log_file = log_file
        self.load_usage_data()
    
    def load_usage_data(self):
        """Load existing usage data from file"""
        try:
            with open(self.log_file, 'r') as f:
                self.usage_data = json.load(f)
        except FileNotFoundError:
            self.usage_data = {
                "sessions": [],
                "total_queries": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "created_at": datetime.now().isoformat()
            }
    
    def save_usage_data(self):
        """Save usage data to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.usage_data, f, indent=2)
    
    def log_query(self, query, results_count, tokens_used=0, cost=0.0):
        """Log a query and its usage"""
        session = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results_count": results_count,
            "tokens_used": tokens_used,
            "cost": cost
        }
        
        self.usage_data["sessions"].append(session)
        self.usage_data["total_queries"] += 1
        self.usage_data["total_tokens"] += tokens_used
        self.usage_data["total_cost"] += cost
        
        self.save_usage_data()
    
    def get_usage_stats(self):
        """Get usage statistics"""
        total_queries = self.usage_data["total_queries"]
        total_tokens = self.usage_data["total_tokens"]
        total_cost = self.usage_data["total_cost"]
        
        if total_queries > 0:
            avg_tokens = total_tokens / total_queries
            avg_cost = total_cost / total_queries
        else:
            avg_tokens = 0
            avg_cost = 0
        
        return {
            "total_queries": total_queries,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_tokens_per_query": avg_tokens,
            "avg_cost_per_query": avg_cost
        }
    
    def estimate_cost(self, tokens, model=None):
        """Estimate cost based on tokens and model"""
        if model is None:
            model = config.LLM_MODEL
        
        # OpenAI pricing (per 1K tokens)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.002, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015}
        }
        
        if model in pricing:
            # Assume 70% input, 30% output tokens
            input_tokens = int(tokens * 0.7)
            output_tokens = int(tokens * 0.3)
            
            input_cost = (input_tokens / 1000) * pricing[model]["input"]
            output_cost = (output_tokens / 1000) * pricing[model]["output"]
            
            return input_cost + output_cost
        
        return 0.0
    
    def display_usage_report(self):
        """Display usage report"""
        stats = self.get_usage_stats()
        
        print("=== RAG System Usage Report ===")
        print()
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        print()
        print(f"Average Tokens per Query: {stats['avg_tokens_per_query']:.1f}")
        print(f"Average Cost per Query: ${stats['avg_cost_per_query']:.4f}")
        print()
        
        # Show recent queries
        if self.usage_data["sessions"]:
            print("Recent Queries:")
            print("-" * 40)
            recent_sessions = self.usage_data["sessions"][-5:]
            for session in recent_sessions:
                timestamp = session["timestamp"][:19]  # Remove microseconds
                query = session["query"][:50] + "..." if len(session["query"]) > 50 else session["query"]
                tokens = session["tokens_used"]
                cost = session["cost"]
                print(f"{timestamp} | {query}")
                print(f"  Tokens: {tokens}, Cost: ${cost:.4f}")
                print()
        
        # Cost projections
        if stats["total_queries"] > 0:
            print("Cost Projections:")
            print("-" * 40)
            daily_queries = stats["total_queries"]  # Assuming this is daily usage
            monthly_cost = stats["total_cost"] * 30
            yearly_cost = stats["total_cost"] * 365
            
            print(f"Daily Cost: ${stats['total_cost']:.4f}")
            print(f"Monthly Cost (projected): ${monthly_cost:.2f}")
            print(f"Yearly Cost (projected): ${yearly_cost:.2f}")
            print()
        
        # Recommendations
        print("Recommendations:")
        print("-" * 40)
        if stats["avg_tokens_per_query"] > 1000:
            print("- Consider reducing MAX_TOKENS in config.py")
        if stats["total_cost"] > 10:
            print("- Monitor usage closely to avoid high costs")
        if stats["total_queries"] > 100:
            print("- Consider using cheaper models for simple queries")
        
        print("- Use fallback mode for testing to avoid costs")
        print("- Set up usage alerts if available")
        print()

def main():
    """Main monitoring function"""
    monitor = UsageMonitor()
    
    print("RAG System Usage Monitor")
    print("=" * 50)
    print()
    
    # Display current usage
    monitor.display_usage_report()
    
    # Show configuration
    print("Current Configuration:")
    print(f"  LLM Provider: {config.LLM_PROVIDER}")
    print(f"  Model: {config.LLM_MODEL}")
    print(f"  Max Tokens: {config.MAX_TOKENS}")
    print(f"  Temperature: {config.TEMPERATURE}")
    print()
    
    # Cost estimation for different usage levels
    print("Cost Estimation (per query):")
    print("-" * 40)
    
    test_tokens = [100, 500, 1000, 2000]
    for tokens in test_tokens:
        cost = monitor.estimate_cost(tokens)
        print(f"  {tokens:,} tokens: ${cost:.4f}")
    
    print()
    print("Usage Log File:", monitor.log_file)
    print("To reset usage data, delete the log file and restart.")

if __name__ == "__main__":
    main()
