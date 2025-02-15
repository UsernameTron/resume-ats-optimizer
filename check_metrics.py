"""
Quick validation script to check current metrics
"""
from app.core.matchers.semantic_matcher import SemanticMatcher
from app.validation.cs_validator import CSValidator

def main():
    # Initialize components
    matcher = SemanticMatcher()
    validator = CSValidator()
    
    # Run validation
    metrics = validator.validate_skill_extraction(matcher)
    
    # Print metrics
    print("\nValidation Metrics:")
    print("-" * 50)
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"Precision: {metrics.precision:.2%}")
    print(f"Recall: {metrics.recall:.2%}")
    print(f"F1 Score: {metrics.f1_score:.2%}")
    print(f"False Positive Rate: {metrics.false_positive_rate:.2%}")
    print("\nSkill Coverage:")
    print("-" * 50)
    for category, coverage in metrics.skill_coverage.items():
        print(f"{category}: {coverage:.2%}")
    print("\nPerformance:")
    print("-" * 50)
    print(f"Memory Usage: {metrics.memory_usage:.1f}MB")
    print(f"Processing Time: {metrics.processing_time:.1f}ms")
    print(f"Batch Performance: {metrics.batch_performance:.1f} docs/sec")

if __name__ == "__main__":
    main()
