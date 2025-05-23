from src.config.rag_config import KNOWLEDGE_BASE_CONFIG
from src.config.evaluation_config import create_evaluator_model
from src.data.data_loader import get_data_loader
from src.rag.rag_handler import RAGHandler
from src.evaluation.test_case_generator import TestCaseGenerator
from src.evaluation.evaluator import Evaluator
from src.reporting.report_generator import ReportGenerator

def main():
    # Initialize components
    data_loader = get_data_loader()
    rag_handler = RAGHandler(KNOWLEDGE_BASE_CONFIG["knowledge_base_id"])  # Uses its own RAG model
    test_generator = TestCaseGenerator(rag_handler, data_loader)
    
    # Create evaluator with separate model
    eval_model = create_evaluator_model()  # Uses model specified in AWS_EVALUATOR_MODEL_ID
    evaluator = Evaluator(eval_model)
    report_generator = ReportGenerator()

    # Generate test cases
    print("Generating test cases...")
    test_cases = test_generator.generate_test_cases()

    # Run evaluation
    print("Running evaluation...")
    evaluation_results = evaluator.evaluate_test_cases(test_cases)

    # Generate reports
    print("Generating reports...")
    summary = report_generator.generate_summary_report(evaluation_results)
    details = report_generator.generate_detailed_report(evaluation_results)
    
    # Save reports
    report_generator.save_reports(summary, details)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total Test Cases: {summary['total_test_cases']}")
    print(f"Passed Test Cases: {summary['passed_test_cases']}")
    print(f"Pass Rate: {summary['pass_rate']:.2%}")

if __name__ == "__main__":
    main()