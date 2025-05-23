from typing import List, Dict
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_summary_report(self, evaluation_results) -> dict:
        """
        Generate a summary of evaluation results
        
        Args:
            evaluation_results: Results from evaluator
            
        Returns:
            dict: Summary statistics
        """
        # Calculate overall statistics
        total_cases = len(evaluation_results.test_results)
        passed_cases = sum(1 for test_result in evaluation_results.test_results 
                         if all(metric.success for metric in test_result.metrics_data))
        
        # Calculate average score across all test cases
        total_score = sum(test_result.metrics_data[0].score 
                         for test_result in evaluation_results.test_results)
        avg_score = total_score / total_cases if total_cases > 0 else 0
        
        # Get model info from first test case (these should be constant)
        model_info = evaluation_results.test_results[0].metrics_data[0]
        
        # Calculate overall pass/fail based on pass rate and threshold
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0
        overall_success = pass_rate >= model_info.threshold
        
        # Create summary dictionary for run_evaluation.py compatibility
        summary = {
            "total_test_cases": total_cases,
            "passed_test_cases": passed_cases,
            "pass_rate": pass_rate,
            "metric_name": model_info.name,
            "score": avg_score,
            "threshold": model_info.threshold,
            "success": overall_success,
            "model_used": model_info.evaluation_model,
            "analysis": f"Overall pass rate: {pass_rate:.2%}. Average score: {avg_score:.4f}",
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # Create DataFrame format for CSV output (notebook format)
        self.summary_df_format = {
            'Metric': ['Timestamp', 'Total Test Cases', 'Overall Score', 'Pass/Fail', 'Model Used'],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                total_cases,
                f"{avg_score:.4f}",  # Using average score across all test cases
                'PASS' if overall_success else 'FAIL',  # Based on overall pass rate vs threshold
                model_info.evaluation_model
            ]
        }
        
        return summary
    
    def generate_detailed_report(self, evaluation_results) -> List[Dict]:
        """
        Generate detailed report for each test case
        
        Args:
            evaluation_results: Results from evaluator
            
        Returns:
            List[Dict]: Detailed results for each test case
        """
        detailed_results = []
        
        for idx, test_result in enumerate(evaluation_results.test_results, 1):
            metric_result = test_result.metrics_data[0]
            result = {
                'Test ID': idx,
                'Input Query': test_result.input,
                'Expected Output': test_result.expected_output,
                'Actual Output': test_result.actual_output,
                'Score': f"{metric_result.score:.4f}",
                'Threshold': metric_result.threshold,
                'Success': metric_result.success,
                'Reason': metric_result.reason
            }
            detailed_results.append(result)
            
        return detailed_results
        
    def save_reports(self, summary: dict, detailed_results: List[Dict]):
        """
        Save reports to CSV files
        
        Args:
            summary: Summary statistics
            detailed_results: Detailed test case results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrames using notebook format
        summary_df = pd.DataFrame(self.summary_df_format)
        detailed_df = pd.DataFrame(detailed_results)
        
        # Save as CSV files
        summary_file = self.output_dir / f"summary_report_{timestamp}.csv"
        detailed_file = self.output_dir / f"detailed_report_{timestamp}.csv"
        
        # Save with auto-adjusted column widths
        summary_df.to_csv(summary_file, index=False)
        detailed_df.to_csv(detailed_file, index=False)
        
        print(f"Reports generated successfully:")
        print(f"Summary: {summary_file}")
        print(f"Detailed: {detailed_file}")