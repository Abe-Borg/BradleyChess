import os
import argparse
import pytest

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run tests and generate HTML report.")
    parser.add_argument("test_class", help="The name of the test class to run (e.g., TestEnviron).")
    args = parser.parse_args()

    # Define the test class from the CLI input
    test_class = args.test_class

    # Define the expected test file path
    test_file = f"test_{test_class}.py"
    test_file_path = os.path.join(os.getcwd(), test_file)  # Use os.getcwd() to ensure the correct path

    # Check if the test file exists
    if not os.path.isfile(test_file_path):
        print(f"Error: Test file '{test_file_path}' not found.")
        return

    # Define the report filename
    report_filename = f"{test_class}_report.html"
    report_path = os.path.join("test_reports", report_filename)

    # Run pytest with the specified parameters
    pytest.main([
        test_file_path,  # Adjust this to the actual test file path
        "--maxfail=1",
        "-qq",  # Use -q for quiet or -qq for very quiet
        f"--html={report_path}",
        "--self-contained-html"
    ])

if __name__ == "__main__":
    main()
