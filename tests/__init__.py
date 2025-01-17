import pytest
import coverage

def run_tests():
    # Start coverage measurement
    cov = coverage.Coverage()
    cov.start()

    # Run pytest to execute the tests
    pytest.main()

    # Stop coverage measurement and save the report
    cov.stop()
    cov.save()

    # Generate the coverage report
    cov.report()

if __name__ == "__main__":
    run_tests()
