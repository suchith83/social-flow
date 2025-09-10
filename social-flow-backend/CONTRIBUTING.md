# Contributing Guidelines
Contributing to Social Flow Backend
Thank you for your interest in contributing to Social Flow! This document outlines the guidelines for contributing to the backend repository, ensuring a consistent and collaborative development process.
Getting Started

Fork the Repository:

Fork the repository on GitHub and clone your fork:git clone https://github.com/nirmal-mina/social-flow-backend.git
cd social-flow-backend




Set Up Development Environment:

Install prerequisites (Docker, Go, Node.js, Python, Java, Kotlin, Terraform).
Run the setup script:./scripts/setup/setup.sh


Configure environment variables in config/environments/development/config.yaml.


Run Tests:

Execute unit tests:./scripts/testing/run_unit.sh


Execute integration tests:./scripts/testing/run_integration.sh





Contribution Process

Create an Issue:

Before starting work, create or claim an issue in the GitHub Issues section.
Describe the feature, bug, or improvement clearly, referencing relevant documentation or designs.


Create a Branch:

Create a feature branch from main:git checkout -b feature/<issue-number>-description


Example: git checkout -b feature/123-add-user-authentication.


Code Standards:

Go: Follow Go style guidelines, use gofmt and golint.
Node.js: Use ESLint with the provided .eslintrc configuration.
Python: Adhere to PEP 8, use pylint for linting.
Kotlin: Use detekt for static analysis.
General: Write clear, concise code with comments for complex logic.


Write Tests:

Add unit tests in services/<service>/tests/unit.
Add integration tests in services/<service>/tests/integration.
Ensure test coverage remains above 80%.


Commit Guidelines:

Use conventional commits (e.g., feat: add user authentication, fix: resolve video upload bug).
Keep commits small and focused.
Reference the issue number in the commit message (e.g., Closes #123).


Submit a Pull Request:

Push your branch to your fork:git push origin feature/<issue-number>-description


Open a pull request against the main branch.
Include a detailed description, referencing the issue and any design documents.
Ensure all CI checks pass (linting, tests, code coverage).


Code Review:

At least two reviewers must approve the PR.
Address feedback promptly and make necessary changes.
Ensure the PR is merged using squash-and-merge to maintain a clean history.



Development Guidelines

Modularity: Follow the microservices architecture, keeping services independent and loosely coupled.
Security: Use secure coding practices, validate inputs, and avoid hardcoding sensitive data.
Performance: Optimize for low latency and high throughput, especially for video and recommendation services.
Documentation: Update relevant documentation (docs/*) for any new features or changes.
Testing: Include unit, integration, and performance tests for all changes.

Tools and Standards

Code Quality: Use SonarQube, ESLint, Pylint, and Detekt for static analysis.
Testing Frameworks: Go (testing), Node.js (Jest), Python (pytest), Kotlin (JUnit).
CI/CD: Use GitLab CI or GitHub Actions for automated testing and deployment.
Monitoring: Integrate with Prometheus and Grafana for metrics.

Community Guidelines

Be respectful and inclusive, adhering to the Code of Conduct.
Provide constructive feedback during code reviews.
Help new contributors by answering questions and providing guidance.

Reporting Bugs

Open an issue with a clear title and description.
Include steps to reproduce, expected behavior, and actual behavior.
Attach relevant logs or screenshots.

Proposing Features

Create an issue with a detailed proposal, including use cases and benefits.
Discuss with the community before starting implementation.

Contact
For questions, contact the backend team at backend@socialflow.com or join our Slack channel.