# 🤝 **Contributing to Social Flow Backend**

Thank you for your interest in contributing to **Social Flow Backend**! This document outlines comprehensive guidelines for contributing to ensure a consistent, collaborative, and high-quality development process.

---

## 🚀 **Getting Started**

### **📥 Fork the Repository**

1. **Fork** the repository on GitHub by clicking the "Fork" button
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/social-flow-backend.git
   cd social-flow-backend
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/nirmal-mina/social-flow-backend.git
   ```

### **🔄 Stay Updated**

Keep your fork synchronized with the upstream repository:
```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## 🛠️ **Development Environment**

### **📋 Prerequisites**

- **Python 3.11+** with pip and virtualenv
- **Docker & Docker Compose** for containerization
- **PostgreSQL 15+** for database
- **Redis 7+** for caching
- **Git** for version control

### **⚙️ Setup Development Environment**

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start Development Services**:
   ```bash
   docker-compose up -d postgres redis
   ```

5. **Run Database Migrations**:
   ```bash
   alembic upgrade head
   ```

6. **Start the Application**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### **🧪 Verify Setup**

Run the test suite to ensure everything is working:
```bash
pytest
```

Access the API documentation at: http://localhost:8000/api/v1/docs

---

## 📝 **Contribution Process**

### **1. 📋 Create or Claim an Issue**

- **Search existing issues** before creating a new one
- **Use issue templates** for bug reports and feature requests
- **Provide detailed information** including:
  - Clear description of the problem/feature
  - Steps to reproduce (for bugs)
  - Expected vs actual behavior
  - Environment details

### **2. 🌿 Create a Feature Branch**

Create a new branch for your work:
```bash
git checkout -b feature/issue-number-short-description
```

**Branch naming conventions**:
- **Features**: `feature/123-add-user-authentication`
- **Bug fixes**: `bugfix/456-fix-video-upload`
- **Documentation**: `docs/789-update-api-docs`

### **3. 💻 Implement Your Changes**

- Follow our Code Standards
- Write comprehensive tests
- Update documentation as needed
- Ensure your code passes all CI checks

### **4. 🔄 Submit a Pull Request**

- Push your branch to your fork
- Create a pull request against the `main` branch
- Fill out the PR template completely
- Link to the related issue(s)

---

## 📏 **Code Standards**

### **🐍 Python Standards**

- **PEP 8** compliance for code style
- **Type hints** for all function parameters and return values
- **Docstrings** for all public functions and classes
- **Maximum line length**: 88 characters (Black formatter)

### **📁 File Organization**

- **Modules**: Keep files focused and cohesive
- **Imports**: Group imports (standard library, third-party, local)
- **Constants**: Use UPPER_CASE for constants
- **Classes**: Use PascalCase
- **Functions/Variables**: Use snake_case

### **🔒 Security Standards**

- **Never commit** secrets, API keys, or passwords
- **Use environment variables** for configuration
- **Validate all inputs** at API boundaries
- **Use parameterized queries** to prevent SQL injection

---

## 🧪 **Testing Guidelines**

### **📊 Coverage Requirements**

- **Minimum 85%** overall test coverage
- **95%+ coverage** for critical paths (authentication, payments)

### **🔬 Test Types**

- **Unit Tests** (`tests/unit/`) - Individual component testing
- **Integration Tests** (`tests/integration/`) - API endpoint testing
- **End-to-End Tests** (`tests/e2e/`) - Complete workflow testing

---

## 📝 **Commit Guidelines**

### **📋 Conventional Commits**

We use [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>
```

### **🏷️ Commit Types**

- **feat**: New feature implementation
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes
- **refactor**: Code refactoring without feature changes
- **test**: Adding or updating tests

### **📏 Commit Best Practices**

- **Keep commits atomic** (one logical change per commit)
- **Write clear, descriptive messages** in present tense
- **Reference issues** in commit messages (`Closes #123`, `Fixes #456`)

---

## 🔄 **Pull Request Process**

### **🔍 PR Guidelines**

- **Keep PRs focused** - one feature/fix per PR
- **Provide context** - explain the why, not just the what
- **Include tests** - ensure your changes are tested
- **Update documentation** - keep docs in sync with code

### **✅ Approval Requirements**

- **Minimum 2 approvals** from core team members
- **All CI checks** must pass (tests, linting, security scans)
- **No unresolved conflicts** with target branch

---

## 🤝 **Community Guidelines**

### **💝 Code of Conduct**

We follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). In summary:

- **Be welcoming** to newcomers
- **Be respectful** of different viewpoints and experiences
- **Accept constructive criticism** gracefully

---

## 🐛 **Reporting Issues**

### **📋 Bug Report Template**

```markdown
## 🐛 Bug Description
Clear description of the bug

## 📋 Steps to Reproduce
1. Step one
2. Step two
3. Step three

## ✅ Expected Behavior
What should happen

## ❌ Actual Behavior
What actually happens

## 🖥️ Environment
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.11.5]
- Version: [e.g. 1.2.3]
```

### **🔒 Security Issues**

For security vulnerabilities:
- **DO NOT** create a public issue
- **Email** security@socialflow.com

---

## 💡 **Feature Requests**

### **📝 Feature Request Template**

```markdown
## 🎯 Feature Summary
Brief description of the feature

## 🤔 Motivation
Why is this feature needed?

## 📋 Detailed Description
Comprehensive description of the feature
```

---

## 📞 **Contact & Support**

### **👥 Core Team**

- **Nirmal Meena** - Lead Backend Developer
  - GitHub: [@nirmal-mina](https://github.com/nirmal-mina)
  - LinkedIn: [Nirmal Mina](https://www.linkedin.com/in/nirmal-mina-4b0b951b2)
  - Mobile: +91 93516 88554

- **Sumit Sharma** - Backend Developer
  - Mobile: +91 93047 68420

- **Koduru Suchith** - Backend Developer  
  - Mobile: +91 84650 73250

### **📧 Email Contacts**

- **General Questions**: hello@socialflow.com
- **Technical Support**: tech@socialflow.com  
- **Security Issues**: security@socialflow.com
- **Business Inquiries**: business@socialflow.com

### **🔗 Links**

- **Documentation**: [docs.socialflow.com](https://docs.socialflow.com)
- **API Reference**: [api.socialflow.com/docs](https://api.socialflow.com/docs)
- **Discord Community**: [Join our Discord](https://discord.gg/socialflow)

---

## 🙏 **Acknowledgments**

Thank you to all contributors who have helped make Social Flow Backend better! Your contributions are what make this project successful.

---

**Happy Contributing! 🚀**

*We're excited to see what you'll build with Social Flow Backend!*