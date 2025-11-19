## Release Notes

### Version 2.0.1

**What's New**:
- Enhanced README with modern API examples
- Improved CI/CD pipeline with security scanning integration
- Fixed import sorting across entire codebase
- Updated documentation to match current API implementation

**Fixed**:
- Version consistency across all namespaces (groundeddino_vl and groundingdino)
- README Quick Start examples now match actual API signatures
- Removed redundant GPU workflow from ci.yml
- Fixed import formatting across 12 files (isort compliance)
- Resolved all flake8 style violations

**Dependencies**:
- torch >= 2.7.0, < 3.0
- All 13 core dependencies properly versioned
- Zero known vulnerabilities

**Tested Environments**:
- Python 3.10, 3.11, 3.12, 3.13
- PyTorch 2.7+ (CPU and CUDA 12.8)
- Linux (ubuntu-latest)

**Installation**:
```bash
pip install groundeddino_vl
```

---

## Sign-Off

**Package Status**: ✅ **APPROVED FOR RELEASE**

**Recommendation**: The package meets all PyPI requirements and is production-ready.

**Last Validated**: 2025-11-19

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-19