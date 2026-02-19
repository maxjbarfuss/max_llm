# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in max_llm, please **do not** open a public issue. Instead, use [GitHub's private vulnerability reporting](https://github.com/maxjbarfuss/max_llm/security/advisories/new) with:

1. Description of the vulnerability
2. Steps to reproduce (if applicable)
3. Potential impact assessment
4. Any suggested fixes

We will respond to security reports within 48 hours and work with you to develop and deploy a fix.

## Security Updates

- Security patches will be released as soon as practical
- Updates will include CVE references where applicable
- All dependencies are monitored via `pip list --outdated` and pre-commit hooks

## Supported Versions

| Version | Status | Security Updates |
|---------|--------|------------------|
| 0.1.x   | Current | Yes |

## Security Best Practices

When using max_llm:

- Keep dependencies updated: `pip install --upgrade -e ".[dev]"`
- Use pre-commit hooks to catch issues early (see CONTRIBUTING.md)
- Review type annotations and tests before running untrusted model code
- Validate all TOML configuration files before deployment
- Run tests in isolated environments: `pytest` with `--cov`

## Dependencies

All direct dependencies are pinned in `requirements.txt` and `pyproject.toml`. Regular audits are performed via:

```bash
pip-audit
safety check
```

## License

This project is licensed under Apache License 2.0. See [LICENSE](LICENSE) for details.
