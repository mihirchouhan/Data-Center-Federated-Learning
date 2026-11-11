# Authentication and security boundary

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

[NOT APPLICABLE] The inspected application is a server CLI, not a multi-user web service. No application login/signup/logout, JWT, OAuth, password reset or role database exists. College SSH accounts/filesystem access are infrastructure, not authentication implemented here.

[IMPLEMENTED] Particular operations use ownership/symlink checks and advisory locks. Source/config hashes, exact inventories, receipt chains, model locks and no-clobber outputs bind local evidence. Training-only APIs reduce accidental leakage. None authenticates remote FL clients, provides secure aggregation/differential privacy or protects against privileged/same-UID tampering.

Do not embed passwords/tokens/private keys in code, docs or logs. Preserve TLS/hostname verification. Do not alter shared-server permissions or administrator policy. Load only trusted local checkpoints with supported safeguards.

[PLANNED] Real deployment needs a threat model, authenticated participants, protected transport, credential/permission management, audit and security/privacy tests. This note implements none of them and asserts no guarantee. See [Integration](Integration.md), [MCP](MCP.md) and [Agents](Agents.md).
