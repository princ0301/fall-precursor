# Coding Style

Comments explain why, not what. A line of code that is already clear from reading it gets no comment. Comment only where the reasoning is not obvious from the code itself: a non-obvious choice, a workaround, a formula taken from a paper, a unit or shape assumption that would otherwise cause a silent bug.

No emojis anywhere in code, comments, commit messages, or CLI output.

No unnecessary blank lines. One blank line between logically separate blocks inside a function is fine. No blank lines at the start or end of a function body, no double blank lines inside a function.

Every public function has a docstring: one line stating what it does, plus input/output shapes or types where not obvious from type hints.

Type hints on all function signatures.

Functions stay small and single-purpose. If a function needs a comment to explain what section 2 of it does, it should be two functions instead.
