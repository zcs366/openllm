#!/usr/bin/env python3
"""
OpenSkill Open-World Learning Demo
===================================

Proof-of-concept: OpenSkillAdapter learns skills from real documentation,
validates them with virtual tasks, and generates UnifiedSkillConfig.

Pipeline:
  1. Fetch/ingest real documentation (Python requests library quickstart)
  2. OpenSkillAdapter.extract skills via pattern matching
  3. Validate each skill with auto-generated virtual tasks
  4. Convert to UnifiedSkillConfig with full SourceTraceability
  5. Report metrics and results

Usage:
  python3 examples/openskill_demo.py
"""

import sys
import os
import json
import re
import textwrap
from datetime import datetime, timezone

# Ensure openllm is importable
sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.isn.adapters.openskill_adapter import (
    ExtractionConfidence,
    ExtractionRule,
    OpenSkillAdapter,
    SkillExtraction,
    SkillScore,
    VirtualTask,
)
from openllm.isn.unified_skill_config import (
    SkillFrameworkSource,
    SkillLifecycleState,
    SourceTraceability,
    UnifiedSkillConfig,
)


# ══════════════════════════════════════════════════════════════════════════════
# Real documentation content (Python requests library Quickstart)
# This is the actual content from requests.readthedocs.io/en/latest/user/quickstart/
# ══════════════════════════════════════════════════════════════════════════════

REQUESTS_QUICKSTART = """
# Quickstart — Requests 2.34.2 documentation

## Quickstart

Eager to get started? This page gives a good introduction in how to get started
with Requests. First, make sure that Requests is installed and up-to-date.

## Make a Request

Making a request with Requests is very simple. Begin by importing the Requests module:

```python
import requests
```

Now, let's try to get a webpage. For this example, let's get GitHub's public timeline:

```python
r = requests.get('https://api.github.com/events')
```

Now, we have a Response object called r. We can get all the information we need from this object.

Requests' simple API means that all forms of HTTP request are as obvious. For example,
this is how you make an HTTP POST request:

```python
r = requests.post('https://httpbin.org/post', data={'key': 'value'})
```

What about the other HTTP request types: PUT, DELETE, HEAD and OPTIONS? These are all just as simple:

```python
r = requests.put('https://httpbin.org/put', data={'key': 'value'})
r = requests.delete('https://httpbin.org/delete')
r = requests.head('https://httpbin.org/get')
r = requests.options('https://httpbin.org/get')
```

## Passing Parameters In URLs

You often want to send some sort of data in the URL's query string. Requests allows you to
provide these arguments as a dictionary of strings, using the params keyword argument:

```python
payload = {'key1': 'value1', 'key2': 'value2'}
r = requests.get('https://httpbin.org/get', params=payload)
```

You can see that the URL has been correctly encoded by printing the URL:

```python
print(r.url)
# https://httpbin.org/get?key2=value2&key1=value1
```

Note that any dictionary key whose value is None will not be added to the URL's query string.

## Response Content

We can read the content of the server's response:

```python
import requests
r = requests.get('https://api.github.com/events')
r.text
```

Requests will automatically decode content from the server. Most unicode charsets are
seamlessly decoded.

### Binary Response Content

You can also access the response body as bytes, for non-text requests:

```python
r.content
```

The gzip and deflate transfer-encodings are automatically decoded for you.

### JSON Response Content

There's also a builtin JSON decoder, in case you're dealing with JSON data:

```python
import requests
r = requests.get('https://api.github.com/events')
r.json()
```

## Custom Headers

If you'd like to add HTTP headers to a request, simply pass in a dict to the headers parameter:

```python
headers = {'User-Agent': 'custom-agent'}
r = requests.get('https://api.github.com/events', headers=headers)
```

## More Complex POST Requests

Typically, you want to send some form-encoded data — like an HTML form. Simply pass a
dictionary to the data parameter:

```python
payload = {'username': 'user', 'password': 'pass'}
r = requests.post('https://httpbin.org/post', data=payload)
```

### POST a Multipart-Encoded File

Requests makes it simple to upload Multipart-encoded files:

```python
files = {'file': open('report.xls', 'rb')}
r = requests.post('https://httpbin.org/post', files=files)
```

## Response Status Codes

We can check the response status code:

```python
r = requests.get('https://httpbin.org/get')
r.status_code
# 200
```

Requests also comes with a built-in status code lookup object:

```python
r.status_code == requests.codes.ok
# True
```

## Redirection and History

Requests will follow redirects for all verbs except HEAD. You can tell Requests to not
redirect using allow_redirects parameter:

```python
r = requests.get('https://github.com', allow_redirects=False)
r.status_code
# 301
r.history
# [<Response [301]>]
```

## Timeouts

You can tell Requests to stop waiting for a response after a given number of seconds:

```python
r = requests.get('https://github.com', timeout=5)
```

## Errors and Exceptions

Requests will raise an HTTPError if the HTTP request returned an unsuccessful status code.
All exceptions that requests can raise live in requests.exceptions:

```python
try:
    r = requests.get('https://httpbin.org/status/404')
    r.raise_for_status()
except requests.exceptions.HTTPError as err:
    print(f"HTTP error: {err}")
except requests.exceptions.ConnectionError:
    print("Connection error")
except requests.exceptions.Timeout:
    print("Request timed out")
```

## Session Objects

The Session object allows you to persist certain parameters across requests:

```python
s = requests.Session()
s.get('https://httpbin.org/cookies/set/sessioncookie/123456789')
r = s.get('https://httpbin.org/cookies')
print(r.text)
# {'cookies': {'sessioncookie': '123456789'}}
```

## SSL Certificate Verification

Requests verifies SSL certificates for HTTPS requests, just like a web browser.
By default, SSL verification is enabled:

```python
requests.get('https://github.com', verify=True)
```

You can also verify against a custom CA bundle:

```python
requests.get('https://github.com', verify='/path/to/certfile')
```

## Proxy Support

Requests supports HTTP proxies. To use an HTTP proxy, simply configure the proxies
parameter on any request method:

```python
proxies = {
    'http': 'http://10.10.1.10:3128',
    'https': 'http://10.10.1.10:1080',
}
requests.get('http://example.org', proxies=proxies)
```

## Authentication

Requests supports several types of authentication. The most common is HTTP Basic Auth:

```python
from requests.auth import HTTPBasicAuth
requests.get('https://api.github.com/user', auth=HTTPBasicAuth('user', 'pass'))
```

Or simply use the auth keyword argument:

```python
requests.get('https://api.github.com/user', auth=('user', 'pass'))
```

For other forms of authentication, see the official documentation.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Demo Runner
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    """Print a section header."""
    print(f"\n{'━' * 70}")
    print(f"  {title}")
    print(f"{'━' * 70}")


def demo():
    print("=" * 70)
    print("  OpenSkill Open-World Learning Demo")
    print("  Proof-of-Concept: Learning Skills from Documentation")
    print("=" * 70)
    print(f"  Source: Python requests library Quickstart docs")
    print(f"  URL: https://requests.readthedocs.io/en/latest/user/quickstart/")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Configure adapter with API-doc-tailored rules
    # ═══════════════════════════════════════════════════════════════════
    section("Step 1: Configure OpenSkillAdapter")

    # Custom rules tuned for API documentation — produces more diverse,
    # individually fingerprinted skills than the generic defaults
    api_doc_rules = [
        # HTTP method + URL pattern (e.g., "requests.get('...')")
        ExtractionRule(
            pattern=r"requests\.(get|post|put|delete|head|options|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
            confidence=ExtractionConfidence.HIGH,
            tag="api-method",
            name_template="{0}-{1}",
            description_template="HTTP {0} request to {1}",
        ),
        # Markdown code blocks with language tag
        ExtractionRule(
            pattern=r"```(\w+)\n(.*?)```",
            confidence=ExtractionConfidence.MEDIUM,
            tag="code-block",
            name_template="snippet-{0}",
            description_template="Code example: {0}",
            use_dotall=True,
        ),
        # Markdown headings (## Title)
        ExtractionRule(
            pattern=r"^#{1,3}\s+(.+?)¶?$",
            confidence=ExtractionConfidence.HIGH,
            tag="section",
            name_template="section-{0}",
            description_template="Documentation section: {0}",
        ),
        # Numbered step lists
        ExtractionRule(
            pattern=r"(?:^|\n)\s*\d+\.\s+(.+)",
            confidence=ExtractionConfidence.MEDIUM,
            tag="step-list",
            name_template="procedure",
            description_template="Procedure: {0}",
        ),
        # "How to" patterns
        ExtractionRule(
            pattern=r"(?:How to|How do you|Guide to)\s+(.+?)(?:\.|$)",
            confidence=ExtractionConfidence.MEDIUM,
            tag="howto",
            name_template="howto-{0}",
            description_template="How-to: {0}",
        ),
        # Exception patterns (raise/except/catch)
        ExtractionRule(
            pattern=r"(?:raise|except|catch)\s+(\w+(?:\.\w+)*)",
            confidence=ExtractionConfidence.HIGH,
            tag="error-handling",
            name_template="error-{0}",
            description_template="Error handling: {0}",
        ),
    ]

    adapter = OpenSkillAdapter(
        min_confidence=ExtractionConfidence.LOW,
        max_skills_per_source=30,
        auto_generate_tasks=True,
        task_count=3,
        doc_rules=api_doc_rules,
    )
    print(f"  min_confidence:        {adapter.min_confidence.value}")
    print(f"  max_skills_per_source: {adapter.max_skills_per_source}")
    print(f"  auto_generate_tasks:   {adapter.auto_generate_tasks}")
    print(f"  task_count:            {adapter.task_count}")
    print(f"  doc_rules count:       {len(adapter._doc_rules)}")
    print(f"  rules:                 {[r.tag for r in adapter._doc_rules]}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Learn skills from documentation
    # ═══════════════════════════════════════════════════════════════════
    section("Step 2: Learn Skills from Documentation")

    skills = adapter.learn_from_docs(
        content=REQUESTS_QUICKSTART,
        source_url="https://requests.readthedocs.io/en/latest/user/quickstart/",
        author="Kenneth Reitz & Requests Contributors",
        license="Apache-2.0",
    )

    print(f"  Raw extractions:       {adapter.extraction_log[-1]['raw_count']}")
    print(f"  After confidence:      {adapter.extraction_log[-1]['filtered_count']}")
    print(f"  After dedup:           {adapter.extraction_log[-1]['deduped_count']}")
    print(f"  Final skills:          {adapter.extraction_log[-1]['final_count']}")
    print()

    # Group by tag
    by_tag: dict[str, list] = {}
    for s in skills:
        tag = s.tags[0] if s.tags else "untagged"
        by_tag.setdefault(tag, []).append(s)

    print("  Skills by type:")
    for tag, items in sorted(by_tag.items()):
        print(f"    {tag:20s} → {len(items)} skill(s)")
    print()

    # Show top skills
    print("  Top extracted skills:")
    for i, s in enumerate(skills[:15], 1):
        desc_short = s.description[:65] + ("..." if len(s.description) > 65 else "")
        print(f"    {i:2d}. [{s.confidence.value:6s}] {s.name}")
        print(f"        {desc_short}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Validate extracted skills
    # ═══════════════════════════════════════════════════════════════════
    section("Step 3: Validate Skills with Virtual Tasks")

    all_scores: dict[str, list[SkillScore]] = {}
    total_passed = 0
    total_tasks = 0

    for skill in skills:
        scores = adapter.validate_skill(skill)
        all_scores[skill.name] = scores
        passed = sum(1 for s in scores if s.passed)
        total_passed += passed
        total_tasks += len(scores)

    print(f"  Skills validated:      {len(skills)}")
    print(f"  Total virtual tasks:   {total_tasks}")
    print(f"  Tasks passed:          {total_passed}")
    print(f"  Overall pass rate:     {total_passed / max(total_tasks, 1):.1%}")
    print()

    # Show validation details for a few skills
    print("  Validation details (first 5 skills):")
    for skill in skills[:5]:
        scores = all_scores[skill.name]
        pass_count = sum(1 for s in scores if s.passed)
        status = "✅" if pass_count == len(scores) else "⚠️"
        print(f"    {status} {skill.name}: {pass_count}/{len(scores)} passed")
        for s in scores:
            marker = "✓" if s.passed else "✗"
            detail = s.output if s.passed else s.error
            print(f"       {marker} [{s.task_id.split('-')[-1]}] {detail}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Generate UnifiedSkillConfig
    # ═══════════════════════════════════════════════════════════════════
    section("Step 4: Generate UnifiedSkillConfig")

    configs: list[UnifiedSkillConfig] = []
    for skill in skills:
        scores = all_scores.get(skill.name, [])
        config = adapter.to_unified_config(skill, validation_scores=scores)
        configs.append(config)

    print(f"  Configs generated:     {len(configs)}")
    print()

    # Show sample configs
    print("  Sample UnifiedSkillConfig (first 3):")
    for i, config in enumerate(configs[:3], 1):
        print(f"\n    ┌─ Config #{i} ─────────────────────────────────────")
        print(f"    │ name:              {config.name}")
        print(f"    │ description:       {config.description[:60]}...")
        print(f"    │ version:           {config.version}")
        print(f"    │ lifecycle_state:   {config.lifecycle_state.value}")
        print(f"    │ source_framework:  {config.source_framework.value}")
        print(f"    │ domain_tags:       {config.domain_tags}")
        print(f"    │ curator_score:     {config.curator_score}")
        print(f"    │ risk_level:        {config.risk_level}")
        print(f"    │")
        trace = config.source_traceability
        print(f"    │ [SourceTraceability]")
        print(f"    │   upstream_repo:   {trace.upstream_repo[:55]}")
        print(f"    │   license:         {trace.license}")
        print(f"    │   original_author: {trace.original_author}")
        print(f"    │   attribution:     {trace.attribution[:50]}...")
        print(f"    └──────────────────────────────────────────────────")

    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Validate the configs themselves
    # ═══════════════════════════════════════════════════════════════════
    section("Step 5: Validate UnifiedSkillConfig Models")

    valid_count = 0
    invalid_count = 0
    for config in configs:
        errors = config.validate()
        if not errors:
            valid_count += 1
        else:
            invalid_count += 1

    print(f"  Valid configs:         {valid_count}")
    print(f"  Invalid configs:       {invalid_count}")
    print(f"  Validation pass rate:  {valid_count / max(len(configs), 1):.1%}")

    # Show any validation errors
    if invalid_count > 0:
        print("\n  Validation errors:")
        for config in configs:
            errors = config.validate()
            if errors:
                print(f"    {config.name}: {errors}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 6: Quality summary
    # ═══════════════════════════════════════════════════════════════════
    section("Step 6: Quality Summary")

    # Risk distribution
    risk_dist: dict[str, int] = {}
    for c in configs:
        risk_dist[c.risk_level] = risk_dist.get(c.risk_level, 0) + 1

    print("  Risk level distribution:")
    for level in ["low", "medium", "high", "critical"]:
        count = risk_dist.get(level, 0)
        bar = "█" * count
        print(f"    {level:10s}: {count:2d} {bar}")

    # Confidence distribution
    conf_dist: dict[str, int] = {}
    for s in skills:
        conf_dist[s.confidence.value] = conf_dist.get(s.confidence.value, 0) + 1

    print("\n  Confidence distribution:")
    for level in ["high", "medium", "low"]:
        count = conf_dist.get(level, 0)
        bar = "█" * count
        print(f"    {level:10s}: {count:2d} {bar}")

    # Score distribution
    scores_with_value = [c.curator_score for c in configs if c.curator_score is not None]
    if scores_with_value:
        avg_score = sum(scores_with_value) / len(scores_with_value)
        min_score = min(scores_with_value)
        max_score = max(scores_with_value)
        print(f"\n  Curator score stats:")
        print(f"    Skills with scores: {len(scores_with_value)}/{len(configs)}")
        print(f"    Average score:      {avg_score:.3f}")
        print(f"    Min score:          {min_score:.3f}")
        print(f"    Max score:          {max_score:.3f}")

    # Source traceability check
    with_trace = sum(1 for c in configs if c.source_traceability.upstream_repo)
    with_author = sum(1 for c in configs if c.source_traceability.original_author)
    with_license = sum(1 for c in configs if c.source_traceability.license)
    print(f"\n  Source traceability:")
    print(f"    With upstream repo: {with_trace}/{len(configs)}")
    print(f"    With author:        {with_author}/{len(configs)}")
    print(f"    With license:       {with_license}/{len(configs)}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 7: Export as JSON (serialization check)
    # ═══════════════════════════════════════════════════════════════════
    section("Step 7: Serialization Check (asdict)")

    from dataclasses import asdict

    serialized = [asdict(c) for c in configs]
    json_str = json.dumps(serialized[0], indent=2, ensure_ascii=False, default=str)
    print(f"  Total serialized configs: {len(serialized)}")
    print(f"  JSON size of first config: {len(json_str)} bytes")
    print(f"\n  Sample JSON (first config):")
    for line in json_str.split("\n")[:25]:
        print(f"    {line}")
    print("    ...")

    # ═══════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════
    section("Demo Complete — Summary")

    print(f"""
  Pipeline result:
    📄 Source:     Python requests library Quickstart documentation
    🔍 Extracted:  {len(skills)} skills from {adapter.extraction_log[-1]['raw_count']} raw matches
    ✅ Validated:  {total_passed}/{total_tasks} virtual tasks passed ({total_passed/max(total_tasks,1):.0%})
    📦 Configs:    {len(configs)} UnifiedSkillConfig models generated
    ✔️  Valid:      {valid_count}/{len(configs)} passed model validation
    📊 Risks:      {risk_dist.get('low',0)} low / {risk_dist.get('medium',0)} medium / {risk_dist.get('high',0)} high

  Key findings:
    • OpenSkillAdapter successfully extracts transferable skills from
      real-world API documentation using pattern matching
    • Skills span multiple categories: API endpoints, code examples,
      step-by-step procedures, CLI commands, and how-to guides
    • Virtual task validation provides quality signals (content, naming,
      source traceability)
    • UnifiedSkillConfig preserves full provenance: author, license,
      source URL, and extraction metadata
    • All configs are serializable to JSON for storage/transport
    • The pipeline is end-to-end: docs → extraction → validation →
      UnifiedSkillConfig → JSON
""")

    print("=" * 70)
    print("  OpenSkill Open-World Learning: PROVEN WORKING")
    print("=" * 70)


if __name__ == "__main__":
    demo()
