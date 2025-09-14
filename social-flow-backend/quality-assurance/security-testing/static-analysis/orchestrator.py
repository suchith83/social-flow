"""
Static Analysis Orchestrator
Coordinates file discovery, parsing, analyzer execution, rule engine, and reporting.
Designed to be extendable and to allow plugging in additional adapters.
"""

import multiprocessing as mp
from typing import List, Dict, Any
from .config import SCAN_CONFIG, DEFAULT_LANGUAGES
from .utils import iter_source_files, logger, normalize_severity, SEVERITY_ORDER
from .parsers import PythonParser, JavaScriptParser, JavaParser
from .analyzers import PylintAdapter, ESLintAdapter, CustomRuleEngine
from .report_generator import StaticReportGenerator

LANG_PATTERNS = {
    "python": ["*.py"],
    "javascript": ["*.js", "*.jsx", "*.ts", "*.tsx"],
    "java": ["*.java"]
}

PARSER_MAP = {
    "python": PythonParser,
    "javascript": JavaScriptParser,
    "java": JavaParser
}

ANALYZER_MAP = {
    "python": PylintAdapter,
    "javascript": ESLintAdapter,
    # Java adapter left as future work (call to SpotBugs/Spotless/PMD)
}

class StaticAnalysisOrchestrator:
    def __init__(self, languages: List[str] = None, config: dict = None):
        self.languages = languages or DEFAULT_LANGUAGES
        self.config = config or SCAN_CONFIG
        self.reporter = StaticReportGenerator()
        self.custom_engine = CustomRuleEngine()
        # instantiate adapters lazily
        self._adapters = {}

    def _get_adapter(self, lang: str):
        if lang in self._adapters:
            return self._adapters[lang]
        adapter_cls = ANALYZER_MAP.get(lang)
        if adapter_cls:
            adapter = adapter_cls()
            self._adapters[lang] = adapter
            return adapter
        return None

    def discover_files(self, root: str) -> Dict[str, List[str]]:
        """Return dict lang -> list of files to analyze"""
        files_by_lang = {}
        exclude = self.config.get("exclude_paths", [])
        max_size = self.config.get("max_file_size_kb", 512)
        for lang in self.languages:
            patterns = LANG_PATTERNS.get(lang, [])
            files = list(iter_source_files(root, patterns, exclude, max_file_size_kb=max_size))
            files_by_lang[lang] = files
            logger.info(f"Discovered {len(files)} files for {lang}")
        return files_by_lang

    def _run_parsers_and_rules(self, lang: str, file_paths: List[str]) -> List[Dict[str, Any]]:
        findings = []
        parser_cls = PARSER_MAP.get(lang)
        adapter = self._get_adapter(lang)
        # 1) run adapter (pylint/eslint) over chunked files if adapter present
        if adapter and file_paths:
            try:
                adapter_findings = adapter.run(file_paths)
                findings.extend(adapter_findings)
            except Exception:
                logger.exception("Adapter run failed; continuing")

        # 2) run parser + custom rules per file
        for p in file_paths:
            try:
                parsed = parser_cls.parse_file(p) if parser_cls else {"path": p}
                # builtin rule
                findings.extend(CustomRuleEngine.builtin_hardcoded_secret_rule(parsed))
                # dynamic rules
                try:
                    findings.extend(self.custom_engine.run(parsed))
                except Exception:
                    logger.exception("Custom rule engine failed on %s", p)
            except Exception:
                logger.exception("File analysis failed for %s", p)
        return findings

    def _filter_by_severity(self, findings: List[Dict[str, Any]], threshold: str) -> List[Dict[str, Any]]:
        thr = threshold.upper()
        thr_value = SEVERITY_ORDER.get(thr, 1)
        filtered = [f for f in findings if SEVERITY_ORDER.get(normalize_severity(f.get("severity")), 1) >= thr_value]
        logger.info(f"Filtered findings by severity {thr}: {len(filtered)} remain")
        return filtered

    def run(self, project_root: str, severity_threshold: str = None) -> Dict[str, Any]:
        """
        Main entry: discover files, analyze them, apply rules, and generate report.
        Returns dict with report paths and raw findings.
        """
        severity_threshold = severity_threshold or (REPORT_CONFIG.get("severity_threshold") if 'REPORT_CONFIG' in globals() else "LOW")
        logger.info(f"Starting SAST run on {project_root} for languages {self.languages}")
        files_by_lang = self.discover_files(project_root)

        # Parallel analysis per language (not per file) to avoid heavy process startup costs.
        manager = mp.Manager()
        results = manager.list()
        jobs = []

        def worker(lang, paths, out_list):
            try:
                res = self._run_parsers_and_rules(lang, paths)
                out_list.extend(res)
            except Exception:
                logger.exception("Worker failed for %s", lang)

        for lang, paths in files_by_lang.items():
            p = mp.Process(target=worker, args=(lang, paths, results))
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        findings = list(results)
        # normalize severities (ensure every finding has severity)
        for f in findings:
            f["severity"] = normalize_severity(f.get("severity", "LOW"))
            f.setdefault("file", f.get("file") or f.get("path") or "unknown")
            f.setdefault("line", f.get("line"))

        # filter by configured severity
        final_findings = self._filter_by_severity(findings, severity_threshold)

        # build metadata
        meta = {
            "project_root": project_root,
            "languages": self.languages,
            "generated_at": timestamp()
        }

        report_paths = self.reporter.generate(final_findings, meta)
        return {"reports": report_paths, "findings": final_findings}
