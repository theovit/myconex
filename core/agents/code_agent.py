"""
MYCONEX Code Agent
Specialized agent for code generation, analysis, debugging, and software development tasks.
Inherits from BaseAgent with code-specific capabilities and tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from orchestration.agents.base_agent import AgentConfig, AgentContext, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class CodeAgent(BaseAgent):
    """
    Agent specialized in software development tasks.

    Capabilities:
    - Code generation and completion
    - Code review and analysis
    - Debugging and error fixing
    - Refactoring suggestions
    - Documentation generation
    - Testing assistance
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # Code-specific configuration
        self.supported_languages = [
            "python", "javascript", "typescript", "java", "cpp", "c", "go",
            "rust", "php", "ruby", "swift", "kotlin", "scala", "shell", "sql"
        ]

        self.code_prompts = {
            "generate": "Generate {language} code for: {description}. Include comments and error handling.",
            "complete": "Complete this {language} code snippet: {code}",
            "debug": "Debug this {language} code. Find and fix the error: {code}\nError: {error}",
            "review": "Review this {language} code for best practices, security, and performance: {code}",
            "refactor": "Refactor this {language} code to improve readability and maintainability: {code}",
            "test": "Generate unit tests for this {language} code: {code}",
            "document": "Generate comprehensive documentation for this {language} code: {code}",
        }

    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle the task type."""
        return task_type in [
            "code", "generate", "complete", "debug", "review",
            "refactor", "test", "document", "programming"
        ]

    async def handle_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """Handle code-related tasks."""

        try:
            # Extract task parameters
            language = payload.get("language", "python").lower()
            description = payload.get("description", "")
            code = payload.get("code", "")
            error_msg = payload.get("error", "")
            requirements = payload.get("requirements", [])

            # Validate language support
            if language not in self.supported_languages:
                return AgentResult(
                    task_id=task_id,
                    agent_name=self.name,
                    success=False,
                    error=f"Unsupported language: {language}. Supported: {', '.join(self.supported_languages)}",
                )

            # Build system prompt
            system_prompt = self._build_system_prompt(language, task_type, requirements)

            # Build user prompt
            user_prompt = self._build_user_prompt(task_type, language, description, code, error_msg)

            # Generate response
            response = await self._generate_code_response(system_prompt, user_prompt)

            # Parse and validate response
            result = self._parse_code_response(response, task_type)

            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=True,
                output=result,
                model_used=self.config.model,
                metadata={
                    "language": language,
                    "task_type": task_type,
                    "response_length": len(response),
                }
            )

        except Exception as e:
            logger.error(f"[code_agent] task failed: {e}")
            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    def _build_system_prompt(self, language: str, task_type: str, requirements: List[str]) -> str:
        """Build system prompt for code tasks."""
        base_prompt = f"""You are an expert {language} developer. Provide high-quality, production-ready code.

Guidelines:
- Write clean, readable, and well-documented code
- Follow {language} best practices and conventions
- Include proper error handling
- Add type hints where applicable
- Use meaningful variable and function names
- Include docstrings for functions and classes
"""

        if requirements:
            base_prompt += f"\nSpecific requirements:\n" + "\n".join(f"- {req}" for req in requirements)

        if task_type == "debug":
            base_prompt += "\nFor debugging: Identify the root cause, explain the fix, and provide corrected code."
        elif task_type == "review":
            base_prompt += "\nFor code review: Assess code quality, security, performance, and suggest improvements."
        elif task_type == "test":
            base_prompt += "\nFor testing: Write comprehensive unit tests covering edge cases and error conditions."

        return base_prompt

    def _build_user_prompt(
        self,
        task_type: str,
        language: str,
        description: str,
        code: str,
        error_msg: str
    ) -> str:
        """Build user prompt based on task type."""

        if task_type in ["generate", "code"]:
            return f"Generate {language} code for: {description}"

        elif task_type == "complete":
            return f"Complete this {language} code:\n\n{code}"

        elif task_type == "debug":
            return f"Debug this {language} code and fix the error:\n\nCode:\n{code}\n\nError:\n{error_msg}"

        elif task_type == "review":
            return f"Review this {language} code:\n\n{code}"

        elif task_type == "refactor":
            return f"Refactor this {language} code for better readability and maintainability:\n\n{code}"

        elif task_type == "test":
            return f"Generate unit tests for this {language} code:\n\n{code}"

        elif task_type == "document":
            return f"Generate documentation for this {language} code:\n\n{code}"

        else:
            return f"Process this {language} code task: {description}\n\nCode:\n{code}"

    async def _generate_code_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Ollama or LiteLLM."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self.config.use_litellm:
            return await self._call_litellm(messages)
        else:
            return await self._call_ollama(messages)

    async def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama API."""
        try:
            # Convert messages to Ollama format
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

            prompt = f"System: {system_msg}\n\nUser: {user_msg}" if system_msg else user_msg

            response = await self._http.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

        except Exception as e:
            raise Exception(f"Ollama API error: {e}")

    async def _call_litellm(self, messages: List[Dict[str, str]]) -> str:
        """Call LiteLLM API."""
        try:
            response = await self._http.post(
                f"{self.config.litellm_url}/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            raise Exception(f"LiteLLM API error: {e}")

    def _parse_code_response(self, response: str, task_type: str) -> Dict[str, Any]:
        """Parse and structure the code response."""

        result = {
            "response": response,
            "task_type": task_type,
        }

        # Extract code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            result["code_blocks"] = [
                {"language": lang or "text", "code": code.strip()}
                for lang, code in code_blocks
            ]

        # For debugging tasks, try to identify the fix
        if task_type == "debug":
            result["analysis"] = self._extract_debug_analysis(response)

        # For review tasks, extract suggestions
        elif task_type == "review":
            result["suggestions"] = self._extract_review_suggestions(response)

        return result

    def _extract_debug_analysis(self, response: str) -> Dict[str, str]:
        """Extract debugging analysis from response."""
        analysis = {}

        # Look for problem identification
        problem_match = re.search(r'(?:problem|issue|error).*?[:\-]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if problem_match:
            analysis["problem"] = problem_match.group(1).strip()

        # Look for solution
        solution_match = re.search(r'(?:solution|fix).*?[:\-]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if solution_match:
            analysis["solution"] = solution_match.group(1).strip()

        return analysis

    def _extract_review_suggestions(self, response: str) -> List[str]:
        """Extract review suggestions from response."""
        suggestions = []

        # Look for bullet points or numbered lists
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('- ', '* ', '• ')) or re.match(r'^\d+\.', line):
                suggestions.append(line[2:] if line.startswith(('- ', '* ', '• ')) else line)

        return suggestions

    # ─── Code Analysis Tools ──────────────────────────────────────────────────

    def analyze_code_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        # Basic complexity analysis
        lines = len(code.split('\n'))
        functions = len(re.findall(r'def\s+\w+', code)) if language == "python" else 0
        classes = len(re.findall(r'class\s+\w+', code)) if language == "python" else 0

        return {
            "lines": lines,
            "functions": functions,
            "classes": classes,
            "complexity_score": min(lines // 10 + functions + classes, 10),  # 0-10 scale
        }

    def detect_security_issues(self, code: str, language: str) -> List[str]:
        """Basic security issue detection."""
        issues = []

        # Python-specific checks
        if language == "python":
            if "eval(" in code:
                issues.append("Use of eval() - potential security risk")
            if "exec(" in code:
                issues.append("Use of exec() - potential security risk")
            if "subprocess.call(" in code and "shell=True" in code:
                issues.append("Shell injection risk in subprocess call")

        # Generic checks
        if "password" in code.lower() and ("hardcode" in code.lower() or "=" in code):
            issues.append("Potential hardcoded credentials")

        return issues