"""
MYCONEX Research Agent
Specialized agent for research, information gathering, data analysis, and knowledge synthesis.
Inherits from BaseAgent with research-specific capabilities and tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from orchestration.agents.base_agent import AgentConfig, AgentContext, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Agent specialized in research and information gathering tasks.

    Capabilities:
    - Web research and information synthesis
    - Data analysis and interpretation
    - Knowledge base querying
    - Report generation
    - Trend analysis
    - Comparative analysis
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # Research-specific configuration
        self.research_domains = [
            "technology", "science", "business", "health", "education",
            "environment", "politics", "economics", "culture", "sports"
        ]

        self.analysis_types = [
            "summarization", "comparison", "trend_analysis", "gap_analysis",
            "impact_assessment", "feasibility_study", "literature_review"
        ]

        # Research prompt templates
        self.research_prompts = {
            "web_search": "Research and summarize information about: {query}. Focus on recent developments and key insights.",
            "data_analysis": "Analyze this data and provide insights: {data}\n\nAnalysis type: {analysis_type}",
            "comparison": "Compare these options: {options}. Provide pros/cons and recommendations.",
            "trend_analysis": "Analyze trends in: {topic}. Include historical context and future projections.",
            "literature_review": "Review existing literature on: {topic}. Summarize key findings and gaps.",
            "feasibility_study": "Assess the feasibility of: {proposal}. Include technical, economic, and operational analysis.",
        }

    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle the task type."""
        return task_type in [
            "research", "analyze", "search", "compare", "trend",
            "review", "study", "investigate", "explore"
        ]

    async def handle_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """Handle research-related tasks."""

        try:
            # Extract task parameters
            query = payload.get("query", "")
            topic = payload.get("topic", "")
            data = payload.get("data", "")
            analysis_type = payload.get("analysis_type", "summarization")
            sources = payload.get("sources", [])
            options = payload.get("options", [])
            criteria = payload.get("criteria", [])

            # Validate analysis type
            if analysis_type not in self.analysis_types:
                return AgentResult(
                    task_id=task_id,
                    agent_name=self.name,
                    success=False,
                    error=f"Unsupported analysis type: {analysis_type}. Supported: {', '.join(self.analysis_types)}",
                )

            # Build system prompt
            system_prompt = self._build_research_system_prompt(task_type, analysis_type, criteria)

            # Build user prompt
            user_prompt = self._build_research_user_prompt(
                task_type, query, topic, data, analysis_type, sources, options
            )

            # Generate response
            response = await self._generate_research_response(system_prompt, user_prompt)

            # Parse and structure response
            result = self._parse_research_response(response, task_type, analysis_type)

            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=True,
                output=result,
                model_used=self.config.model,
                metadata={
                    "task_type": task_type,
                    "analysis_type": analysis_type,
                    "query": query or topic,
                    "response_length": len(response),
                }
            )

        except Exception as e:
            logger.error(f"[research_agent] task failed: {e}")
            return AgentResult(
                task_id=task_id,
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    def _build_research_system_prompt(self, task_type: str, analysis_type: str, criteria: List[str]) -> str:
        """Build system prompt for research tasks."""
        base_prompt = """You are an expert research analyst. Provide thorough, well-structured, and evidence-based analysis.

Guidelines:
- Be objective and data-driven in your analysis
- Cite sources and evidence for claims
- Consider multiple perspectives and viewpoints
- Identify assumptions and limitations
- Provide actionable insights and recommendations
- Structure responses with clear sections and headings
- Use bullet points and numbered lists for clarity
"""

        if criteria:
            base_prompt += f"\nEvaluation criteria:\n" + "\n".join(f"- {criterion}" for criterion in criteria)

        if task_type == "web_search":
            base_prompt += "\nFor web research: Focus on recent, reliable sources and synthesize key insights."
        elif task_type == "data_analysis":
            base_prompt += f"\nFor {analysis_type}: Apply appropriate analytical methods and statistical reasoning."
        elif task_type == "comparison":
            base_prompt += "\nFor comparisons: Use structured frameworks and weighted criteria."
        elif task_type == "trend_analysis":
            base_prompt += "\nFor trend analysis: Include historical context, current state, and future projections."

        return base_prompt

    def _build_research_user_prompt(
        self,
        task_type: str,
        query: str,
        topic: str,
        data: str,
        analysis_type: str,
        sources: List[str],
        options: List[str]
    ) -> str:
        """Build user prompt based on task type."""

        if task_type in ["research", "web_search"]:
            prompt = f"Research and analyze: {query or topic}"
            if sources:
                prompt += f"\n\nConsider these sources:\n" + "\n".join(f"- {source}" for source in sources)
            return prompt

        elif task_type == "analyze":
            prompt = f"Analyze this data using {analysis_type}:\n\n{data}"
            if sources:
                prompt += f"\n\nAdditional context from sources:\n" + "\n".join(f"- {source}" for source in sources)
            return prompt

        elif task_type == "compare":
            prompt = f"Compare these options:\n" + "\n".join(f"- {option}" for option in options)
            if data:
                prompt += f"\n\nAdditional data for comparison:\n{data}"
            return prompt

        elif task_type == "trend":
            return f"Analyze trends in: {topic}\n\nInclude historical data and future projections."

        elif task_type == "review":
            return f"Conduct a literature review on: {topic}\n\nSummarize key findings and identify research gaps."

        elif task_type == "study":
            return f"Conduct a feasibility study for: {query or topic}\n\nAssess technical, economic, and operational aspects."

        else:
            return f"Research task: {query or topic}\n\nData: {data}"

    async def _generate_research_response(self, system_prompt: str, user_prompt: str) -> str:
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

    def _parse_research_response(self, response: str, task_type: str, analysis_type: str) -> Dict[str, Any]:
        """Parse and structure the research response."""

        result = {
            "response": response,
            "task_type": task_type,
            "analysis_type": analysis_type,
        }

        # Extract sections
        sections = self._extract_sections(response)
        result["sections"] = sections

        # Extract key insights
        insights = self._extract_key_insights(response)
        result["key_insights"] = insights

        # Extract recommendations
        recommendations = self._extract_recommendations(response)
        result["recommendations"] = recommendations

        # Extract sources/references
        sources = self._extract_sources(response)
        result["sources"] = sources

        # Task-specific parsing
        if task_type == "compare":
            result["comparison_matrix"] = self._extract_comparison_data(response)
        elif task_type == "trend":
            result["trend_data"] = self._extract_trend_data(response)
        elif analysis_type == "feasibility_study":
            result["feasibility_assessment"] = self._extract_feasibility_data(response)

        return result

    def _extract_sections(self, response: str) -> Dict[str, str]:
        """Extract structured sections from response."""
        sections = {}

        # Look for common section headers
        section_patterns = [
            r'(?i)(?:^|\n)##?\s*(.+?)\s*\n(.*?)(?=\n##?\s*|\n*$)',
            r'(?i)(?:^|\n)#+\s*(.+?)\s*\n(.*?)(?=\n#+\s*|\n*$)',
        ]

        for pattern in section_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            for title, content in matches:
                sections[title.strip()] = content.strip()

        return sections

    def _extract_key_insights(self, response: str) -> List[str]:
        """Extract key insights from response."""
        insights = []

        # Look for insight indicators
        insight_patterns = [
            r'(?i)key insight[s]?:?\s*(.+?)(?:\n|$)',
            r'(?i)important finding[s]?:?\s*(.+?)(?:\n|$)',
            r'(?i)main conclusion[s]?:?\s*(.+?)(?:\n|$)',
        ]

        for pattern in insight_patterns:
            matches = re.findall(pattern, response)
            insights.extend(matches)

        # Also look for bullet points under insights sections
        if "insights" in response.lower():
            lines = response.split('\n')
            in_insights = False
            for line in lines:
                if "insight" in line.lower() and ("##" in line or "###" in line):
                    in_insights = True
                    continue
                elif in_insights and line.strip().startswith(('- ', '* ', '• ')):
                    insights.append(line.strip()[2:])
                elif in_insights and ("##" in line or "###" in line):
                    break

        return insights

    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from response."""
        recommendations = []

        # Look for recommendation indicators
        rec_patterns = [
            r'(?i)recommendation[s]?:?\s*(.+?)(?:\n|$)',
            r'(?i)suggestion[s]?:?\s*(.+?)(?:\n|$)',
            r'(?i)action item[s]?:?\s*(.+?)(?:\n|$)',
        ]

        for pattern in rec_patterns:
            matches = re.findall(pattern, response)
            recommendations.extend(matches)

        # Look for numbered or bulleted recommendations
        lines = response.split('\n')
        in_recommendations = False
        for line in lines:
            if "recommend" in line.lower() and ("##" in line or "###" in line):
                in_recommendations = True
                continue
            elif in_recommendations and (line.strip().startswith(('- ', '* ', '• ')) or re.match(r'^\d+\.', line.strip())):
                recommendations.append(line.strip().lstrip('1234567890. -*•'))
            elif in_recommendations and ("##" in line or "###" in line):
                break

        return recommendations

    def _extract_sources(self, response: str) -> List[str]:
        """Extract sources and references from response."""
        sources = []

        # Look for URLs
        urls = re.findall(r'https?://[^\s<>"{}|\\^`[\]]+', response)
        sources.extend(urls)

        # Look for citations
        citations = re.findall(r'\[([^\]]+)\]', response)
        sources.extend(citations)

        # Look for reference sections
        if "references" in response.lower() or "sources" in response.lower():
            lines = response.split('\n')
            in_refs = False
            for line in lines:
                if ("reference" in line.lower() or "source" in line.lower()) and ("##" in line or "###" in line):
                    in_refs = True
                    continue
                elif in_refs and line.strip():
                    if line.strip().startswith(('- ', '* ', '• ')) or re.match(r'^\d+\.', line.strip()):
                        sources.append(line.strip().lstrip('1234567890. -*•'))
                    elif not line.startswith('##'):
                        sources.append(line.strip())
                elif in_refs and ("##" in line or "###" in line):
                    break

        return list(set(sources))  # Remove duplicates

    def _extract_comparison_data(self, response: str) -> Dict[str, Any]:
        """Extract comparison matrix data."""
        comparison = {
            "options": [],
            "criteria": [],
            "scores": {},
        }

        # Try to identify options and criteria from the response
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for section headers
            if "option" in line.lower() and ("##" in line or "###" in line):
                current_section = "options"
                continue
            elif "criteri" in line.lower() and ("##" in line or "###" in line):
                current_section = "criteria"
                continue

            # Extract items
            if current_section == "options" and (line.startswith(('- ', '* ', '• ')) or re.match(r'^\d+\.', line)):
                comparison["options"].append(line.lstrip('1234567890. -*•'))
            elif current_section == "criteria" and (line.startswith(('- ', '* ', '• ')) or re.match(r'^\d+\.', line)):
                comparison["criteria"].append(line.lstrip('1234567890. -*•'))

        return comparison

    def _extract_trend_data(self, response: str) -> Dict[str, Any]:
        """Extract trend analysis data."""
        trend_data = {
            "historical": [],
            "current": [],
            "future": [],
        }

        # Look for trend sections
        sections = self._extract_sections(response)

        for section_name, content in sections.items():
            section_lower = section_name.lower()
            if "histor" in section_lower:
                trend_data["historical"] = [line.strip('- *•') for line in content.split('\n') if line.strip().startswith(('- ', '* ', '• '))]
            elif "current" in section_lower or "present" in section_lower:
                trend_data["current"] = [line.strip('- *•') for line in content.split('\n') if line.strip().startswith(('- ', '* ', '• '))]
            elif "future" in section_lower or "projection" in section_lower:
                trend_data["future"] = [line.strip('- *•') for line in content.split('\n') if line.strip().startswith(('- ', '* ', '• '))]

        return trend_data

    def _extract_feasibility_data(self, response: str) -> Dict[str, Any]:
        """Extract feasibility study data."""
        feasibility = {
            "technical_feasibility": "",
            "economic_feasibility": "",
            "operational_feasibility": "",
            "overall_assessment": "",
        }

        sections = self._extract_sections(response)

        for section_name, content in sections.items():
            section_lower = section_name.lower()
            if "technical" in section_lower:
                feasibility["technical_feasibility"] = content
            elif "economic" in section_lower or "cost" in section_lower:
                feasibility["economic_feasibility"] = content
            elif "operational" in section_lower or "practical" in section_lower:
                feasibility["operational_feasibility"] = content
            elif "overall" in section_lower or "conclusion" in section_lower:
                feasibility["overall_assessment"] = content

        return feasibility

    # ─── Research Tools ──────────────────────────────────────────────────────

    def validate_source(self, url: str) -> Dict[str, Any]:
        """Basic source validation."""
        parsed = urlparse(url)
        return {
            "is_valid": bool(parsed.scheme and parsed.netloc),
            "domain": parsed.netloc,
            "scheme": parsed.scheme,
            "path": parsed.path,
            "reliability_score": self._assess_source_reliability(parsed.netloc),
        }

    def _assess_source_reliability(self, domain: str) -> float:
        """Simple domain reliability assessment (0-1 scale)."""
        # This is a basic implementation - in practice, you'd use more sophisticated methods
        reliable_domains = [
            "edu", "gov", "org", "ac.uk", "ac.au", "edu.au",
            "nature.com", "science.org", "ieee.org", "acm.org"
        ]

        if any(reliable in domain for reliable in reliable_domains):
            return 0.8

        suspicious_domains = ["blogspot", "wordpress", "medium"]
        if any(suspicious in domain for suspicious in suspicious_domains):
            return 0.3

        return 0.5  # Neutral

    def analyze_data_quality(self, data: str) -> Dict[str, Any]:
        """Analyze quality of research data."""
        quality_metrics = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0,
        }

        # Basic heuristics
        if data.strip():
            quality_metrics["completeness"] = 0.7  # Has content

        # Check for data patterns
        if re.search(r'\d{4}-\d{2}-\d{2}', data):  # Date patterns
            quality_metrics["timeliness"] = 0.8

        # Check for citations
        if re.search(r'\[.*?\]|\(.*?\d{4}.*?\)', data):
            quality_metrics["accuracy"] = 0.7

        return quality_metrics