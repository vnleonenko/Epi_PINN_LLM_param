# agents/IntentParserAgent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from formats.data_formats import PipelineState, ExpertIntent
import json
import re

class IntentParserAgent:
    """
    Parses expert comment ONCE using LLM only.
    No keyword fallbacks — pure LLM reasoning.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ExpertIntent)
        self.prompt = self._create_prompt()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an epidemiologist. Analyze the expert comment and determine how the infected (I) peak should change.

    **Think causally about interventions:**
    - If a measure reduces transmission (mask, lockdown, distancing) → peak LATER and LOWER
    - If a measure increases transmission (reopen, lift) → peak EARLIER and HIGHER
    - If a measure only affects recovery/treatment → height may change, position UNCHANGED

    **CRITICAL: Restrictive measures ALWAYS change the peak.**
    - "mask", "lockdown", "restrictions", "distancing" → cares_about_position: true, position_direction: "later", cares_about_height: true, height_direction: "lower"

    **Output JSON with these EXACT fields:**
    - cares_about_position: boolean
    - position_direction: "later", "earlier", or "any"
    - cares_about_height: boolean
    - height_direction: "higher", "lower", or "any"
    - primary_metric: "position", "height", or "both"
    - reasoning: brief causal explanation

    **Example for "Mask mandate will be introduced":**
    {{
    "cares_about_position": true,
    "position_direction": "later",
    "cares_about_height": true,
    "height_direction": "lower",
    "primary_metric": "height",
    "reasoning": "Mask mandate reduces transmission, delaying and lowering the peak."
    }}

    {format_instructions}

    Return ONLY valid JSON. No other text."""),
            ("human", "Expert comment: {expert_comment}")
        ]).partial(format_instructions=self.parser.get_format_instructions())
    
    def parse(self, comment: str) -> ExpertIntent:
        """Parse expert comment using LLM only."""
        if not comment:
            return ExpertIntent(
                cares_about_position=False,
                position_direction="any",
                cares_about_height=False,
                height_direction="any",
                primary_metric="both",
                reasoning="No comment provided"
            )
        
        chain = self.prompt | self.llm | self.parser
        result = chain.invoke({"expert_comment": comment})
        return result
    
    def __call__(self, state: PipelineState) -> PipelineState:
        """Parse intent and save to state."""
        print("=" * 60)
        print("🎯 INTENT PARSER AGENT (LLM only)")
        print("=" * 60)
        
        comment = state.get('expert_comment', '')
        print(f"💬 Expert: {comment}")
        
        intent = self.parse(comment)
        
        state['expected_position'] = intent.position_direction if intent.cares_about_position else "unchanged"
        state['expected_height'] = intent.height_direction if intent.cares_about_height else "unchanged"
        state['expert_intent'] = intent
        
        print(f"   cares_about_position: {intent.cares_about_position} → {intent.position_direction}")
        print(f"   cares_about_height:   {intent.cares_about_height} → {intent.height_direction}")
        print(f"   primary_metric: {intent.primary_metric}")
        print(f"   reasoning: {intent.reasoning}")
            
        
        return state