from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import logging

class LLMInterface:
    """Interface for LLM inference."""
    def __init__(self, config: dict):
        try:
            self.llm = ChatOpenAI(
                model=config['model']['type'],
                api_key=config['model']['api_key'],
                base_url=config['model']['base_url'],
                temperature=config['model']['temperature']
            )
            logging.info(f"Initialized LLM: {config['model']['type']}")
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            raise

    def generate_code(self, prompt_template: str, analysis_text: str) -> str:
        """Generate code using LLM."""
        try:
            chain = PromptTemplate.from_template(prompt_template) | self.llm | StrOutputParser()
            output = chain.invoke({"analysis_text": analysis_text})
            logging.info("Generated code")
            return output
        except Exception as e:
            logging.error(f"Error generating code: {e}")
            raise


    def generate_pred(self, prompt: str) -> str:
        """Generate code using LLM."""
        try:
            chain = self.llm | StrOutputParser()
            output = chain.invoke(prompt)
            logging.info("Generated pred")
            return output
        except Exception as e:
            logging.error(f"Error generating code: {e}")
            raise


    def generate_raw_data(self, prompt_template: str, analysis_text: str) -> str:
        """Generate code using LLM."""
        try:
            chain = PromptTemplate.from_template(prompt_template) | self.llm | StrOutputParser()
            output = chain.invoke({"text": analysis_text})
            logging.info("Generated raw data")
            return output
        except Exception as e:
            logging.error(f"Error generating raw data: {e}")
            raise


    def generate_metrics(self, prompt_template: str, instruction: str, predict: str) -> str:
        """Generate code using LLM."""
        try:
            chain = PromptTemplate.from_template(prompt_template) | self.llm | JsonOutputParser()
            output = chain.invoke({"instruction": instruction, "predict": predict})
            logging.info(f"Generated metrics: {output}")
            return output
        except Exception as e:
            logging.error(f"Error generating metrics: {e}")
            raise

    def generate_code_metrics(self, prompt_template: str, code: str, gt_code: str) -> str:
        try:
            chain = PromptTemplate.from_template(prompt_template) | self.llm | JsonOutputParser()
            output = chain.invoke({"code": code, "gt_code": gt_code})
            logging.info(f"Generated code metrics: {output}")
            return output
        except Exception as e:
            logging.error(f"Error generating code metrics: {e}")
            raise
