from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
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