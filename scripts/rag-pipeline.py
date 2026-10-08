import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CHROMA_PERSIST_DIRECTORY = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma-db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pharma_dita_topics")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
PROMPT_LIBRARY_DIR = PROJECT_ROOT / os.getenv("PROMPT_LIBRARY_DIR", "prompt-library")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TASK_PROMPT_MAP: Dict[str, str] = {
    "adverse_reactions": "task-prompts/adverse_reactions-prompt.json",
    "dosage_administration": "task-prompts/dosage-prompt.json",
    "safety_information": "task-prompts/safety-information-prompt.json",
    "clinical_overview": "task-prompts/clinical-overview-prompt.json",
    "fallback_unrelated": "task-prompts/fallback-prompt.json",
}


class PharmaRAGPipeline:
    def __init__(self) -> None:
        print("[Pipeline Init] Loading HuggingFace embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        print(f"[Pipeline Init] Connecting to ChromaDB vector store at: {CHROMA_PERSIST_DIRECTORY}")
        self.vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_PERSIST_DIRECTORY),
        )
        
        system_prompts_file = PROMPT_LIBRARY_DIR / "system-prompts.json"
        print(f"[Pipeline Init] Loading system prompts from: {system_prompts_file}")
        self.system_prompts: Dict[str, Any] = self._load_json(system_prompts_file).get("system_prompts", {})
        print("[Pipeline Init] Initialization complete.\n")

    @staticmethod
    def _load_json(file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_product_from_query(self, query: str) -> Optional[str]:
        """Dynamically detects product references in user text (e.g., 'Product A', 'Product B')."""
        match = re.search(r"product\s+([a-z0-9_]+)", query, re.IGNORECASE)
        if match:
            return f"product_{match.group(1).lower()}"
        return None

    def _resolve_task_type(self, query: str, explicit_intent: Optional[str] = None) -> str:
        if explicit_intent and explicit_intent in TASK_PROMPT_MAP:
            return explicit_intent

        q = query.lower()
        if any(k in q for k in ["adverse", "side effect", "reaction", "hypotension", "bradycardia"]):
            return "adverse_reactions"
        elif any(k in q for k in ["dosage", "dose", "administer", "mg/kg", "flush", "iv"]):
            return "dosage_administration"
        elif any(k in q for k in ["contraindication", "precaution", "warning", "safety"]):
            return "safety_information"
        elif any(k in q for k in ["overview", "indication", "mechanism", "action", "description"]):
            return "clinical_overview"
        else:
            return "fallback_unrelated"

    def _get_prompt_config(self, task_type: str) -> Dict[str, Any]:
        rel_path = TASK_PROMPT_MAP.get(task_type, TASK_PROMPT_MAP["fallback_unrelated"])
        task_prompt_path = PROMPT_LIBRARY_DIR / rel_path
        task_prompt_data = self._load_json(task_prompt_path)

        sys_ref = task_prompt_data.get("system_prompt_ref", "pharma_rag_default_v1")
        sys_config = self.system_prompts.get(sys_ref, {})

        role = sys_config.get("role", "")
        guardrails = "\n".join([f"- {g}" for g in sys_config.get("guardrails", [])])
        template = task_prompt_data.get("template", "")

        full_system_block = f"{role}\n\nStrict Regulatory Guardrails:\n{guardrails}"
        combined_template = f"{full_system_block}\n\n---\n\n{template}"

        return {
            "template": combined_template,
            "model_config": task_prompt_data.get("model_config", {"model": OLLAMA_MODEL_NAME, "temperature": 0.0}),
            "variables": task_prompt_data.get("variables", ["query", "product", "target_audience", "context_chunks"]),
        }

    def _format_docs(self, docs: List[Any]) -> str:
        if not docs:
            return "No relevant DITA context retrieved."

        formatted = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Unknown Doc")
            sec = doc.metadata.get("section_title", "General")
            formatted.append(f"[Chunk {i}] Document: {title} | Section: {sec}\nContent:\n{doc.page_content}\n")
        return "\n".join(formatted)

    def query(
        self,
        query_text: str,
        product: Optional[str] = None,
        category: Optional[str] = None,
        target_audience: str = "HCP",
        intent: Optional[str] = None,
        k: int = 5,
    ) -> Dict[str, Any]:
        print(f"[Query] User Question: '{query_text}'")
        
        # 1. Resolve Intent & Prompt Config
        task_type = self._resolve_task_type(query_text, explicit_intent=intent)
        prompt_config = self._get_prompt_config(task_type)
        print(f"[Query] Detected Intent/Task: {task_type}")

        # 2. Dynamic Product Detection
        detected_prod_key = self._extract_product_from_query(query_text)
        if not product and detected_prod_key:
            product_display = detected_prod_key.replace("_", " ").title()
            prod_key = detected_prod_key
        elif product:
            product_display = product
            prod_key = product.lower().replace(" ", "_")
        else:
            product_display = "Target Product"
            prod_key = None

        print(f"[Query] Target Product Identified: {product_display} (Filter key: {prod_key})")

        # 3. Build Metadata Filter
        filter_conditions: List[Dict[str, Any]] = []
        if category:
            filter_conditions.append({"category": {"$eq": category.lower()}})
        if prod_key:
            filter_conditions.append({"product": {"$eq": prod_key}})

        if len(filter_conditions) == 1:
            metadata_filter: Optional[Dict[str, Any]] = filter_conditions[0]
        elif len(filter_conditions) > 1:
            metadata_filter = {"$and": filter_conditions}
        else:
            metadata_filter = None

        # 4. Vector Retrieval
        retriever_args: Dict[str, Any] = {"k": k}
        if metadata_filter:
            retriever_args["filter"] = metadata_filter

        retrieved_docs = self.vectorstore.similarity_search(query_text, **retriever_args)

        seen_contents = set()
        unique_docs = []
        for doc in retrieved_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

        retrieved_docs = unique_docs
        context_str = self._format_docs(retrieved_docs)
        
        print("\n--- RETRIEVED DITA CHUNKS FOR CONTEXT ---")
        for doc in retrieved_docs:
            print(f"-> Doc: {doc.metadata.get('title')} | Section: {doc.metadata.get('section_title')}")
        print("------------------------------------------\n")

        # 5. LLM Chain Execution
        llm_kwargs = prompt_config.get("model_config", {})
        llm = ChatOllama(
            model=llm_kwargs.get("model", OLLAMA_MODEL_NAME),
            temperature=llm_kwargs.get("temperature", 0.0),
            top_p=llm_kwargs.get("top_p", 0.9),
            repeat_penalty=1.1,
            base_url=OLLAMA_BASE_URL,
        )

        prompt_template = ChatPromptTemplate.from_template(prompt_config["template"])
        chain = prompt_template | llm | StrOutputParser()

        # Build execution payload
        payload = {
            "query": query_text,
            "product": product_display,
            "target_audience": target_audience,
            "context_chunks": context_str,
        }

        print("[Query] Executing LLM generation via Ollama...")
        response_text = chain.invoke(payload)

        return {
            "query": query_text,
            "intent_detected": task_type,
            "product": product_display,
            "response": response_text,
            "retrieved_sources": [
                {
                    "title": doc.metadata.get("title"),
                    "section": doc.metadata.get("section_title"),
                    "source_file": doc.metadata.get("source_file"),
                }
                for doc in retrieved_docs
            ],
        }


if __name__ == "__main__":
    pipeline = PharmaRAGPipeline()

    print("\n" + "=" * 60)
    print("RUNNING TEST QUERY 1: Product Overview for Product A")
    print("=" * 60)
    res3 = pipeline.query("What's the clinical overview for Product A?")
    print("\n[RESPONSE 1]:\n")
    print(res3["response"])