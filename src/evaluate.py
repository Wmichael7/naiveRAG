from datasets import Dataset
from rag_qa import qa                       
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy  

question = "What do new hires do in week one?"
out       = qa.invoke(question)       
answer    = out["result"]
contexts  = [doc.page_content for doc in out["source_documents"]]

hf_ds = Dataset.from_dict(
    {
        "question": [question],
        "answer":   [answer],
        "contexts": [contexts],             
    }
)

metrics = evaluate(
    hf_ds,
    metrics=[faithfulness, answer_relevancy]  
)
print("📊 RAGAS scores:", metrics)
