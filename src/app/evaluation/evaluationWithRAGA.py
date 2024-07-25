from datasets import Dataset,DatasetDict
from tqdm import tqdm
import pandas as pd


from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_utilization,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate

def create_ragas_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline.invoke({"question" : row["question"]})
    rag_dataset.append(
        {"question" : row["question"],
          "answer" : answer["response"].content,
          "contexts" : [context.page_content for context in answer["context"]],
          "ground_truth" : row["ground_truth"]
        }
    )
    
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_utilization,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result

def fix_context_format(dataset):
  
    if dataset.features["contexts"].dtype == dataset.features.Sequence:
        # Check if the elements within the sequence are strings
        if not all(isinstance(x, str) for x in dataset["contexts"][0]):
        # Preprocess if necessary (replace with your specific logic)
            def preprocess_context(context):
                # Example: Split context string using a delimiter
                return context.split(" || ")

        # Apply preprocessing to all contexts
        dataset["contexts"] = dataset.features.Sequence(
            dataset.Value("string"), compute=preprocess_context
        )
        dataset = dataset.map(preprocess_context, batched=True)
    return dataset


        