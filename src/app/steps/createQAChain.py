from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from steps.prompt import prompt
prompt=prompt()

def create_qa_chain(retriever):
  primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
  created_qa_chain = (
    {"context": itemgetter("question") | retriever,
     "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(
        context=itemgetter("context")
      )
    | {
         "response": prompt | primary_qa_llm,
         "context": itemgetter("context"),
      }
  )

  return created_qa_chain