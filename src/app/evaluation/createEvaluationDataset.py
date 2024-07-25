

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm
import pandas as pd
from datasets import Dataset

evaluation_dataset_path = "/Users/geetikasaraf/Library/Mobile Documents/com~apple~CloudDocs/OGM/OGM.Thesis/OGM.Thesis.Geetika/OGM.insy/src/app/data/evaluation_dataset2.csv"

def CreateEvalDataset(chunks, question_generation_llm,answer_generation_llm, open_ai_key):
    question_schema = ResponseSchema(
        name="question",
        description="a question about the context."
    )

    question_response_schemas = [
        question_schema,
    ]

    question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
    format_instructions = question_output_parser.get_format_instructions()

    #question_generation_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    #question_generation_llm1=question_generation_llm

    bare_prompt_template = "{content}"
    bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template, open_api_key=open_ai_key)

    qa_template = """\
    You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

    question: a question about the context.

    Format the output as JSON with the following keys:
    question

    context: {context}
    """

    prompt_template = ChatPromptTemplate.from_template(template=qa_template)

    messages = prompt_template.format_messages(
        context=chunks[0],
        format_instructions=format_instructions
    )

    question_generation_chain = bare_template | question_generation_llm

    response = question_generation_chain.invoke({"content": messages})
    output_dict = question_output_parser.parse(response.content)

    for k, v in output_dict.items():
        print(k)
        print(v)

    qac_triples = []

    for text in tqdm(chunks[:10]):
        messages = prompt_template.format_messages(
            context=text,
            #question="This is a placeholder question",
            format_instructions=format_instructions
        )
        response = question_generation_chain.invoke({"content": messages})
        try:
            output_dict = question_output_parser.parse(response.content)
        except Exception as e:
            continue
        output_dict["context"] = text
        qac_triples.append(output_dict)

        ######### answer generation ##################################


    #qac_triples = [6]

    answer_schema = ResponseSchema(
        name="answer",
        description="an answer to the question"
    )

    answer_response_schemas = [
        answer_schema,
    ]

    answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
    format_instructions = answer_output_parser.get_format_instructions()

    qa_template = """\
        You are a University Professor creating a test for advanced students. For each question and context, create an answer.

        answer: a answer about the context.

        Format the output as JSON with the following keys:
        answer

        question: {question}
        context: {context}
        """

    prompt_template = ChatPromptTemplate.from_template(template=qa_template)

    messages = prompt_template.format_messages(
    context=qac_triples[0]["context"],
    question=qac_triples[0]["question"],
    format_instructions=format_instructions
    )
    
    answer_generation_chain = bare_template | answer_generation_llm
    response = answer_generation_chain.invoke({"content" : messages})
    output_dict = answer_output_parser.parse(response.content)

    
    for triple in tqdm(qac_triples):
        messages = prompt_template.format_messages(
            context=triple["context"],
            question=triple["question"],
            format_instructions=format_instructions
        )
        response = answer_generation_chain.invoke({"content" : messages})
        try:
            output_dict = answer_output_parser.parse(response.content)
        except Exception as e:
            continue
        triple["answer"] = output_dict["answer"]

    ground_truth_qac_set = pd.DataFrame(qac_triples)
    ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))

    ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer": "ground_truth"})

    eval_dataset = Dataset.from_pandas(ground_truth_qac_set)

    eval_dataset.to_csv(evaluation_dataset_path)

    return eval_dataset



