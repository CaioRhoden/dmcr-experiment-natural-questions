from langchain.prompts import PromptTemplate
from dmcr.datamodels.pipeline.DatamodelsPipelineData import DatamodelsPreCollectionsData


def nq_context_strategy(idx_row: int, idx_test: int, rag_indexes: dict, datamodels: DatamodelsPreCollectionsData) -> str:
    template = """
        Documents:
        {context}
        Question: 
        {input}
        Answer: 
    """

    context = ""
    count = 0
    for collection_idx in datamodels.train_collections_idx[idx_row]:
        
        idx = rag_indexes[str(idx_test)][collection_idx]
        title = datamodels.train_set[idx]["title"].to_numpy().flatten()[0]
        text = datamodels.train_set[idx]["text"].to_numpy().flatten()[0]
        context += f"Document[{count}](Title: {title}){text}\n\n"
        count += 1

    print(f"test_set: {datamodels.test_set}")
    input = datamodels.test_set[idx_test]["question"].to_numpy().flatten()[0]

    prompt = PromptTemplate.from_template(template).format(context=context, input=input)

    return prompt