import streamlit as st  # Import python packages
from snowflake.snowpark.context import get_active_session

from snowflake.core import Root

import pandas as pd
import json

pd.set_option("max_colwidth", None)

NUM_CHUNKS = st.sidebar.slider("Number of retrieved documents", min_value=1, max_value=10, value=3, step=1)

# service parameters
CORTEX_SEARCH_DATABASE = "CHEMDB"
CORTEX_SEARCH_SCHEMA = "PUBLIC"
CORTEX_SEARCH_SERVICE = "CC_SEARCH_SERVICE_CS"
######
######

# columns to query in the service
COLUMNS = [
    "ABSTRACT",
    "AUTHORS",
    "PUBLICATION_YEAR"
]

session = get_active_session()
root = Root(session)

svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]


### Functions

def config_options():
    st.sidebar.expander("Session State").write(st.session_state)


def get_similar_chunks_search_service(query):
    response = svc.search(query, COLUMNS, limit=NUM_CHUNKS)

    st.sidebar.json(response.json())

    return response.json()


def create_prompt(myquestion):
    if st.session_state.rag == 1:
        prompt_context = get_similar_chunks_search_service(myquestion)

        if json.loads(prompt_context)['results'] is not None:

            prompt = f"""
               You are a scientific chemical chat agent that extracts information from research papers provided
               between <context> and </context> tags to respond to questions.
               When ansering the question contained between <question> and </question> tags
               be concise and do not hallucinate.

               If any provided papers are irrelevant, ignore them and do not mention them.

               Summarize any relevant information from the papers and reason over it before giving your final answer using the formatting \'Answer: [answer]\'. 
               Your response should be in the tone of scientific writing, and cite papers in APA style as is appropriate. 
               If any conclusions are uncertain, you should answer \'maybe\'.

               Your final answer should be \'yes\', \'no\', or \'maybe\'. 

               <context>          
               {prompt_context}
               </context>
               <question>  
               {myquestion}
               </question>
               Answer: 
               """

            json_data = json.loads(prompt_context)

        else:
            prompt = f"""
               You are a scientific chemical chat agent.
               When ansering the question contained between <question> and </question> tags
               be concise and do not hallucinate.

               Summarize relevant information from the paper and reason over it before giving your final answer using the formatting \'Answer: [answer]\'. Your response should be in the tone of scientific writing, and cite the papers in the context in APA style as needed. If the conclusions of the paper are uncertain, you should answer \'maybe\'.

               Your final answer should be \'yes\', \'no\', or \'maybe\'. 


               <question>  
               {myquestion}
               </question>
               Answer: 
               """

            json_data = json.loads(prompt_context)

        relative_paths = "None"  # set(item['relative_path'] for item in json_data['results'])

    else:
        prompt = f"""[0]
         'Question:  
           {myquestion} 
           Answer: '
           """
        relative_paths = "None"

    return prompt, relative_paths


def complete(myquestion):
    prompt, relative_paths = create_prompt(myquestion)
    cmd = """
            select snowflake.cortex.complete(?, ?) as response
          """

    df_response = session.sql(cmd, params=['mistral-large2', prompt]).collect()
    return df_response, relative_paths


def main():
    st.title(f"Chemical Research Question-Answering Assistant with Snowflake Cortex")
    # st.write("This is the list of documents you already have and that will be used to answer your questions:")
    # docs_available = session.sql("ls @docs").collect()
    # list_docs = []
    # for doc in docs_available:
    #    list_docs.append(doc["name"])
    # st.dataframe(list_docs)

    config_options()

    st.session_state.rag = st.sidebar.checkbox('Use RAG?', value=True)

    question = st.text_input("Enter question", placeholder="Can stellar wobble in triple systems mimic a planet?",
                             label_visibility="collapsed")

    if question:
        response, relative_paths = complete(question)
        res_text = response[0].RESPONSE
        st.markdown(res_text)

        if relative_paths != "None":
            with st.sidebar.expander("Related Documents"):
                for path in relative_paths:
                    cmd2 = f"select GET_PRESIGNED_URL(@docs, '{path}', 360) as URL_LINK from directory(@docs)"
                    df_url_link = session.sql(cmd2).to_pandas()
                    url_link = df_url_link._get_value(0, 'URL_LINK')

                    display_url = f"Doc: [{path}]({url_link})"
                    st.sidebar.markdown(display_url)

    if st.button("Evaluate on ScholarChemQA (~35 minutes)"):

        eval_query = "SELECT QUESTION, LABEL FROM CHEMDB.PUBLIC.SCHOLARCHEMQA"
        eval_data = session.sql(eval_query).to_pandas()

        y_true = []
        y_pred = []

        progress_bar = st.progress(0)
        total_questions = len(eval_data)
        with st.spinner('Evaluating on ScholarChemQA, this could take some time...'):
            for idx, row in enumerate(eval_data.iterrows()):
                _, data = row
                question = data['QUESTION']
                expected_response = data['LABEL']

                response, _ = complete(question)
                model_response = response[0].RESPONSE

                if "yes" in model_response.lower():
                    prediction = 'yes'
                elif "no" in model_response.lower():
                    prediction = 'no'
                else:
                    prediction = 'maybe'

                y_true.append(expected_response)
                y_pred.append(prediction)

                progress_bar.progress((idx + 1) / total_questions)

        correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct_predictions / len(y_true)

        labels = set(y_true)
        f1_scores = []
        for label in labels:
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append((f1, y_true.count(label)))

        f1 = sum(f1 * count for f1, count in f1_scores) / len(y_true)

        st.write("Evaluation Metrics:")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")


if __name__ == "__main__":
    main()