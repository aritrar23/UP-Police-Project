import streamlit as st
import nest_asyncio
import os
import pickle
import asyncio
from llama_index.llms.groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llama_index.core import QueryBundle
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core import SummaryIndex
from llama_index.core.async_utils import run_jobs
from llama_index.core.schema import IndexNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexAutoRetriever
from pinecone import Pinecone
from pinecone import ServerlessSpec
import shutil


# Initialize nest_asyncio to allow async operations in Streamlit
nest_asyncio.apply()

embed_model = HuggingFaceEmbeddings()
#
Settings.embed_model = embed_model

# Streamlit UI setup
st.set_page_config(page_title="UP Police Circulars Q&A Bot")
with st.sidebar:
    st.title('UP Police Circulars Q&A Bot')

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me anything about UP Police circulars."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initialize variables only once
if 'retriever' not in st.session_state:
    # Set the API key as an environment variable for GROQ LLM
    # os.environ["GROQ_API_KEY"] = "API-KEY"
    # os.environ["PINECONE_API_KEY"] = "API-KEY"
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)

    # Load the pre-generated summaries from the pickle file
    with open('doc_summaries.pkl', 'rb') as file:
        doc_summaries = pickle.load(file)

    # Create and configure the general Groq LLM (Groq)
    llm = Groq(
        model="llama3-70b-8192",  # or whatever model you wish to use for general processing
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create and configure the conversational ChatGroq LLM (ChatGroq)
    llm2 = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name='mixtral-8x7b-32768'  # Chat-specific LLM
    )

    documents = SimpleDirectoryReader(input_dir='corrected-english-circulars').load_data()
    docs = []
    for idx, doc in enumerate(documents):
        doc.metadata["index_id"] = str(doc.id_)
        docs.append(doc)

    pinecone_index = pc.Index("english-circulars")
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace="test",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    doc_index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context
    )

    async def aprocess_doc(doc, summary_txt, include_summary: bool = True):
        """Process doc using pre-generated summaries."""
        metadata = doc.metadata
        
        # No need for ID checks; use the summary directly
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="index_id", operator=FilterOperator.EQ, value=str(doc.metadata["index_id"])
                ),
            ]
        )

        # Create an index node using the summary text and metadata
        index_node = IndexNode(
            text=summary_txt,
            metadata=metadata,
            obj=doc_index.as_retriever(filters=filters),
            index_id=doc.id_,
        )

        return index_node

    async def process_docs(docs, doc_summaries):
        """Process metadata on docs using pre-generated summaries."""
        tasks = []
        for i, doc in enumerate(docs):
            summary_txt = doc_summaries[i]
            task = aprocess_doc(doc, summary_txt)
            tasks.append(task)

        index_nodes = await asyncio.gather(*tasks)
        return index_nodes

    def run_async_function():
        index_nodes = asyncio.run(process_docs(docs, doc_summaries))
        return index_nodes

    index_nodes = run_async_function()

    pinecone_index_metadata = pc.Index("english-circulars-metadata")
    vector_store_auto = PineconeVectorStore(
        pinecone_index=pinecone_index_metadata,
        namespace="test2",
    )
    storage_context_auto = StorageContext.from_defaults(vector_store=vector_store_auto)
    index = VectorStoreIndex(
        objects=index_nodes, storage_context=storage_context_auto
    )

    vector_store_info = VectorStoreInfo(
        content_info="UP Police Circulars",
        metadata_info=[
            MetadataInfo(
                name="file_path",
                description="The path of the file",
                type="string",
            ),
            MetadataInfo(
                name="file_name",
                description="The name of the file",
                type="string",
            ),
            MetadataInfo(
                name="file_type",
                description="The text type of the file, like text/plain",
                type="string",
            ),
            MetadataInfo(
                name="file_size",
                description="The size of the file in kb",
                type="integer",
            ),
            MetadataInfo(
                name="creation_date",
                description="The date the file was created",
                type="string",
            ),
            MetadataInfo(
                name="last_modified_date",
                description="When the file was last modified",
                type="string",
            ),
        ],
    )
    
    Settings.llm = llm
    st.session_state.retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
        similarity_top_k=5,
        empty_query_top_k=10,
        verbose=True,
    )

# Code to run every time a new user prompt is provided
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer soon..."):
                user_query = st.session_state.messages[-1]["content"]
                
                # Retrieve documents based on user query
                dictionary = {'20240513163858316722 circular.pdf': 'https://drive.google.com/file/d/1h_96XyFGcos358x3jb1JXtbMYGpkn-rn/view', '20240513163744379021 circular.pdf': 'https://drive.google.com/file/d/1xMCFoH40hZJ3q5xE_LJC9uTmCbMqMLIb/view', '20230904181806364533 circular.pdf': 'https://drive.google.com/file/d/1SoNCpTa3_RYpv99H2A7LfJbtNNPTGyp3/view', '20230403123339004612   2023 circular.pdf': 'https://drive.google.com/file/d/1apxK9KxhWkpMeuBphN1tap8g_65hl0z0/view', '20230216134112536906.pdf': 'https://drive.google.com/file/d/1N9NBTLkHrhPi4aFhfu-w-OqpoUjKwCNf/view', '20230527141309399015 cir.pdf': 'https://drive.google.com/file/d/13CjcMXGp22BxdqMYlyxJrn1A8-S2ZF4s/view', '20230608104030828818.pdf': 'https://drive.google.com/file/d/1OEkqOZaakFY8FnWqTu8qu172eEG9oh-n/view', '2023031412003784278.pdf': 'https://drive.google.com/file/d/1uDRb8y6vHEWQtPrVKkN-Ew0KkJ3Cmd4m/view', '20230608103756829217.pdf': 'https://drive.google.com/file/d/1Deo7i13E3F3_DZB3mxyXD5sLFKwxFtDF/view', '20230106111705174401.pdf': 'https://drive.google.com/file/d/1kqJCQaeOYqrxTc0D8iZepyHQt37ZSspI/view', '20230620162233579421.pdf': 'https://drive.google.com/file/d/1Zgjwm0ezgwStGMrJyejQqpnjELLlJyhC/view', '20240311105050207841.pdf': 'https://drive.google.com/file/d/1Wx3ezmMOTXTPi-oe3HWZfYsw4SzblZno/view', '20230314175534332610.pdf': 'https://drive.google.com/file/d/1gUKwuHjd4i9_kGzYVhu0BBnDCw14oIES/view', '20230217163427693307.pdf': 'https://drive.google.com/file/d/1PBw9J8Ik-tjfqZ_yYSHN0GnuAKh65gA1/view', '20230202173930064103.pdf': 'https://drive.google.com/file/d/1mIqg0zC8y_KZOxh_YAb7stu1BX_1cQ57/view', '20230202174013331804.pdf': 'https://drive.google.com/file/d/1uqC0rdNtZhRCEyqd6awHbsFrDnKa9ah0/view', '2023020917573151915 cir.pdf': 'https://drive.google.com/file/d/1teqX_aakMYbVT3qh7AezqhshyFB8W6zP/view', '20240326143731543213.pdf': 'https://drive.google.com/file/d/1f7g57TeGwwTMuLzNb2jEvA2JmVxKzAOu/view', '20240308145330234134.pdf': 'https://drive.google.com/file/d/1CYwegKxt2SJBy9dhnaKnK2yqrVfOcXGB/view', '20230421170641517514    2023.pdf': 'https://drive.google.com/file/d/1Su0m2pYHrLIkkfGDYCblNl8btyiapd-i/view', '20230811175511604727 CIRCULAR.pdf': 'https://drive.google.com/file/d/1VUKyJYHtaTWLuFu2VdJvOmn8iU4-Tj3i/view', '20240522113000767423 cir.pdf': 'https://drive.google.com/file/d/146f45PgPzsJOeHJdKFRCu5sureWMrjeA/view', '20240308145424331035.pdf': 'https://drive.google.com/file/d/1G9WokVx8FaNR_rZ5ctzpjpCDVEJ4h9wR/view', '2024031112285386257.pdf': 'https://drive.google.com/file/d/1frJjWLftY46QJ4jT-iTY14DjL0ZKnnTa/view', '2024031112270013514.pdf': 'https://drive.google.com/file/d/1B244RKpcT3jUHi6ol90ynDPLfaPjTCG2/view', '20240311110807312852.pdf': 'https://drive.google.com/file/d/1DYoAQdwWoVMPxPdh54VjTW8G-2Rsb2NH/view', '20240408162919919716-24.pdf': 'https://drive.google.com/file/d/1wwq326k99eCul1XLsFLQ4QpukoyxPv0z/view', '20240424125942498720 circular .pdf': 'https://drive.google.com/file/d/11UVA_jHTCYmsvIbm0avahI6yJArT0X_s/view', '2024031112295629059.pdf': 'https://drive.google.com/file/d/1H-ZvFTxk76_XfgubVe7r8NKrQojIwl--/view', '20240326143642338012.pdf': 'https://drive.google.com/file/d/1-xdAtgMEWxP8ir53VPLiK4Wm_UOuigpy/view', '20230818175031989129.pdf': 'https://drive.google.com/file/d/1mXV2ro59Skp9RVzZpvrYuCappaLn2lFo/view', '20230620162308836122.pdf': 'https://drive.google.com/file/d/1cSlYdT2ZfSvBT0lbaScclrDhgwu7Vzzc/view', '2024031112292147568.pdf': 'https://drive.google.com/file/d/11gL3nBMKS1DOcrdpCddSJTu6cLLJwK9g/view', '20230620162122072520.pdf': 'https://drive.google.com/file/d/16y2cyCKlHFcsyIOHWv30v76SMMfTC7pG/view', '2023031412023485669.pdf': 'https://drive.google.com/file/d/1Z5onEuQAun0j4ShT1d8eFOf73sL1ReXF/view', '20240408162804886815-24.pdf': 'https://drive.google.com/file/d/12Sqb6yfktk4GAjyPUC9DwqxRWpoeTZwB/view', '20240311105654870946.pdf': 'https://drive.google.com/file/d/1LMLMX4lS65qnFL8AspIkdtmL-ZT691pL/view', '20230809120652518926.pdf': 'https://drive.google.com/file/d/1d1fzJv38nsxASaO0jSqxTX0fsTXOTxhf/view', '20240311105345742144.pdf': 'https://drive.google.com/file/d/1RjJupIDbGkv61wrVIqX6T2xwqNGYUSM5/view', '20230608103554115216.pdf': 'https://drive.google.com/file/d/1wkMhbgyriBhGdwQwxhbybCg327Tc-3Km/view', '20240311105519509045.pdf': 'https://drive.google.com/file/d/1xw9VscIWTqci13-oGTZZNxkYImBfqGVi/view', '2024031112281723746.pdf': 'https://drive.google.com/file/d/1Rgci71daAs_GzWnT_B4SVp_UUk6GjHHW/view', '20240408162641208014-24.pdf': 'https://drive.google.com/file/d/10OpIxVnKMjrMl5nfJqLoQIxOR4UJKsCj/view', '20240424125758651818 circular .pdf': 'https://drive.google.com/file/d/1aTcGpgp44D_xMGS44ZWWU1jBWsQPw3vp/view', '20240326143601664211.pdf': 'https://drive.google.com/file/d/1xUDl5oWwiYAid01JMxzSAHasbO33pDqM/view', '20240311105930757047.pdf': 'https://drive.google.com/file/d/1YfXxxYjDtyCRYKHSRnki_rFqWMtvqk9o/view', '20240311110226071149.pdf': 'https://drive.google.com/file/d/1U11buH6z0mCJOgXybhuffU6wjZVgs-H9/view', '20240311110146983748.pdf': 'https://drive.google.com/file/d/1CwpG1aEEPig2MqTXpbKuFqU9UhwOMj84/view', '20230818175321380332.pdf': 'https://drive.google.com/file/d/194RzTCyF4SiLky36gVJotJLpdEwUpuTZ/view', '20240308145511982336.pdf': 'https://drive.google.com/file/d/191jVAgdC873VLR-LChz3v7lwOTLS3v8d/view', '2024031112250190381.pdf': 'https://drive.google.com/file/d/1qmMJcCvi7kQHWmiIIQ3aHaOHL1wYaJav/view', '20240308145550674737.pdf': 'https://drive.google.com/file/d/1iiL5GrCUA1RQv44ZM2XpnOEXiCD9tRXP/view', '20230811175634192128 CIRCULAR.pdf': 'https://drive.google.com/file/d/1S8Hh-NkWZNIx53zmjP-jWXAEPXes_ZnR/view', '2024031112274185425.pdf': 'https://drive.google.com/file/d/1i4j4UhTc1bL-ei6j3bO3eI_Lw1TEFFOo/view', '20240326143500614910.pdf': 'https://drive.google.com/file/d/1BKdTl-u32a1qRB8ofNpCEGyPBEHkcfQt/view', '202307271812041149circular24.pdf': 'https://drive.google.com/file/d/1YT1WlXQDFEHqscrV7Kk9hywqV7mWvDlF/view', '2024031112262603163.pdf': 'https://drive.google.com/file/d/1pX3_xhu6_wTJuN2vtVy1AoLBE360QHAB/view', '20230327132153939511 cr.pdf': 'https://drive.google.com/file/d/1d1sm2aSiPyMt7MfztXLSVrDG-W2cqOc-/view', '20230818175109700630.pdf': 'https://drive.google.com/file/d/1uSCnv3mPEDKfzzrtpeHF7v4qxyV0qqZy/view', '20240311105246799243.pdf': 'https://drive.google.com/file/d/1ZPhYaCB9W5E6WQvrjPYhUTkuzTxQxXLG/view', '20240311105158971042.pdf': 'https://drive.google.com/file/d/1Aog7XM2P2W1gZAC2P0yOlOqYZem3LEG-/view', '20230620162033965119.pdf': 'https://drive.google.com/file/d/19t-Nt_Ay3mgkKIP-O_6idbl2B0wjLXlQ/view', '20230202173623542302.pdf': 'https://drive.google.com/file/d/1oCp7SMoXtirHhNq8GW5pUqrfv7RmJOXk/view', '20230818175201505231.pdf': 'https://drive.google.com/file/d/1BIZtFZk7PTwwrb66cmREgHNHjbMbPvMH/view', '20240408163143441417-24.pdf': 'https://drive.google.com/file/d/1fevh8r8IhwfJLpcP4-ExAhiwMKOhaDK2/view', '20230710170831306523 cir.pdf': 'https://drive.google.com/file/d/1xW1Fwk48-GXKwlHGngqEhUiGadcdR7Gd/view', '20240424125841871319 circular .pdf': 'https://drive.google.com/file/d/14BEMPMRZxtOWXPH8-CBcgmQa3dhqpGwo/view', '20240308145942655240.pdf': 'https://drive.google.com/file/d/1PhD-RUwuyTDa4XGsps1B8X2DBukE0IiC/view', '2024031112254720192.pdf': 'https://drive.google.com/file/d/1uW6Bong5vuPWqyZaXRNRgqD1k_62MVhP/view', '20230411162001136613   2023.pdf': 'https://drive.google.com/file/d/1_ZgqHJFh0UpHnUY9ww13SkP6kgyAwZUL/view', '20230809112846205725.pdf': 'https://drive.google.com/file/d/1Q-8tRvS2bJiHSXQrPYrkW9UlSmuxPdjL/view', '20240311110301135950.pdf': 'https://drive.google.com/file/d/1eiZpKPbUJW2SJA7Swc-k0eVQuyBMW98j/view', '20240522113101126924 cir.pdf': 'https://drive.google.com/file/d/13ZLu2-hDrsFq52t7B7buVSlFm6ZIrpuk/view', '20240308145829679839.pdf': 'https://drive.google.com/file/d/1fTgrx31pCXcUoP6WHq2u2KvQyCMzbuj6/view', '20240311110708508851.pdf': 'https://drive.google.com/file/d/1VMAEv2T-qOmhy89LgHg0J62np3pOFTKx/view', '20240308145746132838.pdf': 'https://drive.google.com/file/d/108z5KIrCIgfe8Ed9DpalQxnVRXeVg2YN/view'}
                
                nodes = st.session_state.retriever.retrieve(QueryBundle(user_query))

                # Extract document paths and names
                doc_names = []
                for i in nodes[:min(5, len(nodes))]:  # Limit to 5 relevant documents
                    doc_names.append(i.node.metadata['file_name'])

                def unique_values_ordered(lst):
                    seen = set()
                    unique_lst = []
                    for item in lst:
                        if item not in seen:
                            unique_lst.append(item)
                            seen.add(item)
                    return unique_lst

                doc_names_unique = unique_values_ordered(doc_names)

                # Use unique-doc-names in RAG Chain
                text = ""
                for doc_name in doc_names_unique:
                    file_path = r"corrected-english-circulars"
                    with open(f"{file_path}/{doc_name}", 'r', encoding='utf-8') as f:
                        text += f.read()

                text_file_path = "Dynamic.txt"
                with open(text_file_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text)

                loader = TextLoader('Dynamic.txt')
                the_text = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(the_text)
                
                # Clear out the existing database directory if it exists
                chroma_persist_directory = "chroma_db_2"
                if os.path.exists(chroma_persist_directory):
                    shutil.rmtree(chroma_persist_directory)

                # Create a new Chroma database from the documents using OpenAI embeddings
                vectorstore = Chroma.from_documents(
                chunks,
                HuggingFaceEmbeddings(),
                persist_directory=chroma_persist_directory
                )
                
                retriever2 = vectorstore.as_retriever()

                rag_template = """You are a chatbot trained on circulars and legal documents of UP Police. Users will ask you questions related to laws and procedures mentioned in the circulars. If you don't know the answer, just say that "I don't know".
                Don't jump to conclusions. Support your answer with proper reasoning always.
                DO NOT USE ANY PRE-EXISTING KNOWLEDGE!!! Only use the knowledge I have provided you with.
                DO NOT HALLUCINATE !!!
                Context: {context}
                Question: {question}
                Answer:
                """
                def append_comprehensive_answer(question):
                    return f"{question} Give a comprehensive answer."
                llm2 = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name='mixtral-8x7b-32768'  # Chat-specific LLM
                )
                rag_prompt = ChatPromptTemplate.from_template(rag_template)
                rag_chain = (
                    {"context": retriever2, "question": (RunnablePassthrough() | append_comprehensive_answer)}
                    | rag_prompt
                    | llm2
                    | StrOutputParser()
                )

                response = rag_chain.invoke(user_query)

                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                chromadb.api.client.SharedSystemClient.clear_system_cache()
            
            st.markdown("### Related Documents:")
            if doc_names_unique:
                for doc_name in doc_names_unique:
                    real_file_name = doc_name.replace('.txt', '.pdf')
                    st.write(f"- {doc_name}, {dictionary[real_file_name]}")
            else:
                st.write("No relevant documents found.")
