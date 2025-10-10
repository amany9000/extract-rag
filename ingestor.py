
import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_qdrant import Qdrant
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import langextract as lx
from langextract.providers import ollama
import textwrap


def extract_with_langextract(documents: List[Document]) -> List[Document]:
    prompt = textwrap.dedent("""\
        Extract label of the text. The 8 possible labels are: Macroeconomics, 
        Government-Work, Currencies, Energy, Commodities, Agriculture, Livestock and 
        Corporate-Finance. Do not create new labels. Use exact text for extractions.""")

    examples = [
        lx.data.ExampleData(
            text="newid: 3001, date: 9-MAR-1987 04:58:41.12, title: \
            U.K. MONEY MARKET SHORTAGE FORECAST AT 250 MLN STG, \
            dateline: LONDON, March 9 -, body:  The Bank of England said it forecast a \
            shortage of around 250 mln stg in the money market today. \
            Among the factors affecting liquidity, it said bills \
            maturing in official hands and the treasury bill take-up would \
            drain around 1.02 billion stg while below target bankers' \
            balances would take out a further 140 mln. \
            Against this, a fall in the note circulation would add 345 \
            mln stg and the net effect of exchequer transactions would be \
            an inflow of some 545 mln stg, the Bank added. \
            REUTER",
            extractions=[
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Macroeconomics",
                ),
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Currencies",
                ),
            ]
        ),
        lx.data.ExampleData(
            text="newid: 3003, date: 9-MAR-1987 05:03:38.51, \
                title: AMOCO REPORTS SOUTH CHINA SEA OIL FIND,  \
                dateline: PEKING, March 9 -, body:  The U.S. <Amoco Petroleum Corp> has \
                reported an offshore oil find at its Pearl River basin \
                concession in the South China Sea, the New China News Agency said. \
                It said the Liu Hua 11-1-1 A well produced at around 2,240 \
                barrels per day at a depth of 305 metres. \
                The news agency said Amoco plans to drill a second well in \
                the area this year, but gave no further details. \
                REUTER",
            extractions=[
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Energy",
                ),
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Macroeconomics",
                ),
            ]
        ),
        lx.data.ExampleData(
            text="newid: 3345, date: 9-MAR-1987 16:00:05.45, \
                title: SENSORMATIC <SNSR> UPS CHECKROBOT <CKRB> STAKE, \
                dateline: DEERFIELD BEACH, Fla., March 9 -, \
                body:  Sensormatic Electronics Corp said it upped its investment \
                in CheckRobot Inc in the form of 2.5 mln dlrs of convertible \
                preferred stock, raising its stake in CheckRobot to 42 pct from \
                37 pct on a fully diluted basis. \
                Reuter",
            extractions=[
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Corporate-Finance",
                ),
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Currencies",
                ),
            ]
        ),
        lx.data.ExampleData(
            text="newid: 3010, date: 9-MAR-1987 08:13:36.29, \
                title: U.S. TO ALLOW TEMPORARY IMPORTS OF S.A. URANIUM, \
                dateline: WASHINGTON, March 9 -, body:  The Treasury Department said \
                it would temporarily permit imports of South African uranium ore and \
                oxide pending clarification of anti-apartheid sanctions laws \
                passed by Congress last fall. The decision was announced late Friday. \
                It applies, until July 1, to uranium ore and oxide imported into the \
                U.S. for processing and re-export to third countries. The Treasury said \
                it took the action because it felt that when Congress passed the \
                comprehensive South African sanctions bill last fall over President Reagan's \
                veto it had not intended to hurt U.S. industry. In addition, the Treasury \
                said it would permit U.S.-made goods to be imported temporarily from \
                South African state-controlled organizations for repair or servicing.\
                Reuter",
            extractions=[
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Energy",
                ),
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Commodities",
                ),
            ]
        ),
        lx.data.ExampleData(
            text="newid: 3330, date: 9-MAR-1987 15:31:05.90, \
                title: USDA RAISES SOVIET GRAIN IMPORT ESTIMATE, \
                dateline: WASHINGTON, March 9 -, body:  The U.S. Agriculture Department \
                increased its estimate of Soviet 1986/87 grain imports to 26 \
                mln tonnes from last month's projection of 23 mln tonnes. \
                    In its monthly USSR Grain Situation and Outlook, USDA said \
                the increase reflected the return of the Soviet Union to the \
                U.S. corn market and continued purchases of both wheat and \
                coarse grain from other major suppliers. USSR wheat imports were \
                projected at 15 mln tonnes, up one mln from last month's estimate \
                and 700,000 tonnes below the preliminary 1985/86 figure. \
                Soviet grain for feed use was estimated at a record 129 mln \
                tonnes. Record or near-record livestock inventories, along with \
                a dry fall which likely reduced late season pasturage, and a \
                cold winter have increased feed demand, USDA said. \
                USSR meat and egg production in January rose only slightly \
                from the previous January's level, while milk production \
                increased by nearly six pct. Unusually cold weather in January and \
                smaller increases in roughage supplies during 1986 than in 1985 \
                kept livestock production from expanding as much as it did a year earlier, \
                USDA said. \
                Reuter",
            extractions=[
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Agriculture",
                ),
                lx.data.Extraction(
                    extraction_class="labels",
                    extraction_text="Livestock",
                ),
            ]
        )
    ]

    for document in documents:
        result = lx.extract(
            text_or_documents=document.__str__(),
            prompt_description=prompt,
            examples=examples,
            model_id="qwen2.5:3b-instruct",
            model_url="http://localhost:11434",
            resolver_params={"format_handler": ollama.OLLAMA_FORMAT_HANDLER},
        )
        labels = list(dict.fromkeys([e.extraction_text for e in result.extractions]))
        print("labels", labels)
        document.metadata["filter"] = labels
        print("Document After", document)
    return documents
    

def process_docs(data_dir: str, db_dir: str, db_col: str):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=4)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=128,
        add_start_index=True,
    )

    p = Path(data_dir)
    documents = []
    for file in p.iterdir(): 
        if file.is_file():
            documents.extend(
                recursive_splitter.create_documents([file.read_text()])
            )
    print(documents)
    print("Chunking done")

    extract_with_langextract(documents)

    print("extraction done", documents[1:5])

    Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name=db_col,
        force_recreate=True
    )



data_dir = os.getenv("DATA_DIR") or "./docs"
db_dir = os.getenv("QDRANT_DIR") or "./db_docs"
db_col = os.getenv("QDRANT_COL") or "extract-rag.default"
process_docs(data_dir, db_dir, db_col)