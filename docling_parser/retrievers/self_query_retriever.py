from langchain_core.vectorstores import VectorStore
from langchain.chains.query_constructor.base import AttributeInfo # type: ignore
from langchain.retrievers.self_query.base import SelfQueryRetriever

from dto.rag.retriever import RetrieverParams


class CustomSelfQueryRetriever:
    """
    A self-querying retriever generates and applies its own structured queries from natural language
    inputs using an LLM chain. It then uses these queries on its VectorStore to perform
    semantic similarity comparisons and metadata filtering for more precise search results.
    """

    def __init__(
        self,
        vectore_store: VectorStore,
        llm: str,
        retriever_params: RetrieverParams,
        filter_folder: str = None, # type: ignore
        metadata_field_info: list[AttributeInfo] = [
            AttributeInfo(name="category", description="The category title in the table of content in the document, here is the list of categories: ADP,ADP - CHANGEMENT RESPONSABLE,ADP - CORRECTION DONNEES,ADP-EMBAUCHE,ADP - EMBAUCHE - ALTERNANT,ADP - EMBAUCHE - AUXILIAIRE DE VACANCES,ADP - EMBAUCHE- CDI,ADP - EMBAUCHE HORS PARCOURS,ADP - EMBAUCHE HORS PARCOURS AUXILIAIRE DE VACANCES,ADP - EMBAUCHE - STAGE,ADP - GESTION PARCOURS - AUXILIAIRE DE VACANCES,ADP - GUIDE DE NAVIGATION,ADP - RECHERCHE AVANCEE,ADP - REEMBAUCHE - AUXILIAIRE DE VACANCES,ADP-SORTIE,ADP - SORTIE - ALL,ADP-VIE,ADP - VIE - CHANGE CONTRAT,ADP - VIE - CHANGE EMPLOI,ADP - VIE - CHANGE INFO PERSO,ADP-VIE-CHANGEMENT UO,ADP - VIE - CHANGE SOCIETE,ADP - VIE MAJ TITRE DE SEJOUR,ADP - VIE - MOBILITE FONCTIONNELLE,ADP - VIE - MOBILITE TEMPORAIRE,ADP-VIE-MODIF-PASSAGE-FORFAIT,ADP-VIE-MODIF-TEMPS-HEBDO,ADP - VIE- MUTATION MOB GEO,ADP - VIE - PVO,ADP - VIE - RENOUVELLEMENT INTERIM,ADP - VIE - SELF-SERVICE,Home Page", type="string"),
        ],
    ) -> None:
        self.vector_store = vectore_store
        self.llm = llm
        self.filter = filter_folder
        self.metadata_field_info = metadata_field_info
        self.document_content_description = "Content for HR and recruitment"
        self.top_k = retriever_params.top_k
        self.score_threshold = retriever_params.score_threshold

    def get_retriever(self) -> SelfQueryRetriever:
        search_kwargs = {"score_threshold": self.score_threshold, "k": self.top_k}

        if self.filter:
            search_kwargs["filter"] = {"file_folder": self.filter}

        return SelfQueryRetriever.from_llm(
            self.llm, # type: ignore
            self.vector_store,
            self.document_content_description,
            self.metadata_field_info,
            search_kwargs=search_kwargs,
        )
