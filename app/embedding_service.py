"""Day7 Embedding 服务：把文档文本和用户问题转换为向量。"""

from functools import lru_cache
from typing import Any

from app.config import EmbeddingSettings, get_embedding_settings


class EmbeddingServiceError(RuntimeError):
    """表示 Embedding 依赖、模型加载或文本向量化失败。"""


class SentenceTransformerEmbeddingService:
    """使用 Sentence Transformers 在本地计算文本向量。"""

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        """保存配置；模型延迟到第一次向量化时才加载。"""
        self.settings = settings or get_embedding_settings()
        # 模型等到第一次真正需要向量化时才加载。这样做的好处是：如果程序启动后根本没用到知识库功能，就不会浪费时间和内存去加载模型。
        self._model: Any | None = None  # self._model 先设为 None，表示模型还没有加载

    def _get_model(self) -> Any:
        """首次调用时加载模型，之后复用同一个模型对象以减少耗时。"""
        if self._model is not None:  
            return self._model  # 下一次调用时直接复用，不再重新加载；这叫“延迟加载+缓存”

        try:
            # 延迟导入保证未使用知识库接口时，不影响原有 FastAPI 功能。
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingServiceError(
                "缺少 sentence-transformers，请先执行 pip install -r requirements.txt。"
            ) from error

        try:
            self._model = SentenceTransformer(self.settings.model_name)
        except Exception as error:
            raise EmbeddingServiceError(
                f"Embedding 模型加载失败：{self.settings.model_name}；{error}"
            ) from error
        return self._model

    def _encode(self, texts: list[str], method_name: str) -> list[list[float]]:
        """调用指定编码方法，并把 NumPy 数组转换成普通 Python 浮点列表。"""
        if not texts:
            return []

        normalized_texts = [text.strip() for text in texts]  # 去掉每条文本首尾空格
        if any(not text for text in normalized_texts):
            raise EmbeddingServiceError("不能对空文本计算 Embedding。")

        model = self._get_model()  # 获取模型，触发懒加载。
        # 这是一个兼容性设计；新版 Sentence Transformers 区分查询encode_query和文档encode_document ；旧版没有时回退到 encode。
        encode_method = getattr(model, method_name, model.encode)
        try:
            vectors = encode_method(
                normalized_texts,
                batch_size=self.settings.batch_size,  # 一次处理多少条，从配置读取
                normalize_embeddings=True,  # 对向量做归一化，让后续用余弦相似度检索时更准确
                show_progress_bar=False,  # 不显示进度条，避免日志污染
            )  # 调用编码方法，传入三个参数
        except Exception as error:
            raise EmbeddingServiceError(f"文本向量化失败：{error}") from error

        raw_vectors = vectors.tolist() if hasattr(vectors, "tolist") else vectors  # 把 NumPy 数组转成 Python 列表
        return [[float(value) for value in vector] for vector in raw_vectors]  # 把每个向量里的每个元素都转成 Python 的 float，确保类型干净

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量把知识库文本块转换为归一化向量。"""
        return self._encode(texts, "encode_document")  # 返回一个向量列表，每个向量对应一条文本

    def embed_query(self, query: str) -> list[float]:
        """把一条用户问题转换为查询向量。"""
        vectors = self._encode([query], "encode_query")
        return vectors[0]  # 因为只传了一条，返回的列表只有一个元素，所以取 [0] 返回单个向量


@lru_cache(maxsize=1)
def get_embedding_service() -> SentenceTransformerEmbeddingService:
    """返回进程内共享的 Embedding 服务，避免重复加载模型。"""
    return SentenceTransformerEmbeddingService()
