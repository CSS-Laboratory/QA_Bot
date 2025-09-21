"""Google Drive / Docs based knowledge source."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from qa_bot.knowledge.base import Document
from qa_bot.utils.text import normalize_text

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
except Exception:  # pragma: no cover - optional dependency
    Credentials = None
    build = None


@dataclass(slots=True)
class GoogleDocInfo:
    doc_id: str
    name: str
    url: str


class GoogleDriveKnowledgeSource:
    """Load Google Docs contained in a Drive folder."""

    def __init__(
        self,
        *,
        service_account_json: Path,
        folder_id: str,
        faq_master_doc_id: str | None = None,
        include_faq_master: bool = True,
    ) -> None:
        self.service_account_json = Path(service_account_json)
        self.folder_id = folder_id
        self.faq_master_doc_id = faq_master_doc_id
        self.include_faq_master = include_faq_master
        if not self.service_account_json.exists():
            raise FileNotFoundError(f"Service account JSON が見つかりません: {service_account_json}")
        if Credentials is None or build is None:
            raise ImportError(
                "google-api-python-client がインストールされていません。"
            )

    def load_documents(self) -> Iterable[Document]:
        credentials = Credentials.from_service_account_file(
            str(self.service_account_json),
            scopes=[
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/documents.readonly",
            ],
        )
        drive_service = build("drive", "v3", credentials=credentials)
        docs_service = build("docs", "v1", credentials=credentials)
        files = self._fetch_files(drive_service)
        documents: List[Document] = []
        for info in files:
            body = docs_service.documents().get(documentId=info.doc_id).execute()
            content = self._extract_text(body)
            documents.append(
                Document(
                    doc_id=info.doc_id,
                    title=info.name,
                    content=normalize_text(content),
                    url=info.url,
                )
            )
        if self.include_faq_master and self.faq_master_doc_id:
            body = docs_service.documents().get(documentId=self.faq_master_doc_id).execute()
            content = self._extract_text(body)
            documents.append(
                Document(
                    doc_id=self.faq_master_doc_id,
                    title="FAQ Master Doc",
                    content=normalize_text(content),
                    url=f"https://docs.google.com/document/d/{self.faq_master_doc_id}",
                )
            )
        return documents

    def _fetch_files(self, drive_service) -> List[GoogleDocInfo]:  # type: ignore[override]
        query = f"'{self.folder_id}' in parents and mimeType='application/vnd.google-apps.document'"
        response = (
            drive_service.files()
            .list(q=query, fields="files(id, name)")
            .execute()
        )
        files = response.get("files", [])
        infos = [
            GoogleDocInfo(
                doc_id=item["id"],
                name=item["name"],
                url=f"https://docs.google.com/document/d/{item['id']}",
            )
            for item in files
        ]
        return infos

    @staticmethod
    def _extract_text(doc_body: dict) -> str:
        texts: List[str] = []
        for element in doc_body.get("body", {}).get("content", []):
            paragraph = element.get("paragraph")
            if not paragraph:
                continue
            for el in paragraph.get("elements", []):
                text_run = el.get("textRun")
                if text_run and "content" in text_run:
                    texts.append(text_run["content"])
        return "".join(texts)
