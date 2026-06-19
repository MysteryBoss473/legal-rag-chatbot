"""Client Groq pour les inférences LLM."""

import logging
from typing import List, Dict, Generator, Optional

from groq import Groq
from groq.types.chat import ChatCompletionChunk

from app.config import get_settings

logger = logging.getLogger(__name__)


class GroqLLMClient:
    """Client pour interagir avec l'API Groq.

    Utilise le SDK officiel groq-python avec gestion des erreurs
    et support du streaming.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None

    def _get_client(self) -> Groq:
        """Initialise et retourne le client Groq."""
        if self._client is None:
            self._client = Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.settings.groq_timeout,
            )
            logger.info("✅ Client Groq initialisé")
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Génère une réponse avec le modèle Groq.

        Args:
            messages: Liste de messages au format OpenAI
            stream: Active le streaming de la réponse
            temperature: Température de génération (0.0-1.0)
            max_tokens: Nombre max de tokens

        Yields:
            Fragments de texte en streaming
        """
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.settings.groq_model,
                messages=messages,
                temperature=temperature or self.settings.groq_temperature,
                max_tokens=max_tokens or self.settings.groq_max_tokens,
                stream=stream,
            )

            if stream:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                content = response.choices[0].message.content
                yield content

        except Exception as e:
            logger.error(f"Erreur Groq API: {e}")
            yield f"[Erreur: {str(e)}]"

    def generate_non_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Génère une réponse complète (non-streaming).

        Utile pour les opérations internes (résumé, classification, etc.)
        """
        result = ""
        for chunk in self.generate(
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            result += chunk
        return result


# Singleton
_groq_client = None


def get_groq_client() -> GroqLLMClient:
    """Retourne une instance singleton du client Groq."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqLLMClient()
    return _groq_client
