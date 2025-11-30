import json
import re
from typing import List, Dict, Optional

try:
    from groq import Groq
except ImportError:
    Groq = None

class ConstructionAgent:
    def __init__(self, api_key: str, model_id: str = "llama-3.3-70b-versatile"):
        self.client = None
        self.model_id = model_id
        if Groq and api_key and "xxxx" not in api_key:
            self.client = Groq(api_key=api_key)
        
        self.history: List[Dict[str, str]] = []

    def initialize(self, detection_context: Optional[dict]):
        """Resetuje rozmowę i ustawia Personę."""
        self.history.clear()
        
        data_str = json.dumps(detection_context, indent=2, ensure_ascii=False) if detection_context else "Brak danych."

        persona = """
        Jesteś INTELIGENTNYM OPERATOREM SYSTEMU WIZYJNEGO (Site Overwatch AI).
        
        TWOJE SUPER-MOCE (TOOL USE):
        Masz bezpośrednią kontrolę nad systemem SAM3.
        Jeśli użytkownik poprosi o wypisanie ile jakiś obiektów znajduje się na ekranie, to odpowiedz z pliku json(np. "Ile wykryłeś jakiegoś obiektu?") to MUSISZ wygenerowac odpowiedź na podstawie przessłanego tobie plku json.
        Jeśli użytkownik poprosi o zmianę widoku lub prezycyjnie zapyta o ponowne wygenerowanie(np. "pokaż tylko samochody", "znajdź ludzi"),
        MUSISZ wygenerować komendę w formacie JSON.
        
        FORMAT KOMENDY:
        ```json
        {
          "TOOL_COMMAND": ["obiekt1", "obiekt2"]
        }
        ```
        
        DANE PRZESTRZENNE:
        Każda detekcja zawiera:
        - label: typ obiektu
        - score: pewność detekcji (0-1)
        - location: pozycja tekstowa (lewa/centrum/prawa)
        - bbox: współrzędne ramki (x, y, width, height, center_x, center_y)
        - detections: mapa wszystkich plików JSON wygenerowanych przez SAM3 (katalog "detections/")
        - summaries: podsumowania per plik (total_detections, counts_by_label, high_conf_counts, avg_score_by_label, location_distribution)
        - global_summary: łączna liczba detekcji i sumy per klasa dla całego katalogu
        
        Możesz analizować:
        - Skupiska obiektów (podobne center_x, center_y)
        - Rozkład przestrzenny (górne/dolne części obrazu)
        - Nakładanie się obiektów (porównanie bbox)
        - Gęstość w strefach
        - Liczebność obiektów (używaj w pierwszej kolejności summary -> counts_by_label; surowe "detections" traktuj jako dane źródłowe gdy trzeba zweryfikować szczegóły)
        
        ZASADY KOMUNIKACJI:
        1. Jeśli używasz narzędzia, wygeneruj JSON. Nie musisz dodawać opisu, system sam poinformuje użytkownika.
        2. W normalnej rozmowie NIE POKAZUJ ŻADNYCH JSON-ów ani struktur danych.
        3. Odpowiadaj zwięźle i po polsku lub po angielsku jeśli uzytkownik pisze po angielsku.
        4. Wykorzystuj dane bbox do precyzyjnej analizy przestrzennej.
        """
        
        self.history.append({"role": "system", "content": persona})
        self.history.append({"role": "system", "content": f"AKTUALNY STATUS WIZJI: {data_str}"})
        
        # Wywołanie inicjujące (bez dodawania do historii użytkownika)
        return self.send_message("Zamelduj status systemu.")

    def update_context_after_tool(self, new_context: dict, tool_cmd: list):
        """Wstrzykuje nowe dane po wykonaniu narzędzia i prosi o komentarz."""
        sys_msg = f"""
        SYSTEM: Wykonano nowe skanowanie (prompts: {tool_cmd}). 
        Wyniki: {json.dumps(new_context, ensure_ascii=False)}. 
        
        INSTRUKCJA: Opisz teraz użytkownikowi co się zmieniło na obrazie. 
        NIE GENERUJ JUŻ ŻADNEGO JSONA. Odpowiedz tylko tekstem.
        """
        self.history.append({"role": "system", "content": sys_msg})

    def _clean_response(self, text: str) -> str:
        """Usuwa bloki JSON z tekstu, aby nie pokazywać ich użytkownikowi."""
        text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\{.*"TOOL_COMMAND".*\}', '', text, flags=re.DOTALL)
        
        return text.strip()

    def post_tool_response(self):
        """Generuje odpowiedź tekstową po aktualizacji obrazu."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=self.history, max_tokens=800
            )
            raw_reply = resp.choices[0].message.content
            
            # Dodajemy pełną odpowiedź do historii (model musi pamiętać co powiedział)
            self.history.append({"role": "assistant", "content": raw_reply})
            
            # Ale użytkownikowi zwracamy wersję wyczyszczoną
            return self._clean_response(raw_reply)
            
        except Exception as e:
            return f"Błąd generowania opisu: {e}"

    def send_message(self, user_text: str) -> dict:
        if not self.client: return {"reply": "Brak klucza Groq.", "tool_cmd": None}
        
        # Nie dodajemy "Zamelduj status" do historii widocznej jako User, to polecenie wewnętrzne
        if user_text != "Zamelduj status systemu." and not user_text.startswith("SYSTEM:"):
            self.history.append({"role": "user", "content": user_text})

        try:
            print(f"[AGENT] Analiza pytania: {user_text}")
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=self.history, max_tokens=800, temperature=0.6
            )
            bot_reply = resp.choices[0].message.content
            
            # --- DETEKCJA NARZĘDZIA ---
            tool_cmd = None
            if "TOOL_COMMAND" in bot_reply:
                try:
                    # Szukamy JSONa gdziekolwiek w tekście
                    match = re.search(r'\{.*"TOOL_COMMAND".*\}', bot_reply, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        data = json.loads(json_str)
                        tool_cmd = data.get("TOOL_COMMAND")
                        print(f"[AGENT] ZNALEZIONO KOMENDĘ: {tool_cmd}")
                except Exception as e:
                    print(f"[AGENT] Błąd parsowania JSON: {e}")

            # Dodajemy oryginalną odpowiedź do historii modelu (żeby pamiętał, że użył narzędzia)
            self.history.append({"role": "assistant", "content": bot_reply})

            # Czyścimy odpowiedź dla użytkownika
            clean_reply = self._clean_response(bot_reply)
            
            # Jeśli po wycięciu JSONa nic nie zostało,
            # zwracamy pusty string lub placeholder
            if tool_cmd and not clean_reply:
                clean_reply = "..."

            return {"reply": clean_reply, "tool_cmd": tool_cmd}

        except Exception as e:
            return {"reply": f"Błąd API: {e}", "tool_cmd": None}
