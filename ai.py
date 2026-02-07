import asyncio
import sys
import os
import pyaudio
import warnings
import datetime
import re
from google import genai
from google.genai import types

# Suppress warnings
warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
def load_env_file_var(key_name, env_path=".env"):
    if not os.path.exists(env_path):
        return None
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() == key_name:
                    return value.strip().strip('"').strip("'")
    except Exception:
        return None
    return None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or load_env_file_var("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"

# Audio Settings
CHUNK_SIZE = 512
SAMPLE_RATE = 16000     # Input rate (Mic)
OUTPUT_RATE = 24000     # Output rate (Gemini usually sends 24k)
SILENCE_THRESHOLD = 600 # Amplitude threshold for VAD
MIN_SPEECH_FRAMES = 3   # Debounce: frames to consider "speech"
MIN_SILENCE_FRAMES = 15 # Debounce: frames to consider "silence"
# =================================================

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    GREY = '\033[90m'

def log(category, message, verbose=True):
    if verbose:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        sys.stdout.write(f"\r{Colors.GREY}[{timestamp}] [{category}] {message}{Colors.RESET}\n")

class SessionLogger:
    def __init__(self):
        if not os.path.exists("context"):
            os.makedirs("context")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join("context", f"session_{timestamp}.txt")
        self.current_turn_text = [] # Buffer for the current AI response
        
        # Create file with header
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(f"--- Gemini Session {timestamp} ---\n")
        
        log("FILE", f"Session logging to: {self.filename}")

    def clean_text(self, text):
        """Removes markdown, thinking traces, and acting cues."""
        # Remove bold/italic markers
        text = text.replace('**', '').replace('*', '')
        # Remove text inside parenthesis (often acting instructions)
        text = re.sub(r'\(.*?\)', '', text)
        # Remove text inside brackets
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()

    def is_internal_monologue(self, text):
        """Detects if the text is AI 'thinking' or 'planning' instead of speaking."""
        text_lower = text.lower()
        
        # Immediate rejection of known "thought" patterns
        bad_starts = [
            "initiating", "i've begun", "i have begun", "i am aiming", 
            "the goal is", "i realized", "i've received", "response:", 
            "thinking:", "analysis:", "identifying"
        ]
        if any(text_lower.startswith(x) for x in bad_starts):
            return True
            
        # Rejection of first-person process descriptions
        if "i'm prioritizing" in text_lower or "i need to pinpoint" in text_lower:
            return True
            
        return False

    def buffer_ai_text(self, text):
        """Accumulates AI text chunks."""
        cleaned = self.clean_text(text)
        
        # Apply the monologue filter
        if self.is_internal_monologue(cleaned):
            # If we detect monologue, we skip buffering this chunk
            return

        if cleaned:
            self.current_turn_text.append(cleaned)

    def commit_turn(self, role="Gemini"):
        """Writes the buffered text to file."""
        if not self.current_turn_text and role == "Gemini":
            return

        full_text = " ".join(self.current_turn_text).strip()
        
        # Additional filter for empty or junk lines
        if not full_text or len(full_text) < 2:
            self.current_turn_text = []
            return

        # Double check the full text isn't just a monologue block
        if self.is_internal_monologue(full_text):
            self.current_turn_text = []
            return

        try:
            with open(self.filename, "a", encoding="utf-8", buffering=1) as f:
                f.write(f"{role}: {full_text}\n")
                f.flush()
                os.fsync(f.fileno()) # Force write to disk
            
            # log("FILE", f"Saved {role} turn ({len(full_text)} chars)")
        except Exception as e:
            log("ERROR", f"Could not write to file: {e}")
        
        self.current_turn_text = [] # Clear buffer

    def log_user_event(self):
        """Logs that the user spoke (since we don't have local STT)."""
        try:
            with open(self.filename, "a", encoding="utf-8", buffering=1) as f:
                f.write(f"User: [Audio Input]\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            log("ERROR", f"File write error: {e}")

    def get_recent_history(self, lines=10):
        """Reads recent history for context injection."""
        if not os.path.exists(self.filename): return ""
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                content = f.readlines()
            return "".join(content[-lines:])
        except:
            return ""

class AudioAgent:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.logger = SessionLogger()
        self.send_queue = asyncio.Queue()
        self.receive_queue = asyncio.Queue()
        self.running = True
        
        # VAD State
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0

    async def mic_stream(self):
        stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, 
            input=True, frames_per_buffer=CHUNK_SIZE
        )
        log("AUDIO", "Microphone listening...")
        
        loop = asyncio.get_running_loop()
        
        while self.running:
            try:
                # Read audio in non-blocking way
                data = await loop.run_in_executor(None, lambda: stream.read(CHUNK_SIZE, exception_on_overflow=False))
                
                # --- VAD Logic (Voice Activity Detection) ---
                # Calculate simple peak amplitude
                peak = 0
                if len(data) > 0:
                    # Check every 32nd sample to save CPU
                    for i in range(0, len(data), 32):
                        val = abs(int.from_bytes(data[i:i+2], byteorder='little', signed=True))
                        if val > peak: peak = val
                
                if peak > SILENCE_THRESHOLD:
                    self.speech_frames += 1
                    self.silence_frames = 0
                    if self.speech_frames == MIN_SPEECH_FRAMES:
                        # User started speaking logic
                        pass 
                else:
                    self.silence_frames += 1
                    self.speech_frames = 0
                    if self.silence_frames == MIN_SILENCE_FRAMES and self.is_speaking:
                        # User stopped speaking logic
                        pass

                # Queue data for sending
                self.send_queue.put_nowait(data)

            except Exception as e:
                log("ERROR", f"Mic error: {e}")
                await asyncio.sleep(0.1)

        stream.stop_stream()
        stream.close()

    async def speaker_stream(self):
        stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=OUTPUT_RATE, output=True
        )
        log("AUDIO", "Speaker ready.")
        
        while self.running:
            data = await self.receive_queue.get()
            if data:
                await asyncio.to_thread(stream.write, data)

        stream.stop_stream()
        stream.close()

    async def send_to_api(self, session):
        while self.running:
            chunk = await self.send_queue.get()
            await session.send(input={"data": chunk, "mime_type": "audio/pcm"}, end_of_turn=False)

    def cleanup(self):
        self.running = False
        self.audio.terminate()

async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Colors.CYAN}--- GEMINI VOICE AGENT (Saving Fixed) ---{Colors.RESET}")
    
    if not GEMINI_API_KEY:
        print(f"{Colors.RED}CRITICAL: GEMINI_API_KEY is missing from environment variables.{Colors.RESET}")
        return

    agent = AudioAgent()
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})

    # --- System Prompt Configuration ---
    # We strictly tell it NOT to output thought processes
    def get_config():
        history = agent.logger.get_recent_history()
        
        system_instructions = """
        You are 'Puck', a friendly, conversational voice assistant.
        
        STRICT RESPONSE GUIDELINES:
        1. OUTPUT ONLY THE SPOKEN RESPONSE. 
        2. NEVER explain your response or your thought process.
        3. NEVER start with phrases like "I've begun", "I realized", "My goal is", "Initiating".
        4. If the user says "Hello", simply reply "Hello! How can I help?". Do not narrate your greeting.
        5. Speak plainly and directly to the user.
        """
        
        if history:
            system_instructions += f"\n\nCONTEXT FROM PREVIOUS TURN:\n{history}\n[System]: Connection resumed."

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"], # We request Audio, but text transcripts usually accompany it
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=system_instructions)])
        )

    # Start background audio tasks
    asyncio.create_task(agent.mic_stream())
    asyncio.create_task(agent.speaker_stream())

    # Main Connection Loop
    while agent.running:
        try:
            print(f"{Colors.YELLOW}Connecting to Gemini...{Colors.RESET}")
            config = get_config()
            
            async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
                print(f"{Colors.GREEN}✅ Connected! Speak now.{Colors.RESET}")
                
                # Start sender task
                send_task = asyncio.create_task(agent.send_to_api(session))
                
                async for message in session.receive():
                    if message.server_content is None:
                        continue

                    # Handle Interruptions
                    if message.server_content.interrupted:
                        print(f"\n{Colors.RED}[Interrupted]{Colors.RESET}")
                        while not agent.receive_queue.empty():
                            try: agent.receive_queue.get_nowait()
                            except: break
                        agent.logger.current_turn_text = [] # Drop interrupted text
                        continue

                    # Handle Model Turn (Text & Audio)
                    model_turn = message.server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            # 1. Handle Text (for logging and display)
                            if part.text:
                                text = part.text
                                # Buffer it for file saving
                                agent.logger.buffer_ai_text(text)
                                # Print it in real-time if it's not internal monologue
                                if not agent.logger.is_internal_monologue(text):
                                    print(text, end="", flush=True)

                            # 2. Handle Audio (for playback)
                            if part.inline_data:
                                await agent.receive_queue.put(part.inline_data.data)

                    # Handle Turn Completion (Save to file now)
                    if message.server_content.turn_complete:
                        print("") # New line on console
                        agent.logger.commit_turn("Gemini")
                        agent.logger.log_user_event() # Mark that user will speak next
                
                # If loop ends naturally
                send_task.cancel()

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"{Colors.RED}Connection error: {e}. Reconnecting...{Colors.RESET}")
            await asyncio.sleep(2)

    agent.cleanup()

if __name__ == "__main__":
    try:
        # Windows specific event loop policy
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")